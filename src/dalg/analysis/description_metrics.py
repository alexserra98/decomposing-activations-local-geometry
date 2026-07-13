from __future__ import annotations

import json
import math
import random
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

from dalg.analysis.cluster_labeling import ORFEO_MODEL, make_orfeo_client, parse_json_response


TARGET_OPEN = "<target>"
TARGET_CLOSE = "</target>"

DETECTION_SYSTEM_PROMPT = """You are evaluating descriptions of latent features in a language model.

Each example is one token inside its local text context. The target token is
wrapped in <target>...</target>. Judge whether the marked target token, not the
whole passage, is an instance of the feature description.

Return JSON only with this schema:
{"examples": [{"example_id": "<id>", "activates": true, "confidence": 0.0, "reason": "<short reason>"}], "summary": "<short summary>"}
"""

TOKEN_EMBEDDING_NOTE = (
    "Experimental: description vectors are mean-pooled sentence embeddings, while "
    "example vectors are hidden states at the marked target token from the same "
    "encoder. The cosine values live in one encoder space but should be treated "
    "as a rough diagnostic, not a calibrated activation-fit score."
)


def load_labeled_clusters(labels_path: str | Path) -> dict[str, Any]:
    """Load the cluster interpretation JSON written by the labeling pipeline."""
    labels_path = Path(labels_path)
    data = json.loads(labels_path.read_text())
    if "clusters" not in data:
        raise ValueError(f"{labels_path} is missing a top-level 'clusters' object")

    clusters: dict[int, dict[str, Any]] = {}
    for key, raw in data["clusters"].items():
        cluster = dict(raw)
        cluster_id = int(cluster.get("cluster_id", key))
        cluster["cluster_id"] = cluster_id
        cluster.setdefault("label", None)
        cluster.setdefault("description", None)
        cluster.setdefault("examples", [])
        cluster.setdefault("top_tokens", [])
        clusters[cluster_id] = cluster

    return {
        "metadata": data.get("metadata", {}),
        "clusters": clusters,
    }


def description_text(cluster: dict[str, Any]) -> str:
    """Return the text used as the feature description for judging/embedding."""
    parts: list[str] = []
    label = (cluster.get("label") or "").strip()
    description = (cluster.get("description") or "").strip()
    if label:
        parts.append(f"Label: {label}")
    if description and description != label:
        parts.append(f"Description: {description}")
    return "\n".join(parts).strip()


def _normalize_cluster_ids(
    clusters: dict[int, dict[str, Any]],
    cluster_ids: Sequence[int] | None,
    max_clusters: int | None,
) -> list[int]:
    if cluster_ids is None:
        out = sorted(clusters)
    else:
        out = [int(k) for k in cluster_ids]
    if max_clusters is not None:
        out = out[: int(max_clusters)]
    missing = [k for k in out if k not in clusters]
    if missing:
        raise ValueError(f"cluster ids not found in labels: {missing[:10]}")
    return out


def _example_payload(
    example: dict[str, Any],
    *,
    example_id: str,
    expected_activates: bool,
    source_cluster_id: int,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "expected_activates": bool(expected_activates),
        "source_cluster_id": int(source_cluster_id),
        "rank": example.get("rank"),
        "responsibility": example.get("responsibility"),
        "token": example.get("token"),
        "snippet": example.get("snippet"),
        "stream_position": example.get("stream_position"),
        "global_row": example.get("global_row"),
        "tok_pos": example.get("tok_pos"),
    }


def select_detection_examples(
    clusters: dict[int, dict[str, Any]],
    cluster_id: int,
    *,
    positive_examples: int = 8,
    negative_examples: int = 8,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Mix top examples from one feature with random examples from other features."""
    rng = random.Random(int(seed) + int(cluster_id) * 1_000_003)
    cluster = clusters[int(cluster_id)]

    positives = []
    for idx, example in enumerate(cluster.get("examples", [])[:positive_examples]):
        positives.append(
            _example_payload(
                example,
                example_id=f"c{cluster_id}_pos{idx}",
                expected_activates=True,
                source_cluster_id=int(cluster_id),
            )
        )

    negative_pool: list[tuple[int, dict[str, Any]]] = []
    for other_id, other in clusters.items():
        if int(other_id) == int(cluster_id):
            continue
        for example in other.get("examples", []):
            negative_pool.append((int(other_id), example))
    if negative_examples < len(negative_pool):
        negative_pool = rng.sample(negative_pool, negative_examples)
    else:
        rng.shuffle(negative_pool)

    negatives = []
    for idx, (other_id, example) in enumerate(negative_pool[:negative_examples]):
        negatives.append(
            _example_payload(
                example,
                example_id=f"c{cluster_id}_neg{idx}_from{other_id}",
                expected_activates=False,
                source_cluster_id=other_id,
            )
        )

    examples = positives + negatives
    rng.shuffle(examples)
    return examples


def build_detection_prompt(cluster: dict[str, Any], examples: Sequence[dict[str, Any]]) -> str:
    """Build the per-feature judge prompt for detection scoring."""
    text = description_text(cluster)
    top_tokens = ", ".join(
        f"{item.get('token')!r} x{item.get('count')}"
        for item in cluster.get("top_tokens", [])[:30]
    )
    if not top_tokens:
        top_tokens = "(not available)"

    lines = [
        f"Cluster id: {cluster['cluster_id']}",
        "Feature description:",
        text or "(missing description)",
        "",
        f"Top target tokens seen while labeling this feature: {top_tokens}",
        "",
        "For each example, decide whether the marked target token should activate this feature.",
        "The examples may include target tokens from unrelated features.",
        "",
        "Examples:",
    ]
    for ex in examples:
        lines.extend(
            [
                f"example_id: {ex['example_id']}",
                f"target_token: {ex.get('token')!r}",
                f"context: {ex.get('snippet')}",
                "",
            ]
        )
    lines.append("Return JSON only.")
    return "\n".join(lines)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "y", "1", "activate", "activates"}:
            return True
        if value in {"false", "no", "n", "0", "inactive", "does not activate"}:
            return False
    return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _classification_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("predicted_activates") is not None]
    tp = sum(1 for row in valid if row["expected_activates"] and row["predicted_activates"])
    tn = sum(1 for row in valid if not row["expected_activates"] and not row["predicted_activates"])
    fp = sum(1 for row in valid if not row["expected_activates"] and row["predicted_activates"])
    fn = sum(1 for row in valid if row["expected_activates"] and not row["predicted_activates"])
    n = len(valid)
    n_total = len(rows)

    accuracy = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    if recall is not None and specificity is not None:
        balanced_accuracy = 0.5 * (recall + specificity)
    else:
        balanced_accuracy = accuracy

    return {
        "n_examples": n_total,
        "n_parsed": n,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "detection_score": balanced_accuracy,
    }


def judge_one_cluster_detection(
    client: Any,
    cluster: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    *,
    model: str = ORFEO_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Ask the judge whether each marked target token matches one description."""
    if not description_text(cluster):
        rows = [{**ex, "predicted_activates": None, "confidence": None, "reason": "missing description"} for ex in examples]
        return {
            "description_text": "",
            "metrics": _classification_metrics(rows),
            "examples": rows,
            "summary": "missing description",
            "raw_response": None,
        }

    prompt = build_detection_prompt(cluster, examples)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DETECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    parsed = parse_json_response(content)

    by_id: dict[str, dict[str, Any]] = {}
    parsed_examples = parsed.get("examples")
    if isinstance(parsed_examples, list):
        for item in parsed_examples:
            if isinstance(item, dict) and item.get("example_id") is not None:
                by_id[str(item["example_id"])] = item

    rows: list[dict[str, Any]] = []
    for ex in examples:
        item = by_id.get(str(ex["example_id"]), {})
        predicted = _coerce_bool(item.get("activates"))
        row = dict(ex)
        row.update({
            "predicted_activates": predicted,
            "confidence": _safe_float(item.get("confidence")),
            "reason": item.get("reason"),
        })
        if predicted is not None:
            row["correct"] = bool(predicted) == bool(ex["expected_activates"])
        else:
            row["correct"] = None
        rows.append(row)

    out = {
        "description_text": description_text(cluster),
        "metrics": _classification_metrics(rows),
        "examples": rows,
        "summary": parsed.get("summary"),
        "raw_response": content,
    }
    if parsed.get("raw_parse_error") is not None:
        out["parse_error"] = parsed["raw_parse_error"]
    return out


def compute_detection_scores(
    labels_path: str | Path,
    *,
    client: Any | None = None,
    model: str = ORFEO_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    positive_examples: int = 8,
    negative_examples: int = 8,
    cluster_ids: Sequence[int] | None = None,
    max_clusters: int | None = None,
    seed: int = 0,
    max_workers: int = 1,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Compute LLM-as-judge activation-detection scores for descriptions."""
    loaded = load_labeled_clusters(labels_path)
    clusters = loaded["clusters"]
    wanted = _normalize_cluster_ids(clusters, cluster_ids, max_clusters)
    if client is None:
        client = make_orfeo_client()

    def _one(cluster_id: int) -> tuple[int, dict[str, Any]]:
        examples = select_detection_examples(
            clusters,
            cluster_id,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            seed=seed,
        )
        result = judge_one_cluster_detection(
            client,
            clusters[cluster_id],
            examples,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return cluster_id, result

    results: dict[int, dict[str, Any]] = {}
    if max_workers <= 1:
        iterator = tqdm(wanted, desc="judging descriptions", disable=not show_progress)
        for cluster_id in iterator:
            key, value = _one(cluster_id)
            results[key] = value
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_one, cluster_id): cluster_id for cluster_id in wanted}
            iterator = tqdm(as_completed(futures), total=len(futures), desc="judging descriptions", disable=not show_progress)
            for fut in iterator:
                key, value = fut.result()
                results[key] = value

    scores = [
        value["metrics"]["detection_score"]
        for value in results.values()
        if value["metrics"].get("detection_score") is not None
    ]
    summary = {
        "n_clusters": len(results),
        "mean_detection_score": float(sum(scores) / len(scores)) if scores else None,
    }
    return {
        "metadata": {
            "labels_path": str(labels_path),
            "judge_model": model,
            "positive_examples": int(positive_examples),
            "negative_examples": int(negative_examples),
            "seed": int(seed),
        },
        "summary": summary,
        "clusters": {str(k): results[k] for k in sorted(results)},
    }


def _strip_target_tags(snippet: str) -> tuple[str, int, int, str]:
    start = snippet.find(TARGET_OPEN)
    end = snippet.find(TARGET_CLOSE)
    if start < 0 or end < 0 or end < start:
        clean = snippet
        return clean, -1, -1, ""
    left = snippet[:start]
    target = snippet[start + len(TARGET_OPEN):end]
    right = snippet[end + len(TARGET_CLOSE):]
    clean = left + target + right
    target_start = len(left)
    target_end = target_start + len(target)
    return clean, target_start, target_end, target


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(last_hidden.dtype).unsqueeze(-1)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class TransformerTextEmbedder:
    """Encode descriptions and marked target tokens with one Transformer encoder."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode_texts(self, texts: Sequence[str], *, batch_size: int = 32) -> torch.Tensor:
        """Return L2-normalized mean-pooled text embeddings on CPU."""
        outs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                toks = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                hidden = self.model(**toks).last_hidden_state
                emb = _mean_pool(hidden, toks["attention_mask"])
                emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
                outs.append(emb.cpu())
        if not outs:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(outs, dim=0)

    def encode_target_tokens(
        self,
        snippets: Sequence[str],
        *,
        batch_size: int = 16,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Return L2-normalized embeddings for the marked target-token span."""
        clean_texts: list[str] = []
        spans: list[tuple[int, int, str]] = []
        for snippet in snippets:
            clean, start, end, target = _strip_target_tags(snippet)
            clean_texts.append(clean)
            spans.append((start, end, target))

        outs: list[torch.Tensor] = []
        infos: list[dict[str, Any]] = []
        with torch.no_grad():
            for start_idx in range(0, len(clean_texts), batch_size):
                batch = clean_texts[start_idx:start_idx + batch_size]
                batch_spans = spans[start_idx:start_idx + batch_size]
                toks = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                offsets = toks.pop("offset_mapping")
                toks = {k: v.to(self.device) for k, v in toks.items()}
                hidden = self.model(**toks).last_hidden_state.float().cpu()
                attention = toks["attention_mask"].cpu()

                for row, (char_start, char_end, target) in enumerate(batch_spans):
                    selected: list[int] = []
                    if char_start >= 0 and char_end >= 0:
                        for pos, (tok_start, tok_end) in enumerate(offsets[row].tolist()):
                            if attention[row, pos].item() == 0:
                                continue
                            if tok_start == tok_end == 0:
                                continue
                            overlap = min(tok_end, char_end) - max(tok_start, char_start)
                            if overlap > 0:
                                selected.append(pos)
                    if selected:
                        emb = hidden[row, selected].mean(dim=0)
                        ok = True
                    else:
                        valid = attention[row].bool()
                        emb = hidden[row, valid].mean(dim=0)
                        ok = False
                    emb = torch.nn.functional.normalize(emb, p=2, dim=0)
                    outs.append(emb)
                    infos.append({
                        "target_text": target,
                        "matched_token_positions": selected,
                        "matched_target_span": bool(ok),
                    })
        if not outs:
            return torch.empty((0, 0), dtype=torch.float32), infos
        return torch.stack(outs, dim=0), infos


def compute_token_embedding_scores(
    labels_path: str | Path,
    *,
    embedder: TransformerTextEmbedder | Any | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 32,
    target_batch_size: int = 16,
    positive_examples: int = 8,
    negative_examples: int = 8,
    cluster_ids: Sequence[int] | None = None,
    max_clusters: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare descriptions to marked target-token embeddings in one encoder space."""
    loaded = load_labeled_clusters(labels_path)
    clusters = loaded["clusters"]
    wanted = _normalize_cluster_ids(clusters, cluster_ids, max_clusters)
    if embedder is None:
        embedder = TransformerTextEmbedder(model_name, device=device)

    descriptions = [description_text(clusters[k]) for k in wanted]
    desc_emb = embedder.encode_texts(descriptions, batch_size=batch_size)

    out_clusters: dict[str, Any] = {}
    for row_idx, cluster_id in enumerate(wanted):
        examples = select_detection_examples(
            clusters,
            cluster_id,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            seed=seed,
        )
        snippets = [ex.get("snippet") or "" for ex in examples]
        token_emb, token_info = embedder.encode_target_tokens(snippets, batch_size=target_batch_size)
        if token_emb.numel() == 0 or desc_emb.numel() == 0:
            sims = torch.empty(0)
        else:
            sims = token_emb @ desc_emb[row_idx]

        rows = []
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        for ex, info, score in zip(examples, token_info, sims.tolist()):
            score = float(score)
            row = dict(ex)
            row.update(info)
            row["cosine_to_description"] = score
            rows.append(row)
            if ex["expected_activates"]:
                pos_scores.append(score)
            else:
                neg_scores.append(score)

        def _mean(values: Sequence[float]) -> float | None:
            return float(sum(values) / len(values)) if values else None

        pos_mean = _mean(pos_scores)
        neg_mean = _mean(neg_scores)
        separation = None
        if pos_mean is not None and neg_mean is not None:
            separation = float(pos_mean - neg_mean)

        out_clusters[str(cluster_id)] = {
            "description_text": descriptions[row_idx],
            "metrics": {
                "positive_mean_cosine": pos_mean,
                "negative_mean_cosine": neg_mean,
                "positive_minus_negative": separation,
                "n_positive": len(pos_scores),
                "n_negative": len(neg_scores),
            },
            "examples": rows,
        }

    separations = [
        cluster["metrics"]["positive_minus_negative"]
        for cluster in out_clusters.values()
        if cluster["metrics"]["positive_minus_negative"] is not None
    ]
    return {
        "metadata": {
            "labels_path": str(labels_path),
            "model_name": getattr(embedder, "model_name", model_name),
            "positive_examples": int(positive_examples),
            "negative_examples": int(negative_examples),
            "seed": int(seed),
            "note": TOKEN_EMBEDDING_NOTE,
        },
        "summary": {
            "n_clusters": len(out_clusters),
            "mean_positive_minus_negative": float(sum(separations) / len(separations)) if separations else None,
        },
        "clusters": out_clusters,
    }


def _topk_similarities(
    embeddings: torch.Tensor,
    *,
    top_k: int,
    batch_size: int,
    save_full_matrix: bool,
    full_matrix_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    n = embeddings.shape[0]
    if n == 0:
        empty_i = torch.empty((0, 0), dtype=torch.long)
        empty_f = torch.empty((0, 0), dtype=torch.float32)
        return empty_i, empty_f, None
    k = max(0, min(int(top_k), n - 1))
    top_indices = torch.empty((n, k), dtype=torch.long)
    top_scores = torch.empty((n, k), dtype=torch.float32)
    full = torch.empty((n, n), dtype=full_matrix_dtype) if save_full_matrix else None

    emb = embeddings.float()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim = emb[start:end] @ emb.T
        rows = torch.arange(start, end)
        sim[torch.arange(end - start), rows] = -float("inf")
        if full is not None:
            full[start:end] = sim.to(full_matrix_dtype)
            full[rows, rows] = 1.0
        if k > 0:
            scores, indices = sim.topk(k, dim=1)
            top_indices[start:end] = indices.long()
            top_scores[start:end] = scores.float()
    return top_indices, top_scores, full


def _connected_groups_from_neighbors(
    cluster_ids: Sequence[int],
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    *,
    threshold: float,
    min_group_size: int,
) -> list[list[int]]:
    n = len(cluster_ids)
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j, score in zip(top_indices[i].tolist(), top_scores[i].tolist()):
            if score >= threshold:
                neighbors[i].add(int(j))
                neighbors[int(j)].add(i)

    seen = [False] * n
    groups: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        component = []
        while q:
            cur = q.popleft()
            component.append(cur)
            for nxt in neighbors[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        if len(component) >= min_group_size:
            groups.append([int(cluster_ids[idx]) for idx in sorted(component)])
    groups.sort(key=lambda group: (-len(group), group[0]))
    return groups


def _mean_internal_similarity(id_to_row: dict[int, int], group: Sequence[int], embeddings: torch.Tensor) -> float | None:
    if len(group) < 2:
        return None
    rows = torch.tensor([id_to_row[int(k)] for k in group], dtype=torch.long)
    sub = embeddings[rows]
    sim = sub @ sub.T
    triu = torch.triu_indices(len(group), len(group), offset=1)
    return float(sim[triu[0], triu[1]].mean().item())


def _summary_stats(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.float().flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _condensed_distance(distance: torch.Tensor) -> torch.Tensor:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"distance matrix must be square, got shape {tuple(distance.shape)}")
    if distance.shape[0] < 2:
        return torch.empty(0, dtype=torch.float32)
    idx = torch.triu_indices(distance.shape[0], distance.shape[1], offset=1)
    condensed = distance[idx[0], idx[1]].float()
    if not torch.isfinite(condensed).all():
        raise ValueError("distance matrix contains non-finite off-diagonal values")
    return condensed


def compute_gaussian_group_label_coherence(
    labels_path: str | Path,
    gaussian_overlap_path: str | Path,
    *,
    embedder: TransformerTextEmbedder | Any | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 32,
    distance_key: str = "db",
    distance_threshold: float,
    linkage: str = "average",
    top_groups: int = 10,
) -> dict[str, Any]:
    """Cluster MFA Gaussians by distance and summarize label coherence per group."""
    from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage

    labels_path = Path(labels_path)
    gaussian_overlap_path = Path(gaussian_overlap_path)
    loaded = load_labeled_clusters(labels_path)
    clusters = loaded["clusters"]

    gaussian_overlap = torch.load(gaussian_overlap_path, map_location="cpu")
    if distance_key not in gaussian_overlap:
        raise ValueError(
            f"{gaussian_overlap_path} is missing distance key {distance_key!r}"
        )
    distance = gaussian_overlap[distance_key].float().cpu()
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"{distance_key} must be a square matrix, got shape {tuple(distance.shape)}")

    K = int(distance.shape[0])
    wanted = list(range(K))
    missing = [cluster_id for cluster_id in wanted if cluster_id not in clusters]
    if missing:
        raise ValueError(
            "labels are missing component ids from Gaussian-overlap matrix: "
            f"{missing[:10]}"
        )

    if embedder is None:
        embedder = TransformerTextEmbedder(model_name, device=device)
    texts = [description_text(clusters[k]) for k in wanted]
    embeddings = embedder.encode_texts(texts, batch_size=batch_size).float()
    if embeddings.shape[0] != K:
        raise ValueError(f"embedder returned {embeddings.shape[0]} embeddings for {K} descriptions")

    condensed = _condensed_distance(distance)
    if K == 1:
        flat_labels = [1]
    else:
        Z = scipy_linkage(condensed.numpy(), method=linkage)
        flat_labels = fcluster(Z, t=float(distance_threshold), criterion="distance").tolist()

    label_to_members: dict[int, list[int]] = {}
    for component_id, group_label in enumerate(flat_labels):
        label_to_members.setdefault(int(group_label), []).append(int(component_id))

    all_groups = sorted(
        (sorted(members) for members in label_to_members.values()),
        key=lambda members: (-len(members), members[0]),
    )
    selected_groups = [group for group in all_groups if len(group) >= 2][: int(top_groups)]

    semantic_sim = embeddings @ embeddings.T if embeddings.numel() else torch.empty((K, K))
    group_outputs: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for group_id, group in enumerate(selected_groups):
        rows = torch.tensor(group, dtype=torch.long)
        pair_idx = torch.triu_indices(len(group), len(group), offset=1)
        group_semantic = semantic_sim[rows][:, rows][pair_idx[0], pair_idx[1]]
        group_distance = distance[rows][:, rows][pair_idx[0], pair_idx[1]]

        semantic_stats = _summary_stats(group_semantic)
        distance_stats = _summary_stats(group_distance)
        members = []
        for component_id in group:
            cluster = clusters[int(component_id)]
            member = {
                "component_id": int(component_id),
                "label": cluster.get("label"),
                "description": cluster.get("description"),
                "text": description_text(cluster),
            }
            members.append(member)
            member_rows.append({
                "group_id": group_id,
                "group_size": len(group),
                "component_id": int(component_id),
                "label": cluster.get("label"),
                "description": cluster.get("description"),
                "mean_label_cosine": semantic_stats["mean"],
                "median_label_cosine": semantic_stats["median"],
                "mean_distance": distance_stats["mean"],
                "median_distance": distance_stats["median"],
            })

        group_outputs.append({
            "group_id": group_id,
            "size": len(group),
            "component_ids": group,
            "label_cosine": semantic_stats,
            "distance": distance_stats,
            "members": members,
        })

    return {
        "metadata": {
            "labels_path": str(labels_path),
            "gaussian_overlap_path": str(gaussian_overlap_path),
            "distance_key": distance_key,
            "distance_threshold": float(distance_threshold),
            "linkage": linkage,
            "top_groups": int(top_groups),
            "model_name": getattr(embedder, "model_name", model_name),
        },
        "summary": {
            "n_components": K,
            "n_gaussian_groups": len(all_groups),
            "n_non_singleton_gaussian_groups": sum(1 for group in all_groups if len(group) >= 2),
            "largest_group_sizes": [len(group) for group in all_groups[: int(top_groups)]],
            "n_selected_groups": len(selected_groups),
        },
        "groups": group_outputs,
        "member_rows": member_rows,
    }


def compute_description_semantics(
    labels_path: str | Path,
    *,
    embedder: TransformerTextEmbedder | Any | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 32,
    similarity_batch_size: int = 1024,
    top_k: int = 25,
    similarity_threshold: float = 0.70,
    min_group_size: int = 2,
    save_full_matrix: bool = False,
    full_matrix_dtype: torch.dtype = torch.float32,
    cluster_ids: Sequence[int] | None = None,
    max_clusters: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Embed descriptions and recover semantic neighbors/groups."""
    loaded = load_labeled_clusters(labels_path)
    clusters = loaded["clusters"]
    wanted = _normalize_cluster_ids(clusters, cluster_ids, max_clusters)
    if embedder is None:
        embedder = TransformerTextEmbedder(model_name, device=device)

    texts = [description_text(clusters[k]) for k in wanted]
    embeddings = embedder.encode_texts(texts, batch_size=batch_size)
    top_indices, top_scores, full = _topk_similarities(
        embeddings,
        top_k=top_k,
        batch_size=similarity_batch_size,
        save_full_matrix=save_full_matrix,
        full_matrix_dtype=full_matrix_dtype,
    )
    groups = _connected_groups_from_neighbors(
        wanted,
        top_indices,
        top_scores,
        threshold=similarity_threshold,
        min_group_size=min_group_size,
    )

    id_to_row = {int(cluster_id): row for row, cluster_id in enumerate(wanted)}
    group_json = []
    for group_id, group in enumerate(groups):
        group_json.append({
            "group_id": group_id,
            "size": len(group),
            "mean_internal_similarity": _mean_internal_similarity(id_to_row, group, embeddings),
            "cluster_ids": group,
            "members": [
                {
                    "cluster_id": int(cluster_id),
                    "label": clusters[int(cluster_id)].get("label"),
                    "description": clusters[int(cluster_id)].get("description"),
                }
                for cluster_id in group
            ],
        })

    tensor_output = {
        "metadata": {
            "labels_path": str(labels_path),
            "model_name": getattr(embedder, "model_name", model_name),
            "top_k": int(top_k),
            "similarity_threshold": float(similarity_threshold),
            "save_full_matrix": bool(save_full_matrix),
        },
        "cluster_ids": torch.tensor(wanted, dtype=torch.long),
        "texts": texts,
        "embeddings": embeddings.float(),
        "topk_indices": top_indices,
        "topk_scores": top_scores,
    }
    if full is not None:
        tensor_output["similarity_matrix"] = full

    json_output = {
        "metadata": tensor_output["metadata"],
        "groups": group_json,
    }
    return tensor_output, json_output
