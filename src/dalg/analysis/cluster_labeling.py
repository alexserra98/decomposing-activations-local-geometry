from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm


ORFEO_BASE_URL = "https://orfeo-llm.areasciencepark.it/vllm/v1"
ORFEO_MODEL = "google/gemma-4-26B-A4B-it"

TARGET_OPEN = "<target>"
TARGET_CLOSE = "</target>"

_LOG_TTY = sys.stderr.isatty()
_TQDM_MININTERVAL = 0.5 if _LOG_TTY else 30.0
_TQDM_MAXINTERVAL = 10.0 if _LOG_TTY else 60.0


SYSTEM_PROMPT = """You are helping interpret latent components in a language model.

You will see target tokens wrapped in <target>...</target>, each shown inside
its original text context. Infer the common role, function, topic, or textual
pattern of the target token itself. Do not describe the whole passage unless
that is what explains the target token.

Return JSON only with this schema:
{"label": "<short label>", "description": "<1-2 sentence description>", "evidence": "<brief evidence from the examples>"}
"""


def load_assignment_data(assignments_path: str | Path) -> dict[str, Any]:
    """Load the assignment tensor bundle produced by cluster_assignments.py."""
    assignments_path = Path(assignments_path)
    data = torch.load(assignments_path, map_location="cpu", weights_only=False)

    required = {"cluster_sizes", "assignments", "max_responsibilities"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{assignments_path} is missing required keys: {missing}")

    cluster_sizes = data["cluster_sizes"].long().cpu()
    assignments = data["assignments"].long().cpu()
    max_responsibilities = data["max_responsibilities"].float().cpu()
    K = int(data.get("K", int(cluster_sizes.numel())))

    if assignments.numel() != max_responsibilities.numel():
        raise ValueError(
            "assignments and max_responsibilities must have the same length: "
            f"{assignments.numel()} vs {max_responsibilities.numel()}"
        )
    if cluster_sizes.numel() != K:
        raise ValueError(f"cluster_sizes has {cluster_sizes.numel()} entries, but K={K}")

    return {
        "cluster_sizes": cluster_sizes,
        "assignments": assignments,
        "max_responsibilities": max_responsibilities,
        "peakedness": data.get("peakedness", {}),
        "K": K,
        "subset_spec": data.get("subset_spec"),
    }


def resolve_labeling_config(
    assignments_path: str | Path,
    *,
    shard_dir: str | Path | None = None,
    layer: int | None = None,
    windows_dataset: str | Path | None = None,
    window: int | None = None,
    drop_prefix: int | None = None,
) -> dict[str, Any]:
    """Resolve paths and token-window metadata from local run configs."""
    assignments_path = Path(assignments_path)
    mfa_dir = assignments_path.parent
    user_shard_dir = shard_dir is not None
    user_windows_dataset = windows_dataset is not None

    mfa_cfg: dict[str, Any] = {}
    mfa_cfg_path = mfa_dir / "config.json"
    if mfa_cfg_path.exists():
        mfa_cfg = json.loads(mfa_cfg_path.read_text())

    if shard_dir is None and mfa_cfg.get("shard_dir") is not None:
        shard_dir = mfa_cfg["shard_dir"]
    if layer is None and mfa_cfg.get("layer") is not None:
        layer = int(mfa_cfg["layer"])
    if layer is None:
        match = re.search(r"layer(\d+)", mfa_dir.name)
        if match:
            layer = int(match.group(1))

    extract_cfg: dict[str, Any] = {}
    if shard_dir is not None:
        shard_dir = Path(shard_dir)
        extract_cfg_path = shard_dir / "config.json"
        if extract_cfg_path.exists():
            extract_cfg = json.loads(extract_cfg_path.read_text())
        elif not user_shard_dir:
            fallback_shard_dir = mfa_dir.parent
            fallback_cfg_path = fallback_shard_dir / "config.json"
            if fallback_cfg_path.exists():
                shard_dir = fallback_shard_dir
                extract_cfg = json.loads(fallback_cfg_path.read_text())

    if window is None:
        window = extract_cfg.get("window", mfa_cfg.get("window"))
    if drop_prefix is None:
        drop_prefix = extract_cfg.get("drop_prefix", mfa_cfg.get("drop_prefix", 32))
    if windows_dataset is None:
        windows_dataset = extract_cfg.get("dataset")
    if windows_dataset is not None and not user_windows_dataset:
        windows_dataset = Path(windows_dataset)
        if not windows_dataset.exists() and shard_dir is not None:
            fallback_windows = Path(shard_dir).parent / windows_dataset.parent.name / windows_dataset.name
            if fallback_windows.exists():
                windows_dataset = fallback_windows

    if shard_dir is None:
        raise ValueError(
            "Could not infer shard_dir. Pass --shard-dir or keep config.json next to the assignments file."
        )
    if layer is None:
        raise ValueError(
            "Could not infer layer. Pass --layer or keep config.json next to the assignments file."
        )
    if window is None:
        raise ValueError(
            "Could not infer token window length. Pass --window or provide shard config.json."
        )
    if drop_prefix is None:
        raise ValueError(
            "Could not infer drop_prefix. Pass --drop-prefix or provide run config.json."
        )

    return {
        "assignments_path": str(assignments_path),
        "mfa_dir": str(mfa_dir),
        "shard_dir": str(Path(shard_dir)),
        "layer": int(layer),
        "windows_dataset": str(Path(windows_dataset)) if windows_dataset is not None else None,
        "window": int(window),
        "drop_prefix": int(drop_prefix),
    }


def _normalize_cluster_ids(cluster_ids: Sequence[int] | None, K: int) -> list[int]:
    if cluster_ids is None:
        return list(range(K))
    out = [int(k) for k in cluster_ids]
    bad = [k for k in out if k < 0 or k >= K]
    if bad:
        raise ValueError(f"cluster ids out of range for K={K}: {bad[:10]}")
    return out


def select_top_activations(
    assignments: torch.Tensor,
    max_responsibilities: torch.Tensor,
    *,
    K: int,
    top_n: int = 20,
    cluster_ids: Sequence[int] | None = None,
    chunk_size: int = 1_000_000,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Select the top assigned activations per cluster by max responsibility.

    The assignment file stores one hard assignment per activation plus that
    activation's top responsibility. For examples assigned to cluster k, this
    is exactly the responsibility for k.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if assignments.numel() != max_responsibilities.numel():
        raise ValueError("assignments and max_responsibilities must have the same length")

    cluster_ids = _normalize_cluster_ids(cluster_ids, K)
    row_for_cluster = {cluster_id: row for row, cluster_id in enumerate(cluster_ids)}
    n_clusters = len(cluster_ids)
    wanted_mask = None
    if n_clusters != K or cluster_ids != list(range(K)):
        wanted_mask = torch.zeros(K, dtype=torch.bool)
        wanted_mask[torch.tensor(cluster_ids, dtype=torch.long)] = True

    top_scores = torch.full((n_clusters, top_n), -float("inf"), dtype=torch.float32)
    top_positions = torch.full((n_clusters, top_n), -1, dtype=torch.long)

    n = int(assignments.numel())
    starts = range(0, n, chunk_size)
    iterator = tqdm(
        starts,
        desc="selecting top activations",
        mininterval=_TQDM_MININTERVAL,
        maxinterval=_TQDM_MAXINTERVAL,
        disable=not show_progress,
    )

    for start in iterator:
        end = min(start + chunk_size, n)
        chunk_assign = assignments[start:end]
        chunk_scores = torch.nan_to_num(
            max_responsibilities[start:end],
            nan=-float("inf"),
            neginf=-float("inf"),
            posinf=float("inf"),
        )
        if chunk_assign.numel() == 0:
            continue

        local_indices = torch.arange(chunk_assign.numel(), dtype=torch.long)
        if wanted_mask is not None:
            keep = wanted_mask[chunk_assign]
            if not bool(keep.any()):
                continue
            chunk_assign = chunk_assign[keep]
            chunk_scores = chunk_scores[keep]
            local_indices = local_indices[keep]

        order = torch.argsort(chunk_assign)
        sorted_assign = chunk_assign[order]
        sorted_scores = chunk_scores[order]
        sorted_indices = local_indices[order]
        clusters, counts = torch.unique_consecutive(sorted_assign, return_counts=True)

        offset = 0
        for cluster, count in zip(clusters.tolist(), counts.tolist()):
            row = row_for_cluster.get(int(cluster))
            group_scores = sorted_scores[offset:offset + count]
            group_indices = sorted_indices[offset:offset + count]
            offset += count
            if row is None:
                continue

            n_local = min(top_n, int(count))
            local_scores, local_sel = group_scores.topk(n_local)
            local_positions = group_indices[local_sel].long() + int(start)

            merged_scores = torch.cat([top_scores[row], local_scores.float()])
            merged_positions = torch.cat([top_positions[row], local_positions])
            new_scores, sel = merged_scores.topk(top_n)

            top_scores[row] = new_scores
            top_positions[row] = merged_positions[sel]

    return {
        "K": int(K),
        "cluster_ids": torch.tensor(cluster_ids, dtype=torch.long),
        "top_n": int(top_n),
        "positions": top_positions,
        "responsibilities": top_scores,
    }


def map_positions_to_token_coordinates(
    positions: torch.Tensor,
    meta_index: Sequence[dict[str, Any]],
    *,
    window: int,
    drop_prefix: int,
    assignment_count: int | None = None,
) -> dict[str, torch.Tensor]:
    """Map flat assignment positions to windows-dataset rows and token positions."""
    tokens_per_row = int(window) - int(drop_prefix)
    if tokens_per_row <= 0:
        raise ValueError(f"drop_prefix={drop_prefix} must be smaller than window={window}")

    total_positions = len(meta_index) * tokens_per_row
    if assignment_count is not None and assignment_count > total_positions:
        raise ValueError(
            f"assignment file has {assignment_count:,} positions, but shard metadata only "
            f"accounts for {total_positions:,}"
        )

    shape = positions.shape
    flat = positions.reshape(-1).long()
    global_rows = torch.full_like(flat, -1)
    tok_pos = torch.full_like(flat, -1)
    shards = torch.full_like(flat, -1)
    rows_in_shard = torch.full_like(flat, -1)

    for i, pos in enumerate(flat.tolist()):
        if pos < 0:
            continue
        if pos >= total_positions:
            raise ValueError(
                f"position {pos:,} is outside the shard metadata range {total_positions:,}"
            )
        row_idx = pos // tokens_per_row
        local_tok = pos % tokens_per_row
        meta = meta_index[int(row_idx)]
        global_rows[i] = int(meta["global_row"])
        tok_pos[i] = int(drop_prefix + local_tok)
        shards[i] = int(meta["shard"])
        rows_in_shard[i] = int(meta["row_in_shard"])

    return {
        "global_row": global_rows.reshape(shape),
        "tok_pos": tok_pos.reshape(shape),
        "shard": shards.reshape(shape),
        "row_in_shard": rows_in_shard.reshape(shape),
    }


def _dataset_rows(dataset: Any, rows: Sequence[int]) -> dict[int, Sequence[int]]:
    """Return token-id windows for the requested global dataset rows.

    Hugging Face datasets support `select`, which is much faster than many
    random `__getitem__` calls. Tests can pass a small list-like fake dataset,
    so fall back to direct indexing when `select` is not available.
    """
    unique_rows = sorted({int(r) for r in rows if int(r) >= 0})
    if hasattr(dataset, "select"):
        selected = dataset.select(unique_rows)
        return {row: selected[i]["token_ids"] for i, row in enumerate(unique_rows)}
    return {row: dataset[row]["token_ids"] for row in unique_rows}


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def build_context_examples(
    top_index: dict[str, Any],
    coordinates: dict[str, torch.Tensor],
    windows_dataset: Any,
    tokenizer: Any,
    *,
    cluster_sizes: torch.Tensor | None = None,
    pad: int = 10,
    max_examples_per_cluster: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Recover context snippets for selected top activations."""
    cluster_ids = top_index["cluster_ids"].long() # shape (n_clusters, top_n)
    positions = top_index["positions"].long() # shape (n_clusters, top_n)
    responsibilities = top_index["responsibilities"].float()# shape (n_clusters, top_n)
    global_rows = coordinates["global_row"].long() # shape (n_clusters, top_n)
    tok_pos = coordinates["tok_pos"].long() # shape (n_clusters, top_n)
    
    
    row_cache = _dataset_rows(windows_dataset, global_rows.reshape(-1).tolist())

    clusters: dict[int, dict[str, Any]] = {}
    for row_idx, cluster_id in enumerate(cluster_ids.tolist()):
        examples: list[dict[str, Any]] = []
        limit = positions.shape[1]
        if max_examples_per_cluster is not None:
            limit = min(limit, int(max_examples_per_cluster))

        for rank in range(limit):
            stream_pos = int(positions[row_idx, rank].item())
            if stream_pos < 0:
                continue
            gr = int(global_rows[row_idx, rank].item())
            p = int(tok_pos[row_idx, rank].item())
            toks = list(row_cache[gr])
            if p < 0 or p >= len(toks):
                raise ValueError(f"tok_pos={p} is outside row {gr} length {len(toks)}")

            lo = max(0, p - pad)
            hi = min(len(toks), p + pad + 1)
            left = _decode(tokenizer, toks[lo:p])
            target = _decode(tokenizer, [toks[p]])
            right = _decode(tokenizer, toks[p + 1:hi])
            snippet = f"{left}{TARGET_OPEN}{target}{TARGET_CLOSE}{right}"

            examples.append({
                "rank": rank + 1,
                "stream_position": stream_pos,
                "responsibility": float(responsibilities[row_idx, rank].item()),
                "global_row": gr,
                "tok_pos": p,
                "token_id": int(toks[p]),
                "token": target,
                "snippet": snippet,
            })

        token_counts = Counter(ex["token"] for ex in examples)
        top_tokens = [
            {"token": token, "count": count}
            for token, count in token_counts.most_common()
        ]
        size = None
        if cluster_sizes is not None:
            size = int(cluster_sizes[int(cluster_id)].item())
        clusters[int(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "cluster_size": size,
            "top_tokens": top_tokens,
            "examples": examples,
        }

    return clusters


def build_label_prompt(cluster: dict[str, Any]) -> str:
    top_tokens = ", ".join(
        f"{item['token']!r} x{item['count']}"
        for item in cluster["top_tokens"][:30]
    )
    if not top_tokens:
        top_tokens = "(no examples)"

    lines = [
        f"Cluster id: {cluster['cluster_id']}",
        f"Cluster size: {cluster.get('cluster_size')}",
        f"Top target tokens: {top_tokens}",
        "",
        "Contexts:",
    ]
    for ex in cluster["examples"]:
        lines.append(
            f"{ex['rank']}. responsibility={ex['responsibility']:.6f} "
            f"token={ex['token']!r} context={ex['snippet']}"
        )
    lines.append("")
    lines.append("Describe the common function of the target token. Return JSON only.")
    return "\n".join(lines)


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON even when the model wraps it in a markdown code fence."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {
        "label": None,
        "description": None,
        "evidence": None,
        "raw_parse_error": text,
    }


def make_orfeo_client(
    *,
    api_key: str | None = None,
    base_url: str = ORFEO_BASE_URL,
) -> Any:
    """Create the OpenAI-compatible Orfeo client used by orfeo_agent.py."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    from openai import OpenAI

    api_key = api_key or os.getenv("ORFEO_API_KEY")
    if not api_key:
        raise RuntimeError("ORFEO_API_KEY is not set")
    return OpenAI(api_key=api_key, base_url=base_url)


def label_one_cluster(
    client: Any,
    cluster: dict[str, Any],
    *,
    model: str = ORFEO_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict[str, Any]:
    prompt = build_label_prompt(cluster)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    parsed = parse_json_response(content)
    parsed.setdefault("label", None)
    parsed.setdefault("description", None)
    parsed.setdefault("evidence", None)
    parsed["raw_response"] = content
    return parsed


def label_clusters_with_llm(
    clusters: dict[int, dict[str, Any]],
    *,
    client: Any,
    model: str = ORFEO_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_workers: int = 1,
    show_progress: bool = True,
) -> dict[int, dict[str, Any]]:
    """Call the LLM once per cluster."""
    if max_workers <= 1:
        out: dict[int, dict[str, Any]] = {}
        iterator = tqdm(
            clusters.items(),
            total=len(clusters),
            desc="labeling clusters",
            mininterval=_TQDM_MININTERVAL,
            maxinterval=_TQDM_MAXINTERVAL,
            disable=not show_progress,
        )
        for cluster_id, cluster in iterator:
            out[int(cluster_id)] = label_one_cluster(
                client,
                cluster,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return out

    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                label_one_cluster,
                client,
                cluster,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ): int(cluster_id)
            for cluster_id, cluster in clusters.items()
        }
        iterator = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="labeling clusters",
            mininterval=_TQDM_MININTERVAL,
            maxinterval=_TQDM_MAXINTERVAL,
            disable=not show_progress,
        )
        for fut in iterator:
            cluster_id = futures[fut]
            out[cluster_id] = fut.result()
    return dict(sorted(out.items()))


def make_output(
    *,
    config: dict[str, Any],
    clusters: dict[int, dict[str, Any]],
    labels: dict[int, dict[str, Any]] | None,
    top_index_path: str | Path | None,
) -> dict[str, Any]:
    out_clusters: dict[str, Any] = {}
    labels = labels or {}
    for cluster_id, cluster in sorted(clusters.items()):
        label = labels.get(cluster_id, {
            "label": None,
            "description": None,
            "evidence": None,
        })
        out_clusters[str(cluster_id)] = {
            "label": label.get("label"),
            "description": label.get("description"),
            "evidence": label.get("evidence"),
            "top_tokens": cluster["top_tokens"],
            "examples": cluster["examples"],
            "raw_response": label.get("raw_response"),
        }
        if cluster.get("cluster_size") is not None:
            out_clusters[str(cluster_id)]["cluster_size"] = cluster["cluster_size"]

    return {
        "metadata": {
            **config,
            "top_index_path": str(top_index_path) if top_index_path is not None else None,
        },
        "clusters": out_clusters,
    }


def label_mfa_clusters(
    assignments_path: str | Path,
    *,
    out_dir: str | Path,
    shard_dir: str | Path | None = None,
    layer: int | None = None,
    windows_dataset: str | Path | None = None,
    tokenizer_name: str = "google/gemma-2b",
    window: int | None = None,
    drop_prefix: int | None = None,
    top_n: int = 20,
    max_examples_per_cluster: int | None = None,
    pad: int = 10,
    cluster_ids: Sequence[int] | None = None,
    max_clusters: int | None = None,
    chunk_size: int = 1_000_000,
    skip_llm: bool = False,
    llm_client: Any | None = None,
    llm_model: str = ORFEO_MODEL,
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 512,
    llm_workers: int = 1,
    top_index_path: str | Path | None = None,
    show_progress: bool = True,
    windows_dataset_obj: Any | None = None,
    tokenizer: Any | None = None,
    meta_index: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full labeling pipeline from assignments file to JSON-ready output."""
    assignments_path = Path(assignments_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_spec = None
    if shard_dir is not None:
        from dalg.data.subset_spec import split_shard_dir_spec

        shard_dir, subset_spec = split_shard_dir_spec(shard_dir)

    config = resolve_labeling_config(
        assignments_path,
        shard_dir=shard_dir,
        layer=layer,
        windows_dataset=windows_dataset,
        window=window,
        drop_prefix=drop_prefix,
    )
    if config["windows_dataset"] is None:
        raise ValueError("windows_dataset is required for context retrieval")

    assignment_data = load_assignment_data(assignments_path)
    K = int(assignment_data["K"])
    # Fall back to the subset spec recorded at assignment time so labeling maps
    # positions through the same row subset even if --shard-dir carried no suffix.
    if subset_spec is None:
        subset_spec = assignment_data.get("subset_spec")
    if cluster_ids is None and max_clusters is not None:
        cluster_ids = list(range(min(int(max_clusters), K)))

    if top_index_path is None:
        top_index_path = out_dir / "top_activations.pt"
    top_index_path = Path(top_index_path)

    top_index = select_top_activations(
        assignment_data["assignments"],
        assignment_data["max_responsibilities"],
        K=K,
        top_n=top_n,
        cluster_ids=cluster_ids,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
    torch.save(top_index, top_index_path)

    if meta_index is None:
        from dalg.data.shard_activations import load_meta_index

        meta_index = load_meta_index(config["shard_dir"], layer=config.get("layer"))
        if subset_spec:
            from dalg.data.subset_spec import resolve_spec_positions

            keep = resolve_spec_positions(
                meta_index,
                subset_spec,
                window=int(config["window"]),
                drop_prefix=int(config["drop_prefix"]),
            )
            meta_index = [meta_index[i] for i in keep]
    coordinates = map_positions_to_token_coordinates(
        top_index["positions"],
        meta_index,
        window=int(config["window"]),
        drop_prefix=int(config["drop_prefix"]),
        assignment_count=int(assignment_data["assignments"].numel()),
    )
    if windows_dataset_obj is None:
        from datasets import load_from_disk

        windows_dataset_obj = load_from_disk(config["windows_dataset"])
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    clusters = build_context_examples(
        top_index,
        coordinates,
        windows_dataset_obj,
        tokenizer,
        cluster_sizes=assignment_data["cluster_sizes"],
        pad=pad,
        max_examples_per_cluster=max_examples_per_cluster,
    )

    examples_path = out_dir / "cluster_examples.json"
    examples_path.write_text(json.dumps(clusters, ensure_ascii=False, indent=2))

    labels = None
    if not skip_llm:
        if llm_client is None:
            llm_client = make_orfeo_client()
        labels = label_clusters_with_llm(
            clusters,
            client=llm_client,
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            max_workers=llm_workers,
            show_progress=show_progress,
        )

    output = make_output(
        config=config,
        clusters=clusters,
        labels=labels,
        top_index_path=top_index_path,
    )
    output_path = out_dir / "cluster_labels.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return output
