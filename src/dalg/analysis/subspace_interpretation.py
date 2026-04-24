import torch
import heapq
from typing import Callable, Dict, List, Tuple, Any, Optional, Union, Literal
from collections import defaultdict


@torch.no_grad()
def get_top_strings_per_concept(
    model,
    loader,
    tok2str: Callable[[Any], str],
    *,
    topk: int = 50,
    device: Optional[torch.device] = None,
    return_scores: bool = False,
    score: Literal["posterior", "likelihood"] = "posterior",
    aggregate: Literal["occurrence", "max", "sum"] = "occurrence",
) -> Dict[int, List[Union[str, Tuple[str, float]]]]:
    """
    For each MFA component k, collect the top-scoring token strings from a loader.

    Args:
        model: Trained MFA model with responsibilities() and log_prob_components() methods.
        loader: DataLoader yielding (activations, token_ids) batches.
        tok2str: Callable that converts a token ID to a human-readable string.
        topk: Number of top tokens to return per component.
        device: Device to run the model on (defaults to the model's device).
        return_scores: If True, return (string, score) tuples instead of strings.
        score: Scoring function.
            "posterior"  — alpha_k(h) = p(k | h),  via model.responsibilities().
            "likelihood" — ll_k(h) = log p(h | k),  via model.log_prob_components().
        aggregate: How to aggregate across occurrences of the same token.
            "occurrence" — keep top individual token occurrences (can repeat).
            "max"        — de-duplicate by string; keep max score per token.
            "sum"        — de-duplicate by string; sum scores across occurrences.

    Returns:
        Dict mapping component index k -> list of top token strings (or (string, score)
        tuples if return_scores=True), sorted by descending score, length <= topk.
    """
    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    heaps: Dict[int, List[Tuple[float, int, str]]] = {}
    agg_maps: Dict[int, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float) if aggregate == "sum" else dict
    )

    counter = 0
    K_seen: Optional[int] = None

    for batch in loader:
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError("Loader must yield (activations, tokens).")
        h, toks = batch[0].to(device, non_blocking=True), batch[1]

        if score == "posterior":
            scores_mat = model.responsibilities(h)     # (B, K)
        else:
            scores_mat = model.log_prob_components(h)  # (B, K)

        B, K = scores_mat.shape
        if K_seen is None:
            K_seen = K
            if aggregate == "occurrence":
                heaps = {k: [] for k in range(K)}

        scores_cpu = scores_mat.detach().cpu()

        for i in range(B):
            row = scores_cpu[i]  # (K,)
            if not torch.isfinite(row).all():
                continue
            s = tok2str(toks[i])

            for k in range(K):
                val = float(row[k])

                if aggregate == "occurrence":
                    hp = heaps[k]
                    if len(hp) < topk:
                        heapq.heappush(hp, (val, counter, s))
                    elif val > hp[0][0]:
                        heapq.heapreplace(hp, (val, counter, s))
                    counter += 1
                elif aggregate == "max":
                    if val > agg_maps[k].get(s, float("-inf")):
                        agg_maps[k][s] = val
                else:  # "sum"
                    agg_maps[k][s] += val

    result: Dict[int, List[Union[str, Tuple[str, float]]]] = {}

    if aggregate == "occurrence":
        for k, hp in heaps.items():
            items = sorted(hp, key=lambda t: t[0], reverse=True)
            if return_scores:
                result[k] = [(s, sc) for (sc, _, s) in items]
            else:
                result[k] = [s for (_, _, s) in items]
    else:
        for k, d in agg_maps.items():
            items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:topk]
            result[k] = items if return_scores else [s for s, _ in items]

    if was_training:
        model.train()

    return result


@torch.no_grad()
def get_top_indices_per_concept(
    model,
    loader,
    *,
    topk: int = 50,
    device: Optional[torch.device] = None,
    return_scores: bool = False,
    score: Literal["posterior", "likelihood"] = "posterior",
) -> Dict[int, List[Union[int, Tuple[int, float]]]]:
    """
    For each MFA component k, collect the top-scoring global sample indices from a loader.

    Same scoring logic as get_top_strings_per_concept, but returns the 0-based
    iteration index of each sample rather than a token string.

    Args:
        model: Trained MFA model.
        loader: DataLoader yielding (activations, ...) batches.
        topk: Number of top samples to return per component.
        device: Device to run the model on.
        return_scores: If True, return (index, score) tuples.
        score: "posterior" or "likelihood" (see get_top_strings_per_concept).

    Returns:
        Dict mapping k -> list of sample indices (or (index, score) if return_scores=True),
        sorted by descending score, length <= topk.
    """
    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    heaps: Dict[int, List[Tuple[float, int, int]]] = {}
    global_idx = 0
    K_seen: Optional[int] = None
    tie_breaker = 0

    for batch in loader:
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 1):
            raise ValueError("Loader must yield (activations, ...).")
        h = batch[0].to(device, non_blocking=True)

        if score == "posterior":
            S = model.responsibilities(h)
        else:
            S = model.log_prob_components(h)

        B, K = S.shape
        if K_seen is None:
            K_seen = K
            heaps = {k: [] for k in range(K)}

        S_cpu = S.detach().cpu()

        for i in range(B):
            row = S_cpu[i]
            if not torch.isfinite(row).all():
                global_idx += 1
                continue

            for k in range(K):
                key = float(row[k])
                hp = heaps[k]
                if len(hp) < topk:
                    heapq.heappush(hp, (key, tie_breaker, global_idx))
                elif key > hp[0][0]:
                    heapq.heapreplace(hp, (key, tie_breaker, global_idx))
                tie_breaker += 1
            global_idx += 1

    result: Dict[int, List[Union[int, Tuple[int, float]]]] = {}
    for k, hp in heaps.items():
        items = sorted(hp, key=lambda t: t[0], reverse=True)
        if return_scores:
            result[k] = [(idx, sc) for (sc, _tb, idx) in items]
        else:
            result[k] = [idx for (sc, _tb, idx) in items]

    if was_training:
        model.train()

    return result
