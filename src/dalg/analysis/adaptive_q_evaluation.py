"""Evaluation used by the toy-manifold adaptive-rank experiment.

The notebook visualizes these results, but the numerical evaluation lives here
so it can run non-interactively after assignments are complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from torch.utils.data import DataLoader

from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("adaptive-q evaluation requested CUDA, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("adaptive-q evaluation requested MPS, but MPS is unavailable")
    return device


def _load_model(run_dir: Path, model_kind: str):
    model_path = run_dir / "mfa_model.pt"
    if model_kind == "ard":
        from dalg.models.adaptive_q.mfa_ard import load_mfa_ard

        return load_mfa_ard(model_path, map_location="cpu")
    if model_kind == "hddc":
        from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc

        return load_mfa_hddc(model_path, map_location="cpu")
    raise ValueError(
        "adaptive_q_toy evaluation supports model.kind 'ard' or 'hddc', "
        f"not {model_kind!r}"
    )


@torch.no_grad()
def _mean_nll(
    model,
    *,
    shard_dir: Path,
    layer: int,
    positions: list[int],
    batch_size: int,
    drop_prefix: int,
    device: torch.device,
) -> float:
    if not positions:
        raise ValueError("cannot evaluate NLL on an empty split")
    dataset = ActivationBatchDataset(
        shard_dir,
        layer=layer,
        row_subset=positions,
        batch_size=batch_size,
        drop_prefix=drop_prefix,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=0,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    total_nll = 0.0
    total_points = 0
    with model.inference_cache():
        for batch in loader:
            x = batch.to(device, non_blocking=(device.type == "cuda"))
            total_nll += float(model.nll(x).item()) * int(x.shape[0])
            total_points += int(x.shape[0])
    if total_points == 0:
        raise ValueError("cannot evaluate NLL on an empty split")
    return total_nll / total_points


def evaluate_adaptive_q_toy(
    run_dir: str | Path,
    *,
    shard_dir: str | Path,
    layer: int,
    model_kind: str,
    assignments_path: str | Path | None = None,
    batch_size: int = 4096,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate one ARD/HDDC run against planted toy-manifold structure."""
    run_dir = Path(run_dir)
    assignments_path = (
        Path(assignments_path)
        if assignments_path is not None
        else run_dir / "mfa_model_assignments.pt"
    )
    required = [
        run_dir / "config.json",
        run_dir / "val_indices.json",
        assignments_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing adaptive-q evaluation artifacts: {missing}")

    clean_shard_dir, subset_spec = split_shard_dir_spec(str(shard_dir))
    shard_config = json.loads((clean_shard_dir / "config.json").read_text())
    if shard_config.get("source_kind") != "toy_manifolds":
        raise ValueError("adaptive_q_toy requires shards from save_toy_manifold_shards")
    window = int(shard_config["window"])
    drop_prefix = int(shard_config.get("drop_prefix", 0))
    if window != 1 or drop_prefix != 0:
        raise ValueError("adaptive_q_toy expects one activation per row")

    metadata_path = clean_shard_dir / shard_config["manifold_metadata"]
    manifold_metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    all_manifold_ids = manifold_metadata["row_manifold_ids"].reshape(-1).long()

    meta_index = load_meta_index(clean_shard_dir, layer=layer)
    positions = resolve_spec_positions(
        meta_index,
        subset_spec,
        window=window,
        drop_prefix=drop_prefix,
    )
    if all_manifold_ids.numel() != len(meta_index):
        raise ValueError(
            "toy manifold labels are not aligned with activation metadata: "
            f"labels={all_manifold_ids.numel()}, rows={len(meta_index)}"
        )
    row_manifold_ids = all_manifold_ids[torch.as_tensor(positions, dtype=torch.long)]

    assignment_bundle = torch.load(
        assignments_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    assignments = assignment_bundle["assignments"].reshape(-1).long()
    cluster_sizes = assignment_bundle["cluster_sizes"].reshape(-1).long()
    if assignment_bundle.get("subset_spec") != subset_spec:
        raise ValueError(
            "assignment subset does not match evaluation dataset: "
            f"assignments={assignment_bundle.get('subset_spec')!r}, "
            f"evaluation={subset_spec!r}"
        )
    if assignments.numel() != len(positions):
        raise ValueError(
            "assignments must cover the selected canonical stream: "
            f"assignments={assignments.numel()}, selected rows={len(positions)}"
        )
    if int(cluster_sizes.sum()) != assignments.numel():
        raise ValueError("cluster_sizes does not sum to the assignment count")

    model = _load_model(run_dir, model_kind)
    if cluster_sizes.numel() != model.K or int(assignment_bundle["K"]) != model.K:
        raise ValueError("assignment K does not match the loaded model")
    if not torch.equal(torch.bincount(assignments, minlength=model.K), cluster_sizes):
        raise ValueError("cluster_sizes is inconsistent with assignments")

    split_info = json.loads((run_dir / "val_indices.json").read_text())
    val_global_rows = set(split_info["val_global_rows"])
    val_positions = [
        position
        for position in positions
        if meta_index[position]["global_row"] in val_global_rows
    ]
    val_position_set = set(val_positions)
    train_positions = [position for position in positions if position not in val_position_set]
    if len(train_positions) != int(split_info["train_rows"]):
        raise ValueError("reconstructed training split does not match val_indices.json")
    if len(val_positions) != int(split_info["val_rows"]):
        raise ValueError("reconstructed validation split does not match val_indices.json")
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device).eval()
    train_nll = _mean_nll(
        model,
        shard_dir=clean_shard_dir,
        layer=layer,
        positions=train_positions,
        batch_size=batch_size,
        drop_prefix=drop_prefix,
        device=resolved_device,
    )
    val_nll = _mean_nll(
        model,
        shard_dir=clean_shard_dir,
        layer=layer,
        positions=val_positions,
        batch_size=batch_size,
        drop_prefix=drop_prefix,
        device=resolved_device,
    )

    true_ids = row_manifold_ids.numpy()
    predicted_ids = assignments.numpy()
    clustering = {
        "homogeneity": float(homogeneity_score(true_ids, predicted_ids)),
        "completeness": float(completeness_score(true_ids, predicted_ids)),
        "adjusted_rand_index": float(adjusted_rand_score(true_ids, predicted_ids)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(true_ids, predicted_ids)
        ),
    }

    w_ranks = (
        model.effective_ranks().cpu().long()
        if model_kind == "ard"
        else model.component_ranks.cpu().long()
    )
    num_manifolds = int(manifold_metadata["num_manifolds"])
    component_by_manifold = torch.bincount(
        assignments * num_manifolds + row_manifold_ids,
        minlength=model.K * num_manifolds,
    ).reshape(model.K, num_manifolds)
    dominant_manifold = component_by_manifold.argmax(dim=1)
    manifold_ranks = torch.tensor(
        [int(item["intrinsic_dim"]) for item in manifold_metadata["manifolds"]],
        dtype=torch.long,
    )
    live = cluster_sizes > 0
    live_w_ranks = w_ranks[live]
    matched_ranks = manifold_ranks[dominant_manifold[live]]
    if live_w_ranks.numel() == 0:
        raise ValueError("cannot evaluate ranks because every component is empty")
    rank_error = live_w_ranks - matched_ranks

    return {
        "evaluation": "adaptive_q_toy",
        "model_kind": model_kind,
        "K": int(model.K),
        "q_max": int(model.q),
        "dataset": {
            "shard_dir": str(clean_shard_dir),
            "subset_spec": subset_spec,
            "layer": int(layer),
            "selected_rows": len(positions),
            "train_rows": len(train_positions),
            "validation_rows": len(val_positions),
        },
        "nll": {"train": train_nll, "validation": val_nll},
        "clustering": clustering,
        "components": {
            "live": int(live.sum()),
            "dead": int((~live).sum()),
        },
        "rank": {
            "mean_learned_live": float(live_w_ranks.float().mean()),
            "exact_match": float((rank_error == 0).float().mean()),
            "within_one_match": float((rank_error.abs() <= 1).float().mean()),
            "mean_absolute_error": float(rank_error.abs().float().mean()),
        },
    }
