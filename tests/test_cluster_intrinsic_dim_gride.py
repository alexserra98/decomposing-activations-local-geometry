from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset

from dalg.analysis.cluster_intrinsic_dim import (
    compute_intrinsic_dims_from_assignments,
    intrinsic_dim_gride,
)
from dalg.cli.run_metrics import build_parser


def _is_undefined(values: torch.Tensor) -> bool:
    return values.shape == (1,) and bool(torch.isnan(values[0]))


def test_gride_swiss_roll_returns_full_multiscale_output() -> None:
    X, _ = make_swiss_roll(n_samples=1000, noise=0.0, random_state=0)

    ids, errors, scales = intrinsic_dim_gride(
        torch.from_numpy(X),
        range_max=64,
    )

    assert ids.shape == errors.shape == scales.shape == (6,)
    assert ids.dtype == errors.dtype == scales.dtype == torch.float64
    assert torch.isfinite(ids).all()
    assert torch.isfinite(errors).all()
    assert torch.isfinite(scales).all()
    assert 1.5 < ids.median().item() < 2.5


def test_gride_one_point_and_constant_clusters_are_undefined() -> None:
    for X in (torch.zeros(1, 3), torch.zeros(8, 3)):
        ids, errors, scales = intrinsic_dim_gride(X)
        assert _is_undefined(ids)
        assert _is_undefined(errors)
        assert _is_undefined(scales)


def test_gride_near_coincident_distinct_points_are_finite() -> None:
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.normal(scale=1e-9, size=(64, 3)))

    ids, errors, scales = intrinsic_dim_gride(X, range_max=32)

    assert ids.shape == errors.shape == scales.shape == (5,)
    assert torch.isfinite(ids).all()
    assert torch.isfinite(errors).all()
    assert torch.isfinite(scales).all()
    assert torch.all(scales > 0)
    assert scales.max().item() < 1e-7


def test_per_cluster_gride_outputs_survive_save_round_trip(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    clusters = [
        torch.randn(8, 3, generator=generator),
        torch.zeros(1, 3),
        10.0 + torch.randn(8, 3, generator=generator) * 1e-6,
    ]
    X = torch.cat(clusters)
    assignments = torch.cat(
        [torch.full((cluster.shape[0],), k) for k, cluster in enumerate(clusters)]
    ).long()
    assignments_path = tmp_path / "assignments.pt"
    torch.save(
        {
            "assignments": assignments,
            "cluster_sizes": torch.tensor([8, 1, 8]),
            "K": 3,
        },
        assignments_path,
    )

    results = compute_intrinsic_dims_from_assignments(
        None,
        DataLoader(TensorDataset(X), batch_size=4, shuffle=False),
        assignments_path,
        min_population=1,
        max_samples=8,
        store_dtype=torch.float32,
        pca_device="cpu",
        compute_gride=True,
        gride_range_max=4,
    )
    save_path = tmp_path / "intrinsic_dims.pt"
    torch.save(results, save_path)
    saved = torch.load(save_path, map_location="cpu", weights_only=True)

    assert saved["gride_enabled"] is True
    assert saved["gride_range_max"] == 4
    for key in (
        "gride_intrinsic_dims",
        "gride_intrinsic_dim_errors",
        "gride_scales",
    ):
        assert len(saved[key]) == 3
        assert saved[key][0].shape == (2,)
        assert _is_undefined(saved[key][1])
        assert saved[key][2].shape == (2,)
    assert torch.isfinite(saved["gride_intrinsic_dims"][0]).all()
    assert torch.isfinite(saved["gride_intrinsic_dims"][2]).all()


def test_gride_is_default_on_and_can_be_disabled(tmp_path: Path) -> None:
    parser = build_parser()
    default_args = parser.parse_args(
        ["intrinsic-dim", "--assignments-path", "assignments.pt", "--act-dir", "acts"]
    )
    disabled_args = parser.parse_args(
        [
            "intrinsic-dim",
            "--assignments-path",
            "assignments.pt",
            "--act-dir",
            "acts",
            "--no-gride",
        ]
    )
    assert default_args.no_gride is False
    assert default_args.gride_range_max == 2048
    assert disabled_args.no_gride is True

    assignments_path = tmp_path / "assignments.pt"
    torch.save(
        {
            "assignments": torch.zeros(4, dtype=torch.long),
            "cluster_sizes": torch.tensor([4]),
            "K": 1,
        },
        assignments_path,
    )
    X = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    results = compute_intrinsic_dims_from_assignments(
        None,
        DataLoader(TensorDataset(X), batch_size=2, shuffle=False),
        assignments_path,
        min_population=1,
        max_samples=4,
        pca_device="cpu",
        compute_gride=False,
    )
    assert results["gride_enabled"] is False
    assert results["gride_range_max"] == 2048
    assert "gride_intrinsic_dims" not in results
    assert "gride_intrinsic_dim_errors" not in results
    assert "gride_scales" not in results
