"""Extract trained MFA means `mu` into a centroids file.

Saves `<run_dir>/mfa_mu.pt` as ``{"centroids": (K, D) float32, ...}`` so it can
be passed to ``dalg.analysis.nearest_centroid_assignments --centroids-path``.
Handles both vanilla runs (``mfa_model.pt``) and component-sharded runs
(``mfa_model_shards.json``) via the ``load_mfa`` fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dalg.models.mfa import load_mfa


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MFA means mu as a centroids file")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args()

    out_path = args.out_path if args.out_path is not None else args.run_dir / "mfa_mu.pt"
    if out_path.exists() and not args.force:
        print(f"Output already exists, skipping: {out_path}")
        return

    model_path = args.run_dir / "mfa_model.pt"
    model = load_mfa(model_path, map_location="cpu")
    mu = model.mu.detach().float().cpu()
    K, D = mu.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "centroids": mu,
            "source_model": str(model_path),
            "K": int(K),
            "D": int(D),
        },
        out_path,
    )
    print(f"Saved mu ({K}, {D}) to {out_path}")


if __name__ == "__main__":
    main()
