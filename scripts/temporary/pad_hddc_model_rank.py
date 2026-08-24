"""One-off adapter that pads an HDDC model to a larger masked q_max.

This belongs to the oracle circle/helix experiment, not the training API. The
output is an ordinary MFA_HDDC model whose shape already matches the target
training configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, load_mfa_hddc, save_mfa_hddc


@torch.no_grad()
def pad_hddc_model_rank(source: MFA_HDDC, target_rank: int) -> MFA_HDDC:
    target_rank = int(target_rank)
    if target_rank <= source.q:
        raise ValueError(f"target rank must exceed source q={source.q}")
    if target_rank > source.D:
        raise ValueError(f"target rank q={target_rank} exceeds D={source.D}")
    if getattr(source, "_rotation_on", False):
        raise ValueError("source model must not have an active factor rotation")

    padded = MFA_HDDC(
        centroids=source.mu.detach().clone(),
        rank=target_rank,
        psi_per_component=source.psi_per_component,
        isotropic_psi=source.isotropic_psi,
        eps_floor=source._eps,
    )
    padded.mu.copy_(source.mu)
    padded.psi_rho.copy_(source.psi_rho)
    padded.pi_logits.copy_(source.pi_logits)
    padded.dir_raw[:, :, :source.q].copy_(source.dir_raw)
    padded.scale_rho[:, :source.q].copy_(source.scale_rho)
    padded.rank_mask.zero_()
    padded.rank_mask[:, :source.q].copy_(source.rank_mask)
    return padded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    torch.manual_seed(args.seed)
    source = load_mfa_hddc(args.input, map_location="cpu")
    padded = pad_hddc_model_rank(source, args.target_rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_mfa_hddc(
        padded,
        str(args.output),
        extra={
            "construction": "temporary_rank_padding",
            "source_model": str(args.input.resolve()),
            "source_q": source.q,
            "target_q": padded.q,
            "seed": args.seed,
        },
    )
    print(
        f"saved padded HDDC model K={padded.K} D={padded.D} "
        f"q={source.q}->{padded.q} to {args.output}"
    )


if __name__ == "__main__":
    main()
