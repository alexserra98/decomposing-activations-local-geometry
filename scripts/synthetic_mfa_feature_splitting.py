"""Feature splitting & covariance reconstruction over a synthetic MFA K/q sweep.

Heavy post-hoc analysis extracted from
``notebooks/synthetic_mfa_qk_sweep_results.ipynb`` so it can run on a cluster
instead of inside Jupyter. For every fitted MFA already on disk
(``K*_q*_seed*/mfa_model.pt``) with ``K_fit >= --k-min`` it:

1. maps every fitted component to the ground-truth (g.t.) cluster that owns most
   of the test points it captures (label-based, using ``y_test``), giving each
   g.t. cluster a *split factor* = number of fitted components mapped to it;
2. merges the fitted components mapped to one g.t. cluster (moment-matched into a
   single Gaussian ``N(mu_bar, C_merged)``) and compares it to the g.t. Gaussian
   ``N(mu_gt, C_gt)`` via the relative Frobenius covariance error, the
   Bhattacharyya distance and the symmetric KL, each against a single-component
   baseline (the dominant fitted component, no merge).

Results are written next to the models so the notebook can just plot them:

* ``<run_dir>/feature_splitting.csv`` — per-fit aggregate scalars
* ``<run_dir>/feature_splitting.pt`` — ``{"agg", "split_factor_hist",
  "eig_spectra", "meta"}`` for the richer plots (histograms, eigenspectra)

Example::

    PYTHONPATH=src python scripts/synthetic_mfa_feature_splitting.py \
        --run-dir dalg-cache/qk_sweep_exploration/Ktrue1000_qtrue20 --k-min 750
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dalg.models.mfa import MFA, load_mfa  # noqa: E402

DEFAULT_RUN_DIR = REPO_ROOT / "dalg-cache/qk_sweep_exploration/Ktrue1000_qtrue20"
FIT_DIR_RE = re.compile(r"^K(\d+)_q(\d+)_seed(\d+)$")

METRIC_COLS = [
    "frob_merged", "frob_single", "db_merged", "db_single",
    "skl_merged", "skl_single", "mean_err",
]


# --------------------------------------------------------------------------- #
# Gaussian helpers (full DxD covariance: C = W W^T + diag(psi))
# --------------------------------------------------------------------------- #
def _component_cov(W_k: torch.Tensor, psi_k: torch.Tensor) -> torch.Tensor:
    """Full covariance C = W W^T + diag(psi) for one component, (D, D)."""
    C = W_k @ W_k.T
    C.diagonal().add_(psi_k)
    return C


def _chol_logdet(C: torch.Tensor, jitter: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Cholesky factor (with jitter) and log|C| for an SPD matrix."""
    eye = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
    L = torch.linalg.cholesky(C + jitter * eye)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    return L, logdet


def gaussian_bhattacharyya(mu0, C0, mu1, C1, *, jitter: float) -> float:
    """Bhattacharyya distance D_B between two full-covariance Gaussians."""
    _, logdet0 = _chol_logdet(C0, jitter)
    _, logdet1 = _chol_logdet(C1, jitter)
    Lbar, logdet_bar = _chol_logdet(0.5 * (C0 + C1), jitter)
    delta = (mu1 - mu0).unsqueeze(-1)
    y = torch.cholesky_solve(delta, Lbar)               # Cbar^{-1} delta
    db_mean = (delta * y).sum() / 8.0
    db_cov = 0.5 * logdet_bar - 0.25 * (logdet0 + logdet1)
    return float(db_mean + db_cov)


def gaussian_sym_kl(mu0, C0, mu1, C1, *, jitter: float) -> float:
    """Symmetric KL = (KL(0||1) + KL(1||0)) / 2 for full-covariance Gaussians."""
    L0, _ = _chol_logdet(C0, jitter)
    L1, _ = _chol_logdet(C1, jitter)
    tr10 = torch.diagonal(torch.cholesky_solve(C0, L1)).sum()   # tr(C1^{-1} C0)
    tr01 = torch.diagonal(torch.cholesky_solve(C1, L0)).sum()   # tr(C0^{-1} C1)
    delta = (mu1 - mu0).unsqueeze(-1)
    maha = (delta * (torch.cholesky_solve(delta, L0) + torch.cholesky_solve(delta, L1))).sum()
    D = C0.shape[-1]
    return float(0.25 * (tr10 + tr01 + maha - 2.0 * D))


# --------------------------------------------------------------------------- #
# Ground truth + per-fit analysis
# --------------------------------------------------------------------------- #
class GroundTruth:
    """Cached per-cluster Gaussian parameters of the data-generating MFA."""

    def __init__(self, dataset_path: Path, device: torch.device):
        ds = torch.load(dataset_path, map_location="cpu", weights_only=False)
        self.x_test = ds["x_test"]
        self.y_test = ds["y_test"].long()
        state = ds["truth"]
        truth = MFA(
            state["mu"].detach().cpu(),
            rank=int(state["dir_raw"].shape[-1]),
            psi_per_component=state["psi_rho"].ndim == 2,
        )
        truth.load_state_dict(state)
        truth.eval()
        self.K_true = truth.K
        self.q_true = truth.q
        self.mu = truth.mu.detach().to(device)            # (K_true, D)
        self.W = truth.W.detach().to(device)              # (K_true, D, q_true)
        self.psi = truth._psi().detach().to(device)       # (K_true, D)


@torch.no_grad()
def _hard_assign(model, x, device, *, batch: int) -> torch.Tensor:
    """argmax responsibility per test point, (N,) on CPU.

    Small batches keep the (batch, K, q) E-step intermediates off the GPU peak.
    """
    out = []
    with model.inference_cache():
        for s in range(0, x.shape[0], batch):
            xb = x[s : s + batch].to(device)
            out.append(model.responsibilities(xb).argmax(dim=1).cpu())
            del xb
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(out)


@torch.no_grad()
def analyze_fit(
    fit_dir: Path,
    K_fit: int,
    q_fit: int,
    seed: int,
    gt: GroundTruth,
    device: torch.device,
    *,
    resp_batch: int,
    jitter: float,
    eig_topn: int,
):
    """Map fitted comps -> g.t. clusters, then merge & compare to ground truth.

    Returns (aggregate_row, split_factors[K_true], eig_record).
    """
    model = load_mfa(fit_dir / "mfa_model.pt", map_location=device, device=device)
    model.eval()
    assign = _hard_assign(model, gt.x_test, device, batch=resp_batch)
    K = model.K

    # contingency owner: fitted comp j -> g.t. cluster with most of its points
    flat = gt.y_test * K + assign
    counts = torch.bincount(flat, minlength=gt.K_true * K).reshape(gt.K_true, K).double()
    won = counts.sum(dim=0)                      # points captured by each fitted comp
    owner = counts.argmax(dim=0)                 # (K,) g.t. owner per fitted comp
    owner[won == 0] = -1                         # ignore comps that won nothing

    mu_f = model.mu.detach().to(device)
    W_f = model.W.detach().to(device)
    psi_f = model._psi().detach().to(device)
    pi_f = torch.softmax(model.pi_logits.detach(), dim=0).to(device)

    split_factors = torch.zeros(gt.K_true, dtype=torch.long)
    records = []
    # remember the merged/single/gt covariances of the most-split cluster for the
    # eigenspectrum plot, so the notebook never has to reload a model.
    best_n = -1
    best_eig = None
    for k in range(gt.K_true):
        idx = (owner == k).nonzero(as_tuple=True)[0]
        split_factors[k] = len(idx)
        if len(idx) == 0:
            continue

        w = pi_f[idx]
        w = w / w.sum()
        mus = mu_f[idx]                                   # (n, D)
        mu_bar = (w[:, None] * mus).sum(dim=0)            # (D,)
        Wi = W_f[idx]                                     # (n, D, q)
        C_merged = torch.einsum("n,ndp,nep->de", w, Wi, Wi)
        C_merged += torch.einsum("n,nd,ne->de", w, mus - mu_bar, mus - mu_bar)
        C_merged.diagonal().add_((w[:, None] * psi_f[idx]).sum(dim=0))

        # dominant single component in this split (most points captured)
        j = idx[counts[k, idx].argmax()]
        C_single = _component_cov(W_f[j], psi_f[j])
        mu_single = mu_f[j]

        C_gt = _component_cov(gt.W[k], gt.psi[k])
        mu_gt = gt.mu[k]
        gt_norm = torch.linalg.norm(C_gt)

        records.append(
            {
                "k": k,
                "n": int(len(idx)),
                "frob_merged": float(torch.linalg.norm(C_merged - C_gt) / gt_norm),
                "frob_single": float(torch.linalg.norm(C_single - C_gt) / gt_norm),
                "db_merged": gaussian_bhattacharyya(mu_bar, C_merged, mu_gt, C_gt, jitter=jitter),
                "db_single": gaussian_bhattacharyya(mu_single, C_single, mu_gt, C_gt, jitter=jitter),
                "skl_merged": gaussian_sym_kl(mu_bar, C_merged, mu_gt, C_gt, jitter=jitter),
                "skl_single": gaussian_sym_kl(mu_single, C_single, mu_gt, C_gt, jitter=jitter),
                "mean_err": float(torch.linalg.norm(mu_bar - mu_gt) / torch.linalg.norm(mu_gt)),
            }
        )

        if len(idx) > best_n:
            def _ev(C):
                return torch.linalg.eigvalsh(C).flip(0)[:eig_topn].cpu().tolist()

            best_n = len(idx)
            best_eig = {
                "cluster": k,
                "n": int(len(idx)),
                "gt": _ev(C_gt),
                "merged": _ev(C_merged),
                "single": _ev(C_single),
            }

    del model, mu_f, W_f, psi_f, pi_f
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pk = pd.DataFrame(records)
    covered = split_factors > 0
    agg = {
        "K_fit": int(K_fit),
        "q_fit": int(q_fit),
        "seed": int(seed),
        "n_covered": int(covered.sum()),
        "coverage": float(covered.float().mean()),
        "mean_split_factor": float(split_factors[covered].double().mean()) if covered.any() else float("nan"),
        "frac_split": float((split_factors >= 2).sum() / max(int(covered.sum()), 1)),
        **{f"{c}_mean": float(pk[c].mean()) if len(pk) else float("nan") for c in METRIC_COLS},
    }
    return agg, split_factors, best_eig


# --------------------------------------------------------------------------- #
# Discovery / IO
# --------------------------------------------------------------------------- #
def discover_fits(run_dir: Path, *, k_min: int, only: list[tuple[int, int]] | None):
    """List (fit_dir, K, q, seed) for every model on disk passing the filters."""
    fits = []
    for path in sorted(run_dir.glob("K*_q*_seed*")):
        m = FIT_DIR_RE.match(path.name)
        if not m or not (path / "mfa_model.pt").exists():
            continue
        K, q, seed = (int(g) for g in m.groups())
        if K < k_min:
            continue
        if only is not None and (K, q) not in only:
            continue
        fits.append((path, K, q, seed))
    return fits


def _resolve_dataset_path(run_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        # fall back to any per-fit config
        candidates = sorted(run_dir.glob("K*_q*_seed*/config.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No config.json under {run_dir}; pass --dataset-path explicitly."
            )
        cfg_path = candidates[0]
    return Path(json.loads(cfg_path.read_text())["dataset_path"])


def _parse_only(value: str | None) -> list[tuple[int, int]] | None:
    if not value:
        return None
    pairs = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        k, q = token.split(":")
        pairs.append((int(k), int(q)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help="Run directory holding K*_q*_seed*/mfa_model.pt and config.json.")
    parser.add_argument("--dataset-path", type=Path, default=None,
                        help="Override the synthetic dataset path (else read from config.json).")
    parser.add_argument("--k-min", type=int, default=750,
                        help="Only analyse fits with K_fit >= this (splitting expected for K_fit > K_true).")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated K:q pairs to (re)compute, e.g. '1250:5,1500:10'.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resp-batch", type=int, default=512,
                        help="Batch size for the responsibility/argmax pass (lower if you hit OOM).")
    parser.add_argument("--eig-topn", type=int, default=40,
                        help="How many top eigenvalues to store for the spectrum plot.")
    parser.add_argument("--jitter", type=float, default=1e-6)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        device = torch.device("cpu")

    run_dir = args.run_dir
    dataset_path = _resolve_dataset_path(run_dir, args.dataset_path)
    only = _parse_only(args.only)
    fits = discover_fits(run_dir, k_min=args.k_min, only=only)
    if not fits:
        raise SystemExit(f"No matching fits under {run_dir} (k_min={args.k_min}, only={args.only}).")

    print(f"run_dir={run_dir}", flush=True)
    print(f"dataset={dataset_path}", flush=True)
    print(f"device={device}  fits={len(fits)}  resp_batch={args.resp_batch}", flush=True)

    gt = GroundTruth(dataset_path, device)
    print(f"K_true={gt.K_true} q_true={gt.q_true} x_test={tuple(gt.x_test.shape)}", flush=True)

    # merge with any previously computed results so --only updates in place
    out_pt = run_dir / "feature_splitting.pt"
    prev = torch.load(out_pt, map_location="cpu", weights_only=False) if out_pt.exists() else {}
    agg_by_key = {(r["K_fit"], r["q_fit"], r["seed"]): r for r in prev.get("agg", [])}
    hist = dict(prev.get("split_factor_hist", {}))
    eig = dict(prev.get("eig_spectra", {}))

    for n, (fit_dir, K, q, seed) in enumerate(fits, 1):
        t = time.time()
        agg, sf, best_eig = analyze_fit(
            fit_dir, K, q, seed, gt, device,
            resp_batch=args.resp_batch, jitter=args.jitter, eig_topn=args.eig_topn,
        )
        key = (K, q, seed)
        agg_by_key[key] = agg
        hist[key] = sf
        if best_eig is not None:
            eig[key] = best_eig
        print(
            f"[{n}/{len(fits)}] K={K:>4} q={q:>3} seed={seed} | "
            f"coverage={agg['coverage']:.3f} mean_split={agg['mean_split_factor']:.3f} "
            f"frac>=2={agg['frac_split']:.3f} | frob m/s={agg['frob_merged_mean']:.3f}/{agg['frob_single_mean']:.3f} "
            f"db m/s={agg['db_merged_mean']:.3f}/{agg['db_single_mean']:.3f} | {time.time() - t:.1f}s",
            flush=True,
        )

    aggs = [agg_by_key[k] for k in sorted(agg_by_key)]
    out_csv = run_dir / "feature_splitting.csv"
    pd.DataFrame(aggs).sort_values(["K_fit", "q_fit", "seed"]).to_csv(out_csv, index=False)
    torch.save(
        {
            "agg": aggs,
            "split_factor_hist": hist,
            "eig_spectra": eig,
            "meta": {
                "K_true": gt.K_true,
                "q_true": gt.q_true,
                "dataset_path": str(dataset_path),
                "run_dir": str(run_dir),
            },
        },
        out_pt,
    )
    print(f"\nsaved {len(aggs)} fits -> {out_csv}\n           -> {out_pt}", flush=True)


if __name__ == "__main__":
    main()
