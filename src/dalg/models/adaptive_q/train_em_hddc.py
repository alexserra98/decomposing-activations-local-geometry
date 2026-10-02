"""Full-data EM for [a_ij, b, Q_i, d_i], streamed over activation batches.

Responsibilities are frozen for a complete pass. Float64 residual moments
feed the existing HDDC M-step; no optimizer or stochastic averaging is used.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from dalg.init.projected_knn import KMeansTorch
from .hddc_surgery import SurgeryConfig, reconstruct_components
from .mfa_hddc import ComponentShardedMFA_HDDC, save_mfa_hddc
from .train_hddc import _atomic_torch_save, _cpu_state_dict, _wandb_active


@dataclass(frozen=True)
class EMConfig:
    threshold: float = 0.01
    component_chunk_size: int = 32
    eig_batch_size: int = 128
    tol: float = 1e-5

    def validate(self):
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("EM Cattell threshold must be finite and positive")
        if self.component_chunk_size <= 0 or self.eig_batch_size <= 0:
            raise ValueError("EM chunk sizes must be positive")
        if not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError("EM tolerance must be finite and non-negative")


@dataclass
class EMStatistics:
    counts: torch.Tensor
    residual_sum: torch.Tensor | None
    scatter: torch.Tensor | None
    n_rows: int
    nll: float | None


def _validate_model(model):
    if isinstance(model, ComponentShardedMFA_HDDC) or not model.shared_b:
        raise ValueError("EM requires a single-process shared-b HDDC model")
    if not 1 <= model.q < model.D:
        raise ValueError("EM requires 1 <= q_max < D")
    if model._rotation_on:
        raise ValueError("EM does not accept an active factor rotation")
    if model.mu.device.type not in {"cpu", "cuda"}:
        raise ValueError("EM float64 computations require CPU or CUDA")


@torch.no_grad()
def expectation_step(
    model, loader, *, component_chunk_size=32, hard=False,
    collect_statistics=True, expected_rows=None,
) -> EMStatistics:
    """Stream exact soft moments, or a nearest-centroid initialization pass.

    Moments are centered on the current (frozen) means. Hard initialization
    uses indexed additions, costing O(N D^2) rather than O(N K D^2).
    With collect_statistics=False only likelihood and effective counts remain.
    """
    _validate_model(model)
    if component_chunk_size <= 0:
        raise ValueError("component_chunk_size must be positive")
    device = model.mu.device
    K, D = model.K, model.D
    mu = model.mu.detach().double()
    counts = torch.zeros(K, device=device, dtype=torch.float64)
    first = mu.new_zeros(K, D) if collect_statistics else None
    scatter = mu.new_zeros(K, D, D) if collect_statistics else None
    total_log_prob = mu.new_zeros(())
    n_rows = 0
    kmeans = KMeansTorch(k=K, metric="euclidean", device=device) if hard else None
    log_pi = model.pi_logits.detach().double().log_softmax(0)

    with model.inference_cache(
        enabled=not hard, dtype=torch.float64, component_chunk_size=component_chunk_size,
    ):
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            if x.ndim != 2 or x.shape[1] != D or not torch.isfinite(x).all():
                raise ValueError(f"EM requires finite activation batches of shape (B, {D})")
            x = x.to(device=device, dtype=torch.float64)
            n_rows += x.shape[0]
            if hard:
                labels = kmeans._assign_streamed(x, mu)
                counts += torch.bincount(labels, minlength=K)
                if collect_statistics:
                    centered = x - mu[labels]
                    first.index_add_(0, labels, centered)
                    scatter.index_add_(0, labels, centered[:, :, None] * centered[:, None, :])
                continue

            joint = model.log_prob_components(x) + log_pi[None]
            normalizer = torch.logsumexp(joint, dim=1)
            responsibilities = (joint - normalizer[:, None]).exp()
            total_log_prob += normalizer.sum()
            counts += responsibilities.sum(0)
            if collect_statistics:
                for start in range(0, K, component_chunk_size):
                    stop = min(start + component_chunk_size, K)
                    centered = x[None] - mu[start:stop, None]
                    weighted = centered * responsibilities[:, start:stop].T[:, :, None]
                    first[start:stop] += weighted.sum(1)
                    scatter[start:stop] += weighted.transpose(1, 2) @ centered

    if n_rows == 0:
        raise ValueError("EM requires a non-empty pass")
    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(f"EM counted {n_rows} training activations, expected {expected_rows}")
    nll = None if hard else -float(total_log_prob) / n_rows
    if not torch.isfinite(counts).all() or (nll is not None and not math.isfinite(nll)):
        raise ValueError("EM E-step produced non-finite likelihood or responsibilities")
    return EMStatistics(counts, first, scatter, n_rows, nll)


@torch.no_grad()
def maximization_step(model, statistics: EMStatistics, cfg: EMConfig):
    """Update live components and give zero weight to zero-membership components."""
    _validate_model(model)
    cfg.validate()
    if statistics.residual_sum is None or statistics.scatter is None:
        raise ValueError("M-step requires first and second moments")
    return reconstruct_components(
        model, statistics.counts, statistics.residual_sum, statistics.scatter,
        SurgeryConfig(threshold=cfg.threshold, min_count=0, eig_batch_size=cfg.eig_batch_size),
        full_mixture_update=True,
    )


@torch.no_grad()
def train_em_hddc(
    model, loader, *, cfg=None, epochs=100, initialize=True, expected_rows=None,
    val_loader=None, val_tensor=None, batch_size=2048, out_dir=None,
    early_stop_patience=None, early_stop_min_delta=0.0,
    epoch_snapshot_every=0, log=print,
):
    """Fit EM, checkpoint each evaluated state, and restore the best model.

    Epoch zero is the initialized model. Each subsequent epoch is one full
    M-step. Its following E-pass both scores that model and collects moments
    for the next update. The final pass needs only likelihoods. Checkpoints
    hold the current state separately from the best state, without an optimizer.
    A restarted pass is recomputed from the last completed iteration.
    """
    cfg = cfg or EMConfig()
    cfg.validate()
    _validate_model(model)
    if epochs <= 0:
        raise ValueError("EM epochs must be positive")
    if early_stop_patience is not None and early_stop_patience <= 0:
        raise ValueError("early_stop_patience must be positive")
    if not math.isfinite(early_stop_min_delta) or early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be finite and non-negative")
    model.eval()
    model_config = {
        "K": model.K, "D": model.D, "q": model.q,
        "eps_floor": model._eps, "dtype": str(model.mu.dtype),
    }
    directory = Path(out_dir) if out_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt" if directory is not None else None
    history = []
    epoch = 0
    best_metric = progress_metric = float("inf")
    best_state = None
    best_epoch = stable_steps = without_improvement = 0
    rank_changes = 0
    m_step_seconds = 0.0
    update_summary = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("fit_method") != "em":
            raise ValueError("Cannot resume an Adam checkpoint as EM; use init_model_path with mfa_model.pt")
        if checkpoint["model_config"] != model_config:
            raise ValueError("Cannot resume EM with different model dimensions, dtype or numerical floor")
        for key in ("threshold", "tol"):
            if checkpoint["em_config"][key] != getattr(cfg, key):
                raise ValueError(f"Cannot change EM {key} when resuming")
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        best_state, best_metric = checkpoint["best_state"], checkpoint["best_metric"]
        best_epoch = checkpoint["best_epoch"]
        progress_metric = checkpoint["progress_metric"]
        stable_steps = checkpoint["stable_steps"]
        without_improvement = checkpoint["without_improvement"]
        log(f"[em] resumed iteration {epoch}; best iteration {best_epoch}")
    elif initialize:
        initial = expectation_step(
            model, loader, hard=True, component_chunk_size=cfg.component_chunk_size,
            expected_rows=expected_rows,
        )
        maximization_step(model, initial, cfg)
        del initial
        log("[em] initialized weights, means, ranks and covariances from nearest-centroid memberships")

    if model.mu.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.mu.device)

    while True:
        started = time.perf_counter()
        statistics = expectation_step(
            model, loader, component_chunk_size=cfg.component_chunk_size,
            collect_statistics=epoch < epochs, expected_rows=expected_rows,
        )
        e_step_seconds = time.perf_counter() - started
        validation = val_tensor.split(batch_size) if val_tensor is not None else val_loader
        val_nll = None
        if validation is not None:
            val_nll = expectation_step(
                model, validation, component_chunk_size=cfg.component_chunk_size,
                collect_statistics=False,
            ).nll
        metric = statistics.nll if val_nll is None else val_nll
        ranks = model.component_ranks
        new_record = not history or history[-1]["iteration"] != epoch
        if new_record:
            if history:
                previous = history[-1]["train_nll"]
                relative = abs(statistics.nll - previous) / max(1.0, abs(previous))
                stable_steps = stable_steps + 1 if cfg.tol > 0 and relative < cfg.tol and rank_changes == 0 else 0
            if metric < best_metric:
                best_metric, best_epoch = metric, epoch
                best_state = _cpu_state_dict(model)
                if directory is not None:
                    save_mfa_hddc(model, str(directory / "mfa_model.pt"), extra={"fit_method": "em", "iteration": epoch})
            if metric < progress_metric - early_stop_min_delta:
                progress_metric, without_improvement = metric, 0
            else:
                without_improvement += 1
            record = {
                "iteration": epoch, "train_nll": statistics.nll, "val_nll": val_nll,
                "b": float(model._psi()[0, 0]), "rank_min": int(ranks.min()),
                "rank_max": int(ranks.max()), "rank_mean": float(ranks.double().mean()),
                "rank_histogram": torch.bincount(ranks, minlength=model.q + 1).tolist(),
                "dead_components": int(torch.isneginf(model.pi_logits).sum()),
                "rank_changes": rank_changes, "membership_min": float(statistics.counts.min()),
                "membership_max": float(statistics.counts.max()), "n_rows": statistics.n_rows,
                "e_step_seconds": e_step_seconds, "m_step_seconds": m_step_seconds,
                "iteration_seconds": time.perf_counter() - started + m_step_seconds,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(model.mu.device)
                    if model.mu.device.type == "cuda" else 0,
                "pruned_directions": int(update_summary.get("n_shared_b_pruned_directions", 0)),
            }
            history.append(record)
            log(f"[em] iteration={epoch} train_nll={statistics.nll:.8g} val_nll={val_nll} "
                f"b={record['b']:.6g} ranks={record['rank_min']}..{record['rank_max']} "
                f"changed={rank_changes} dead={record['dead_components']} seconds={record['iteration_seconds']:.2f}")
            if _wandb_active():
                import wandb
                wandb.log({f"em/{key}": value for key, value in record.items() if value is not None})

        if epoch >= epochs:
            stop_reason = "iteration_limit"
        elif stable_steps >= 3:
            stop_reason = "converged"
        elif early_stop_patience is not None and without_improvement >= early_stop_patience:
            stop_reason = "patience"
        else:
            stop_reason = None

        if directory is not None:
            _atomic_torch_save({
                "fit_method": "em", "em_config": asdict(cfg), "epoch": epoch,
                "model_config": model_config,
                "model": _cpu_state_dict(model), "best_state": best_state,
                "best_metric": best_metric, "best_epoch": best_epoch,
                "progress_metric": progress_metric, "stable_steps": stable_steps,
                "without_improvement": without_improvement, "history": history,
                "stop_reason": stop_reason,
            }, str(checkpoint_path))
            report = directory / "em_history.json"
            temporary = report.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "history": history, "best_iteration": best_epoch, "stop_reason": stop_reason,
            }, indent=2, allow_nan=False))
            temporary.replace(report)
            if new_record and epoch > 0 and epoch_snapshot_every > 0 and (epoch == 1 or epoch % epoch_snapshot_every == 0):
                snapshot = directory / f"epoch_{epoch:04d}"
                snapshot.mkdir(exist_ok=True)
                save_mfa_hddc(model, str(snapshot / "mfa_model.pt"), extra={"fit_method": "em", "iteration": epoch})
        if stop_reason is not None:
            break
        old_ranks = ranks.clone()
        started = time.perf_counter()
        update_summary = maximization_step(model, statistics, cfg)
        rank_changes = int((model.component_ranks != old_ranks).sum())
        m_step_seconds = time.perf_counter() - started
        del statistics
        epoch += 1

    model.load_state_dict(best_state)
    model._inference_cache = None
    if directory is not None:
        save_mfa_hddc(model, str(directory / "mfa_model.pt"), extra={"fit_method": "em", "iteration": best_epoch})
    log(f"[em] stopped: {stop_reason}; restored best iteration {best_epoch}")
    return history
