"""
Exercise 3 stubs — fill in the three functions below.

Run the tests with:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 test_ex03.py
"""
import math
import torch
import torch.nn as nn
import torch.distributed as dist


def distributed_logsumexp(local_values: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute logsumexp over a tensor dimension that is sharded across distributed ranks.

    Every rank calls this with its local slice along `dim`. The result is the true
    global logsumexp (as if all values were concatenated along `dim` first).

    Gradients must flow correctly into `local_values`:
        d result / d local_values[i] = softmax(all_values)[i]

    If dist is not initialized, fall back to torch.logsumexp (single-process case).

    Args:
        local_values: This rank's slice of the values to reduce.
        dim:          The dimension to reduce over (the sharded dimension).

    Returns:
        Tensor with `dim` removed, containing the global logsumexp.

    Key steps:
        1. Compute local_max = local_values.max(dim=dim).values.detach()
        2. global_max = all_reduce(local_max.clone(), MAX)
        3. shifted = local_values - global_max.unsqueeze(dim)
        4. local_sum = shifted.exp().sum(dim=dim)
        5. global_sum = all_reduce(local_sum.detach().clone(), SUM)
        6. correction = (local_sum - local_sum.detach()) / global_sum.clamp_min(tiny)
        7. return global_max + global_sum.log() + correction
    """
    raise NotImplementedError


def sharded_log_prob(
    x: torch.Tensor,
    local_mu: torch.Tensor,
    local_log_pi: torch.Tensor,
) -> torch.Tensor:
    """Compute the mixture log-probability for samples in x, components sharded across ranks.

    Assumes identity covariance: log p(x | k) = -D/2 log(2π) - ½ ||x - mu_k||²

    Args:
        x:            (B, D) batch of activation vectors.
        local_mu:     (K_local, D) component means owned by this rank.
        local_log_pi: (K_local,) raw (unnormalized) log mixture weights for this rank's
                      components. These are logits, NOT log-probabilities.

    Returns:
        (B,) tensor: log p(x) = log Σ_k softmax(log_pi)_k · p(x | k), assembled globally.

    Steps:
        1. Compute ll: (B, K_local) where ll[b, k] = log p(x[b] | mu_k, I).
           ll[b, k] = -D/2 * log(2π) - 0.5 * ||x[b] - local_mu[k]||²
        2. log_num = distributed_logsumexp(ll + local_log_pi[None, :], dim=1)
        3. log_den = distributed_logsumexp(local_log_pi, dim=0)
        4. return log_num - log_den
    """
    raise NotImplementedError


def sync_shared_param_grad(param: nn.Parameter) -> None:
    """All-reduce the gradient of a parameter replicated across component-sharded ranks.

    Each rank computed gradients from its own K_local components, but param is shared
    (same value on every rank). Summing the partial gradients recovers the full gradient.

    Does nothing if:
    - param.grad is None
    - torch.distributed is not initialized

    Args:
        param: A parameter whose .grad should be summed across all ranks.
    """
    raise NotImplementedError
