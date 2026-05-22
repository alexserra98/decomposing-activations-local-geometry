# Exercise 3 — Component Sharding in Detail (~90 min)

## What you will build

Three functions that together implement the core of `ComponentShardedMFA`:
1. `distributed_logsumexp` — the numerically stable distributed logsumexp with correct gradients
2. `sharded_log_prob` — mixture log-probability assembled across ranks
3. `sync_shared_param_grad` — gradient sync for parameters replicated across ranks

---

## Part A — The distributed logsumexp

### Why we need it

In `ComponentShardedMFA.log_prob`, both ranks need to compute:

```
log p(x) = log Σ_k π_k · p(x | k)
         = logsumexp_k ( log π_k + log p(x | k) )
```

But each rank only has half of the k indices. To assemble the global sum, ranks need to
communicate. The naive approach would be to all_gather all `log p(x | k)` tensors and
then compute logsumexp locally — but that defeats the purpose of splitting (you'd need
to materialise K values per sample on every GPU).

Instead, we exploit the structure of logsumexp:

```
logsumexp(x_0, x_1, ..., x_{K-1}) = M + log Σ_k exp(x_k - M)    where M = max(x_k)
```

We can compute M with `all_reduce(MAX)` and the sum with `all_reduce(SUM)` — each rank only
sends one number per sample, regardless of K.

### The gradient problem

Suppose you try the naive implementation:

```python
# BROKEN — do not use
local_sum = shifted.exp().sum(dim=dim)
global_sum = local_sum.clone()
dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)   # global_sum is in the graph!
return global_max + global_sum.log()
```

The problem: `global_sum` depends on `local_sum` from **every rank's** computation graph.
When you call `backward()`, PyTorch tries to differentiate through `all_reduce` — but each
rank only sees its own slice of the graph. Rank 0 would try to propagate gradients through
rank 1's `local_sum`, which doesn't exist in rank 0's process. The gradient is wrong.

Concretely: if rank 0's `local_sum = 3.0` and rank 1's `local_sum = 7.0`, then
`global_sum = 10.0`. The correct gradient of `log(global_sum)` w.r.t. rank 0's `local_sum`
is `1/10 = 0.1`. But PyTorch on rank 0 sees `d log(global_sum) / d local_sum_rank0 = 1/3`
(because it only knows about rank 0's contribution). This is wrong.

### The fix: detach + correction term

```python
global_sum_detached = local_sum.detach().clone()
dist.all_reduce(global_sum_detached, op=dist.ReduceOp.SUM)
# global_sum_detached has the correct value but is detached from the graph

# The correction term is 0 in the forward pass but carries the right gradient:
correction = (local_sum - local_sum.detach()) / global_sum_detached

return global_max + global_sum_detached.log() + correction
```

**Forward**: `global_max + log(global_sum) + 0` = correct logsumexp value ✓

**Backward**: only the `correction` term carries gradient. Its gradient w.r.t. `local_values[i]`:
```
d correction / d local_values[i]
  = d/d local_values[i] [local_sum / global_sum_detached]
  = exp(local_values[i] - global_max) / global_sum_detached
  = exp(local_values[i]) / Σ_j exp(x_j)
  = softmax(x_i)
```
This is exactly the correct gradient for logsumexp. ✓

---

## Part B — The replicated parameter problem

In MFA, `psi_rho` (shape `(D,)`) controls the diagonal noise variance and is **shared** across
all components. With component sharding, every rank holds a copy of `psi_rho`.

After backward, each rank's `psi_rho.grad` only reflects the likelihood contribution from
its K_local components:

```
grad on rank 0 = d NLL_local_0 / d psi_rho   (from components 0–3999 only)
grad on rank 1 = d NLL_local_1 / d psi_rho   (from components 4000–7999 only)
```

The correct gradient is the sum: `d NLL_total / d psi_rho = grad_0 + grad_1`.

Without syncing, ranks 0 and 1 apply different gradient updates to `psi_rho` and diverge.
The fix is a single `all_reduce(SUM)` on `psi_rho.grad` after `loss.backward()`.

---

## Your tasks (fill in `ex03_stubs.py`)

**Task 1 — `distributed_logsumexp`**

Implement the detach+correction trick described above.
- Compute the global max with `all_reduce(MAX)` (detached).
- Subtract, exp, sum locally.
- Compute global sum with `all_reduce(SUM)` (detached).
- Return `global_max + log(global_sum) + correction`.

When `dist` is not initialized, fall back to `torch.logsumexp`.

**Task 2 — `sharded_log_prob`**

Given each rank's component means and log-unnormalized weights, compute the global
mixture log-probability for each sample in x.

Assume identity covariance: `log p(x | k) = -D/2 log(2π) - ½ ||x - mu_k||²`.

Steps:
1. Compute `ll`: (B, K_local) log-likelihoods for this rank's components.
2. Compute `log_num = distributed_logsumexp(ll + local_log_pi[None, :], dim=1)`.
3. Compute `log_den = distributed_logsumexp(local_log_pi, dim=0)`.
4. Return `log_num - log_den`.

**Task 3 — `sync_shared_param_grad`**

All-reduce the gradient of a replicated parameter using SUM.
If `param.grad` is None, do nothing.

---

## Connection to the actual code

After finishing, open `src/dalg/models/mfa.py` and find:
- `_distributed_logsumexp` (line ~377): your Task 1, but handles arbitrary tensor shapes
- `ComponentShardedMFA.log_prob` (line ~459): your Task 2, but using the full MFA `_core`
- `ComponentShardedMFA.sync_replicated_grads` (line ~468): your Task 3, with an early-return for per-component Psi

And in `src/dalg/models/train.py` (line ~181):
```python
sync_replicated_grads = getattr(raw_model, "sync_replicated_grads", None)
if callable(sync_replicated_grads):
    sync_replicated_grads()
```
This is where Task 3 gets called during every training step.
