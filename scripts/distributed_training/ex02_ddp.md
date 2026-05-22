# Exercise 2 — Data-Parallel DDP vs Model Parallelism (~60 min)

## What you will build

Three functions that implement data-parallel DDP training and check that it works correctly.
Then you'll understand why DDP wasn't viable for your MFA and why component sharding was needed.

---

## Background

### Data-Parallel DDP: every rank holds the full model

DDP (DistributedDataParallel) is the standard distributed training strategy:

```
          Rank 0                       Rank 1
      ┌──────────────┐            ┌──────────────┐
      │  Full model  │            │  Full model  │  ← identical copies
      │  (same W)    │            │  (same W)    │
      └──────┬───────┘            └──────┬───────┘
             │ forward on batch_0        │ forward on batch_1
             ↓                           ↓
         loss_0                      loss_1
             │ backward                  │ backward
             ↓                           ↓
          grad_0       all_reduce(SUM)  grad_1
             └───────────────┬───────────┘
                             ↓
                    grad_0 + grad_1  (on both ranks)
                             ↓
                    optimizer.step()
```

The key property: after each step, **both ranks have the same parameters** because they
averaged the gradients. This is guaranteed by `DistributedDataParallel`.

You wrap your model once:

```python
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

PyTorch automatically installs hooks that all-reduce the gradients after `loss.backward()`.
You don't write any `dist.all_reduce` calls yourself.

---

### Why DDP doesn't work for your MFA

Let's compute the memory for one MFA model with K=8000, D=2048, q=160:

| Parameter | Shape | Float32 bytes |
|-----------|-------|--------------|
| `mu` | (8000, 2048) | 65.5 MB |
| `dir_raw` | (8000, 2048, 160) | **10,485 MB = ~10.2 GB** |
| `scale_rho` | (8000, 160) | 5.1 MB |
| `pi_logits` | (8000,) | 0.03 MB |
| `psi_rho` | (2048,) | 0.008 MB |
| **Total** | | **~10.3 GB** |

With DDP and 2 ranks, each GPU needs:
- Model parameters: 10.3 GB
- Adam optimizer state (2 momentum buffers): 20.6 GB
- Activation tensors for the forward pass: varies with batch size
- **Total per GPU: 30+ GB** — tight even on an 80 GB H100

Increasing rank from q=10 to q=160 made `dir_raw` grow from 656 MB to **10.5 GB**.
That's the OOM trigger.

---

### Component sharding: split the model across ranks

Instead of each rank holding all K=8000 components, each rank holds K/2=4000:

```
          Rank 0                       Rank 1
      ┌──────────────────┐       ┌──────────────────┐
      │ Components 0–3999│       │ Components 4000–7999│
      │ dir_raw: 5.1 GB  │       │ dir_raw: 5.1 GB  │
      └──────────────────┘       └──────────────────┘
             ↓ same batch x             ↓ same batch x
         ll[:, 0:4000]             ll[:, 4000:8000]
             └──────── distributed logsumexp ─────────┘
                                ↓
                          log p(x)  (same on both ranks)
```

Both ranks see the **same data**, but each computes likelihoods for only its components.
A distributed logsumexp assembles the global `log p(x)`. This is Exercise 3.

---

## Your tasks (fill in `ex02_stubs.py`)

**Task 1 — `make_loader`**

Create a DataLoader for this rank's slice of a synthetic dataset.
The dataset has 100 samples. Use `torch.utils.data.Subset` to partition them.
Rank 0 gets indices [0, 50), rank 1 gets [50, 100).

The dataset: x is a 4-dimensional vector, y is 0 or 1.
Use any simple dataset — e.g., `TensorDataset(torch.randn(100, 4), torch.randint(0, 2, (100,)))`.
Both ranks should use the same underlying dataset but different subsets.

**Task 2 — `train_one_step_ddp`**

Wrap `model` in `DistributedDataParallel`, take one batch from `loader`, compute the loss
(use `F.cross_entropy`), run `loss.backward()`, and take one optimizer step.
Return the loss value as a float.

The model is a simple 2-layer MLP: `Linear(4, 16) → ReLU → Linear(16, 2)`.
The test creates it and passes it to you — you just wrap it in DDP and train.

**Task 3 — `params_are_equal_across_ranks`**

After a DDP step, verify that both ranks have the same weights.
Hint: all_gather the first parameter's data and compare rank 0 vs rank 1.
Return `True` if they match, `False` otherwise.

---

## Questions to think about

1. In `train_one_step_ddp`, if you call `optimizer.step()` before `loss.backward()`,
   what happens? Why does the order matter?

2. DDP requires that all ranks execute the same sequence of calls. What would happen
   if rank 0 has 50 samples and rank 1 has 51 samples — so rank 1 runs one extra batch?

3. In the MFA component-shard mode (`run_layer.py:495`), you see:
   ```python
   steps_per_epoch = len(train_ds)
   if use_ddp:
       steps_t = torch.tensor([steps_per_epoch], ...)
       dist.all_reduce(steps_t, op=dist.ReduceOp.MIN)
       steps_per_epoch = int(steps_t.item())
   ```
   This is the code's answer to question 2. What would happen without it?
