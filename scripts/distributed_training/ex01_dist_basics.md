# Exercise 1 — torch.distributed basics (~60 min)

## What you will build

Four small functions that cover every collective operation used in the MFA codebase.
Run `test_ex01.py` when you are done to verify them.

---

## Background

### Processes, ranks, and world size

When you run `torchrun --nproc_per_node=2 my_script.py`, PyTorch launches **two copies** of
`my_script.py` as separate OS processes. Each process gets three environment variables:

| Variable | Meaning |
|----------|---------|
| `RANK` | This process's global index (0 or 1 for 2-process runs) |
| `LOCAL_RANK` | GPU index on this machine (same as RANK when running on one node) |
| `WORLD_SIZE` | Total number of processes (2 here) |

Before any distributed operation, each process must call:

```python
dist.init_process_group(backend="nccl")  # "nccl" for GPU, "gloo" for CPU
```

After that, the processes can communicate. At the end, call:

```python
dist.destroy_process_group()
```

**Every process must call every collective.** If one process skips a `dist.all_reduce`, both
processes hang forever waiting for each other — a "distributed deadlock."

---

### The four collectives

#### 1. `broadcast` — one rank sends, everyone receives

```
Before:   rank 0: [42]    rank 1: [0]
           dist.broadcast(t, src=0)
After:    rank 0: [42]    rank 1: [42]
```

```python
dist.broadcast(tensor, src=0)  # all ranks call this; rank 0's value wins
```

#### 2. `all_reduce(SUM)` — every rank contributes, everyone gets the total

```
Before:   rank 0: [3]     rank 1: [7]
           dist.all_reduce(t, op=dist.ReduceOp.SUM)
After:    rank 0: [10]    rank 1: [10]
```

Used in `train.py` to aggregate the total NLL across ranks:
```python
dist.all_reduce(t, op=dist.ReduceOp.SUM)
```

#### 3. `all_reduce(MAX)` — everyone gets the global maximum

```
Before:   rank 0: [1.0]   rank 1: [5.0]
           dist.all_reduce(t, op=dist.ReduceOp.MAX)
After:    rank 0: [5.0]   rank 1: [5.0]
```

Used in `_distributed_logsumexp` for numerical stability (subtract global max before exp).

#### 4. `all_gather` — collect a tensor from every rank, everyone gets them all

```
Before:   rank 0: [1, 2]   rank 1: [3, 4]
           dist.all_gather(output_list, tensor)
After:    rank 0: [[1,2],[3,4]]   rank 1: [[1,2],[3,4]]
```

Different from `reduce`: nothing is summed — each piece stays intact.

```python
parts = [torch.empty_like(local_tensor) for _ in range(world_size)]
dist.all_gather(parts, local_tensor)
# parts[0] is rank 0's tensor, parts[1] is rank 1's, etc.
```

---

## Your tasks (fill in `ex01_stubs.py`)

**Task 1 — `share_from_rank0`**

On rank 0, create a tensor holding `value`. Broadcast it so all ranks end up with the same tensor.
Hint: only rank 0 needs to set the value before the broadcast; other ranks can start with zeros.

**Task 2 — `distributed_sum`**

Each rank starts with a different `local_value` (a float). Return the sum across all ranks.
Both ranks should return the same result.

**Task 3 — `distributed_max`**

Each rank starts with a different `local_value`. Return the global maximum.
Think about why this is used before `exp` in numerical computations.

**Task 4 — `gather_all`**

Each rank has a `local_tensor` of the same shape. Return a list of all tensors, one per rank,
in rank order. Every rank should see the full list.

---

## Questions to think about while implementing

1. What happens if rank 0 calls `broadcast` but rank 1 calls `all_reduce`? Try to predict
   before you look at what happens.

2. In `train.py:209`, you'll see:
   ```python
   t = torch.tensor([total_nll, float(total_n)], device=device, dtype=torch.float64)
   dist.all_reduce(t, op=dist.ReduceOp.SUM)
   ```
   Why does it reduce both `total_nll` and `total_n` in a single call instead of two?

3. The test for `gather_all` requires every rank to see every piece. When would you want
   `all_gather` vs just `gather` (which only collects on rank 0)?
