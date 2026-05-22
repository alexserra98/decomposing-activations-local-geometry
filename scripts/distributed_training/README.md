# Distributed Training Exercises

Three exercises to build intuition for what happens in the component-sharded MFA training.
Each takes 1-1.5 hours. Do them in order — each builds on the previous.

## Prerequisites

- Two GPUs available (H100 node via interactive allocation or sbatch)
- Virtual environment activated: `source .venv/bin/activate`

## How exercises work

Each exercise has three files:

| File | What you do with it |
|------|---------------------|
| `exNN_<name>.md` | Read this first. Explains the concept, then describes the task. |
| `exNN_stubs.py` | Fill in the `raise NotImplementedError` blocks. |
| `test_exNN.py` | Run with `torchrun` to check your implementation. |

## Running tests

```bash
# From the repo root
cd /u/dssc/zenocosini/decomposing-activations-local-geometry

# Exercise 1
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/distributed_training/test_ex01.py

# Exercise 2
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/distributed_training/test_ex02.py

# Exercise 3
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/distributed_training/test_ex03.py
```

A passing run looks like:
```
=== Testing your implementation ===
  PASS  share_from_rank0: both ranks get 42.0
  PASS  distributed_sum: 3.0 + 7.0 = 10.0
  ...
All tests passed.
```

## Verifying the tests themselves are correct

If you want to check that the test assertions are correct before you start:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/distributed_training/test_ex01.py --self-test
```

`--self-test` runs the built-in reference implementations instead of your stubs.
All assertions should pass — if they don't, there is a bug in the test file itself.

## Connection to the actual codebase

| Exercise | Corresponds to |
|----------|---------------|
| ex01 | The `dist.all_reduce` and `dist.broadcast` calls scattered through `run_layer.py` and `train.py` |
| ex02 | The old DDP path in `_train_from_shards` (now replaced by component sharding) |
| ex03 | `ComponentShardedMFA.log_prob`, `_distributed_logsumexp`, `sync_replicated_grads` in `mfa.py` |
