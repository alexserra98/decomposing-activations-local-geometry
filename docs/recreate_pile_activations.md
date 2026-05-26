# Recreating Pile Activations on a New Cluster

This guide walks through rebuilding two datasets from scratch:

1. **`pile_gemma2b_100M_windows`** — tokenised Pile windows (HF Arrow format)
2. **`pile_gemma2b_activations/layer05`, `layer17`, `meta`, `tokens`** — Gemma-2B activation shards

---

## Prerequisites

### Python environment

The repo uses [uv](https://github.com/astral-sh/uv). Install it if missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create and install the environment from the repo root:

```bash
uv sync
```

> **Known issue — huggingface_hub version**: `transformers 5.x` requires
> `huggingface-hub>=1.5.0,<2.0`. The `pyproject.toml` already reflects this
> (fixed). If you hit `ImportError: cannot import name 'is_offline_mode'`
> after syncing, run:
> ```bash
> uv pip install "huggingface-hub>=1.5.0,<2.0"
> ```

### HuggingFace access

Both the Gemma-2B model and the Timaeus Pile subsets are gated or rate-limited on HF. Make sure you are authenticated:

```bash
uv run huggingface-cli login
```

Or set the token via environment variable:

```bash
export HF_TOKEN=hf_...
```

### Scratch storage

Choose a scratch root directory. The defaults in the scripts assume:

```
SCRATCH=/orfeo/scratch/dssc/<username>
WINDOWS_DIR=$SCRATCH/pile_gemma2b_100M_windows
ACTIVATIONS_DIR=$SCRATCH/pile_gemma2b_activations
```

Adjust the paths to match your new cluster. The `scripts/slurm/` scripts have a `# Config` block at the top for this.

---

## Step 0: Quick smoke test (10 samples)

Before launching full jobs, verify the pipeline end-to-end on a tiny dataset. This runs on CPU and takes about 2 minutes.

```bash
# Build 10 windows from one Pile subset
uv run dalg-build-pile-windows \
    --out /tmp/dalg_smoke/windows \
    --subsets pile-wikipedia_en \
    --limit 10

# Extract activations for layers 5 and 17
uv run dalg-run-extraction \
    --dataset /tmp/dalg_smoke/windows/merged \
    --out-dir /tmp/dalg_smoke/activations \
    --model google/gemma-2b \
    --layers 5 17 \
    --dtype float32 \
    --device cpu \
    --num-workers 0 \
    --debug --limit 10

# Verify the output is readable
python -c "
from dalg.data.shard_activations import load_meta_index, ActivationBatchDataset
from torch.utils.data import DataLoader
meta = load_meta_index('/tmp/dalg_smoke/activations', layer=5)
print('rows:', len(meta))
ds = ActivationBatchDataset('/tmp/dalg_smoke/activations', layer=5,
      batch_size=32, shuffle_shards=False, shuffle_within_shard=False)
print('tokens:', ds.num_items, 'd_model:', ds.d_model)
loader = DataLoader(ds, batch_size=None, num_workers=0)
xb = next(iter(loader))
print('batch shape:', xb.shape, '  PASS')
"
```

Expected output:
- `layer05/shard_00000.pt`: shape `(N, 256, 2048)`
- `layer17/shard_00000.pt`: same
- batch shape `(32, 2048)` or smaller for the last partial batch

---

## Step 1: Build the 100M-token windows dataset

This step streams the 17 Timaeus Pile subsets from HuggingFace, tokenises each document with the Gemma-2B tokenizer, samples one 256-token window per document, and writes an HF Arrow dataset to disk.

**Estimated wall time:** 3–6 hours (network-bound; no GPU needed)  
**Estimated disk space:** ~25 GB

### SLURM (recommended)

Edit `scripts/slurm/sbatch_extract_activations.sh` to update `DATASET` and `OUT_DIR` to your cluster paths, then:

```bash
sbatch scripts/slurm/sbatch_extract_activations.sh
```

Or submit the build step separately:

```bash
sbatch --partition=<CPU_PARTITION> --cpus-per-task=8 --mem=32G --time=8:00:00 \
  --wrap="uv run dalg-build-pile-windows --out $WINDOWS_DIR"
```

### Local / interactive

```bash
uv run dalg-build-pile-windows --out /path/to/pile_gemma2b_100M_windows
```

The command is **resume-safe**: per-subset shards are written atomically and skipped on re-run. The final `merged/` dataset is written only after all subsets succeed.

### Output layout

```
pile_gemma2b_100M_windows/
  shards/
    pile-wikipedia_en/   # HF Arrow shard (one per subset)
    pile-arxiv/
    ...
  merged/                # concatenated HF dataset — this is the input to extraction
    dataset_info.json
    state.json
    data-00000-of-NNNNN.arrow
    ...
```

HF dataset columns:

| Column         | Type          | Description                                       |
|----------------|---------------|---------------------------------------------------|
| `text`         | `str`         | Raw document text (capped at window end)          |
| `subset`       | `str`         | e.g. `"pile-wikipedia_en"`                        |
| `token_ids`    | `list[int]`   | 256 token IDs (no special tokens)                 |
| `window_start` | `int`         | Start position in the full document               |
| `window_end`   | `int`         | End position in the full document                 |
| `doc_len`      | `int`         | Total token length of the source document         |

---

## Step 2: Extract activations

This step runs the full Gemma-2B model on each window and saves the residual-stream activations at layers 5 and 17.

**Estimated wall time:** ~4–6 hours on a single A100 for 100M tokens  
**Estimated disk space:** ~200 GB total (float16, 2 layers, 256 window, 2048 d_model)

### SLURM (recommended)

Edit `scripts/slurm/sbatch_extract_activations.sh` and update:

```bash
DATASET="/path/to/pile_gemma2b_100M_windows/merged"
OUT_DIR="/path/to/pile_gemma2b_activations"
```

Then submit:

```bash
sbatch scripts/slurm/sbatch_extract_activations.sh
```

For a debug/smoke run:

```bash
sbatch --export=ALL,DEBUG=1 scripts/slurm/sbatch_extract_activations.sh
```

This sets `OUT_DIR` to `${OUT_DIR}_debug` and limits to 64 rows.

### Local / interactive

```bash
uv run dalg-run-extraction \
    --dataset /path/to/pile_gemma2b_100M_windows/merged \
    --out-dir /path/to/pile_gemma2b_activations \
    --model google/gemma-2b \
    --layers 5 17 \
    --dtype float16 \
    --extract-batch-size 16 \
    --shard-size 512 \
    --num-workers 4 \
    --device cuda
```

The command is **resume-safe**: shards that already exist on disk are skipped. `progress.json` tracks completion.

### Output layout

```
pile_gemma2b_activations/
  config.json          # model, layers, window, d_model, dtype, drop_prefix, ...
  progress.json        # per-shard timing, used for resume
  layer05/
    shard_00000.pt     # float16 tensor, shape (512, 256, 2048) per full shard
    shard_00001.pt
    ...
  layer17/
    shard_00000.pt
    ...
  tokens/
    shard_00000.pt     # int32 token IDs, shape (512, 256)
    ...
  meta/
    shard_00000.json   # per-row metadata: row_indices, subset, window_start/end, doc_len
    ...
```

Key `config.json` fields:

| Field        | Typical value | Meaning                                              |
|--------------|---------------|------------------------------------------------------|
| `model`      | `google/gemma-2b` | TransformerLens model name                       |
| `layers`     | `[5, 17]`     | Extracted layers                                     |
| `window`     | `256`         | Tokens per row (excluding prepended BOS)             |
| `d_model`    | `2048`        | Activation dimension                                 |
| `dtype`      | `float16`     | Tensor dtype on disk                                 |
| `drop_prefix`| `32`          | Recommended tokens to drop at the start of each row  |
| `prepend_bos`| `true`        | BOS token is prepended before each window during extraction (then stripped from saved activations) |

---

## Step 3: Verify the final dataset

```bash
python - <<'EOF'
from dalg.data.shard_activations import load_meta_index, stratified_split, per_subset_counts

for layer in [5, 17]:
    meta = load_meta_index("/path/to/pile_gemma2b_activations", layer=layer)
    train, val = stratified_split(meta, val_frac=0.05, seed=42)
    counts = per_subset_counts(meta, list(range(len(meta))))
    print(f"\nlayer {layer:02d}: {len(meta):,} rows  train={len(train):,} val={len(val):,}")
    for subset, n in counts.items():
        print(f"  {subset}: {n:,}")
EOF
```

Expected: ~390k rows total across 17 subsets, roughly equal across subsets.

---

## Key parameters to adapt for a new cluster

| Parameter          | Where to change                                  | Notes                                                  |
|--------------------|--------------------------------------------------|--------------------------------------------------------|
| Scratch root path  | `scripts/slurm/sbatch_extract_activations.sh`   | Change `DATASET`, `OUT_DIR`                            |
| SLURM partition    | All `scripts/slurm/*.sh`                        | Replace `#SBATCH --partition=DGX --account=LADE`       |
| GPU type           | `scripts/slurm/sbatch_extract_activations.sh`   | Replace `#SBATCH --gres=gpu:A100:1`                    |
| SLURM log path     | All `scripts/slurm/*.sh`                        | Replace `/u/dssc/zenocosini/...` in `--output`         |
| Batch size         | `scripts/slurm/sbatch_extract_activations.sh`   | Reduce `BATCH` if OOM; 16 fits A100 80GB in float16    |
| Number of layers   | `scripts/slurm/sbatch_extract_activations.sh`   | `LAYERS="5 17"` — add more if needed                   |

---

## Troubleshooting

**`ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`**

```bash
uv pip install "huggingface-hub>=1.5.0,<2.0"
```

The `pyproject.toml` already has the correct constraint; this just ensures the venv is in sync.

**`ModuleNotFoundError: No module named 'datasets'`**

You are likely running `python` from the system environment, not the venv. Use `uv run` or activate the venv:

```bash
source .venv/bin/activate
```

**Dataset streaming is slow or times out**

The Timaeus Pile subsets stream from HuggingFace. On a cluster with restricted outbound access, set:

```bash
export HF_DATASETS_OFFLINE=1
```

and pre-cache the dataset locally first, or mirror it.

**OOM during extraction**

Reduce `--extract-batch-size`. At float16, batch size 8 needs ~15 GB GPU memory for Gemma-2B.

**Shard count mismatch when resuming**

The extraction is resume-safe. Just re-run the same command — it will skip completed shards and fill in any missing ones. Do **not** delete `progress.json` unless you want to force a full re-run.
