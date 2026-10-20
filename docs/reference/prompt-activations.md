# Prompt activation extraction

> **Kind:** Reference · **Status:** Current · **Use when:** Extracting selected
> prompt-token representations or recovering their original text and labels.

`dalg-run-prompt-extraction` reads the JSONL prompt schema produced by
`dalg.data.build_sae_manifold_prompts`. It saves one activation row per selected
prompt token using the existing layer-shard format. V1 selects the last token
and supports native Hugging Face Llama-family models, including Llama-3.1-8B.

## Run

From the repository root, using the project environment:

```bash
PYTHONPATH=src .venv/bin/python -m dalg.cli.run_prompt_extraction \
  --dataset dalg-cache/sae_manifold_prompts \
  --out-dir dalg-cache/sae_manifold_llama31_8b_last \
  --model meta-llama/Llama-3.1-8B \
  --device cuda --dtype bfloat16 \
  --extract-batch-size 8 --shard-size 512
```

The installed `dalg-run-prompt-extraction` command accepts the same arguments.
Add `--preview` to print input lengths and a selected-token example for each
subset without loading model weights or writing extraction outputs. It still
needs the model configuration and tokenizer. Hugging Face downloads go to
`dalg-cache/huggingface/hub` by default; `--cache-dir` overrides that directory.
Gated models require an account with access and an existing Hugging Face token.

`--layers 0 5 19` selects specific zero-based blocks; omission selects every
block. `--revision` pins a model/tokenizer revision. Local Llama checkpoints are
also accepted. `--max-length` rejects overlong inputs rather than truncating;
the default is the model's context limit. This count includes the added BOS.

## Input and representation

Each input JSONL record requires:

- `row_id`: unique integer; IDs need not be contiguous or match file positions.
- `prompt`: nonempty string, preserved exactly, including trailing whitespace.
- `subset`: nonempty string used by downstream stratified splitting.
- `labels`: arbitrary JSON dictionary, including nested values.

Additional fields are preserved in the source snapshot. `tokenization` is
reserved for extractor metadata. Adjacent `metadata.json`, if present, is
copied without changes to `source/dataset_metadata.json`.

The tokenizer encodes the original prompt without automatic special tokens.
The extractor prepends exactly one BOS and appends no EOS or chat template.
The selected token is the final token of that original prompt, including any
punctuation or trailing whitespace. Right padding is masked explicitly.

Each layer tensor contains the residual stream after its complete decoder
block, before the model's final normalization. For Llama-3.1-8B these are blocks
0–31, each with 4096 features. Embeddings and final-normalization outputs are
not saved. Inference uses unmodified model weights, evaluation mode, no
gradients, and no KV cache. Hooks copy only selected positions to CPU.

## Output contract

```text
<root>/
  config.json
  progress.json
  source/
    records.jsonl
    dataset_metadata.json
  layer00/shard_00000.pt
  ...
  layer31/shard_00000.pt
  tokens/shard_00000.pt
  meta/shard_00000.json
```

- Layer tensors: `(rows, 1, d_model)`, in the requested floating dtype.
- Token tensors: `(rows, 1)`, int32 selected vocabulary IDs.
- `config.json`: `format=dalg_prompt_activations_v1`, `window=1`,
  `drop_prefix=0`, selected layers, model/tokenizer identity, input and snapshot
  fingerprints, tokenization policy, shard sizes, and library versions.
- `source/records.jsonl`: each original record once, extended with
  `tokenization.input_ids`, `offset_mapping`, and `selected_positions`.
  Input IDs include BOS and exclude batch padding. The BOS offset is null;
  other offsets are the tokenizer's character spans in the original text.
- Shard metadata follows the existing `start`, `end`, `row_indices`, `rows`
  structure. Each row records `source_row_id`, `subset`, `token_index`,
  `token_id`, `token_piece`, `decoded_token`, and `char_span`.

There are two distinct row identities: `row_indices` are contiguous activation
IDs in saved order, while `source_row_id` refers to the original dataset. Every
layer uses the same activation IDs. Duplicate prompts remain distinct records.

**Stored token slot 0 is not the original token position.** The streaming
reader's `tok_pos` is always 0 for this format. `token_index` identifies the
selected token in the full unpadded, BOS-prefixed model input. Vocabulary pieces
and individually decoded tokens may display differently; token ID and index
are authoritative. Character spans can overlap for byte-level Unicode tokens.

## Recover provenance

```python
from dalg.data.prompt_activations import load_prompt_activation_index
import torch

index = load_prompt_activation_index("dalg-cache/sae_manifold_llama31_8b_last", layer=19)
row = index[0]
print(row["prompt"], row["labels"])
print(row["source_row_id"], row["token_index"], repr(row["decoded_token"]))
shard = torch.load(row["activation_path"], weights_only=True, mmap=True)
activation = shard[row["row_in_shard"], row["token_slot"]]
```

The lookup joins metadata without loading activation tensors and requires a
completed extraction. It returns the full original prompt, label dictionary,
and input IDs alongside each activation's coordinates.

`ActivationBatchDataset`, subset stratification, training, and assignment
readers can consume these shards with `drop_prefix=0`. Training configurations
must not explicitly override that with a window-dataset default such as 32.
The existing window-based context-labeling workflow assumes original token
positions equal stored window positions; use this lookup for prompt contexts
until that workflow has an explicit prompt-data adapter. Future multi-token
selection must also group train/validation splits by source prompt.

## Resume and validation

Rerun the same command to resume. Model and tokenizer revisions are resolved to
an immutable commit; local checkpoint files are fingerprinted. Changes to the
input contents, tokenization, layer selection, precision, shard size, or
library versions reject reuse of the output directory. Batch size and device
can change on resume. Moving an identical source dataset is permitted.

Each shard is committed only after all layer tensors, token IDs, and metadata
are written through temporary files. `progress.json` records the completed
shards. Missing files cause the affected shard to be recomputed. Existing
committed metadata, token IDs, shapes, and dtypes are checked before skipping.
A finished resume does not load model weights. Consumers should use completed
outputs rather than read concurrently with extraction.

The focused offline validation script uses a tiny native Llama checkpoint:

```bash
PYTHONPATH=src .venv/bin/python scripts/temporary/validate_prompt_extraction.py
```

It checks batch/single-prompt equivalence, the final-normalization boundary,
whitespace and label round trips, the existing activation reader, incomplete
shard recovery, incompatible resume settings, and rejection of long inputs.

See [Recreating Pile activations](../workflows/recreate-pile-activations.md) for
the separate fixed-window workflow and [the training pipeline](../workflows/training-pipeline.md)
for downstream model fitting.
