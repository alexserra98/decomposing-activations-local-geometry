"""Extract last-token original Llama block outputs."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from dalg.data.prompt_activations import FORMAT, load_prompt_records


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def select_token_positions(input_ids: list[int]) -> list[int]:
    """Select the last token in a BOS-prefixed, unpadded model input."""
    if len(input_ids) < 2:
        raise ValueError("A prompt must contain at least one token besides the added BOS")
    return [len(input_ids) - 1]


def tokenize_prompt_records(records, tokenizer, *, max_length: int):
    """Preserve source records and attach the exact model inputs and token offsets."""
    if not tokenizer.is_fast or tokenizer.bos_token_id is None:
        raise ValueError("Extraction requires a fast tokenizer with a BOS token")
    tokenized = []
    for record in records:
        encoded = tokenizer(
            record["prompt"], add_special_tokens=False, truncation=False,
            return_offsets_mapping=True, return_attention_mask=False,
        )
        ids = [tokenizer.bos_token_id, *encoded["input_ids"]]
        if len(ids) > max_length:
            raise ValueError(
                f"row_id={record['row_id']} has {len(ids)} tokens including BOS; "
                f"max_length={max_length}. Prompts are never truncated."
            )
        tokenized.append({
            **record,
            "tokenization": {
                "input_ids": ids,
                "offset_mapping": [None, *[list(span) for span in encoded["offset_mapping"]]],
                "selected_positions": select_token_positions(ids),
            },
        })
    return tokenized


@torch.inference_mode()
def collect_prompt_activations(model, records, *, layers, pad_token_id, device, dtype):
    """Return CPU tensors (prompts, d_model), selecting the last token on the model device.

    Each record carries its unpadded input_ids and selected_positions. Hooks are
    removed even if a forward pass fails. Right padding leaves real token
    positions unchanged; its attention mask is passed explicitly to the model.
    """
    if model.config.model_type != "llama":
        raise ValueError("Only the native Hugging Face Llama model is supported")
    model.eval()
    lengths = [len(record["tokenization"]["input_ids"]) for record in records]
    ids = torch.full((len(records), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    batch_indices, positions = [], []
    for i, record in enumerate(records):
        tokens = record["tokenization"]
        if tokens["selected_positions"] != select_token_positions(tokens["input_ids"]):
            raise ValueError("Selected positions must contain only the last prompt token")
        ids[i, :lengths[i]] = torch.tensor(tokens["input_ids"], device=device)
        mask[i, :lengths[i]] = 1
        positions.extend(tokens["selected_positions"])
        batch_indices.append(i)
    batch_indices = torch.tensor(batch_indices, device=device)
    positions = torch.tensor(positions, device=device)
    captured, handles = {}, []
    try:
        for layer in layers:
            def capture(module, inputs, output, layer=layer):
                # Native decoder outputs are tensors; older Transformers used a tuple.
                hidden = output if isinstance(output, torch.Tensor) else output[0]
                captured[layer] = hidden[batch_indices, positions].to(device="cpu", dtype=dtype)
            handles.append(model.layers[layer].register_forward_hook(capture))
        model(
            input_ids=ids, attention_mask=mask, use_cache=False,
            output_hidden_states=False, output_attentions=False, return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    for layer in layers:
        if layer not in captured or not torch.isfinite(captured[layer]).all():
            raise ValueError(f"Missing or nonfinite activations at layer {layer}")
    return captured


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _local_checkpoint_hash(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob("*")):
        if file.is_file() and file.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}:
            digest.update(str(file.relative_to(path)).encode())
            digest.update(_sha256(file).encode())
    return digest.hexdigest()


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_tensor(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    temporary.replace(path)


def _target_metadata(records, tokenizer):
    rows = []
    for record in records:
        tokenization = record["tokenization"]
        position, = tokenization["selected_positions"]
        token_id = tokenization["input_ids"][position]
        rows.append({
            "source_row_id": record["row_id"], "subset": record["subset"],
            "token_index": position, "token_id": token_id,
            "token_piece": tokenizer.convert_ids_to_tokens(token_id),
            "decoded_token": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            "char_span": tokenization["offset_mapping"][position],
        })
    return rows


def _shard_complete(root, shard, layers, expected_meta, d_model, dtype):
    """Check a committed shard's complete set of files before skipping it."""
    filename = f"shard_{shard:05d}"
    meta_path = root / "meta" / f"{filename}.json"
    tokens_path = root / "tokens" / f"{filename}.pt"
    layer_paths = [root / f"layer{layer:02d}" / f"{filename}.pt" for layer in layers]
    if not all(path.exists() for path in [meta_path, tokens_path, *layer_paths]):
        return False
    if json.loads(meta_path.read_text()) != expected_meta:
        raise ValueError(f"Shard {shard} metadata differs from its source snapshot")
    n_rows = len(expected_meta["rows"])
    expected_ids = torch.tensor([[row["token_id"]] for row in expected_meta["rows"]], dtype=torch.int32)
    tokens = torch.load(tokens_path, map_location="cpu", weights_only=True, mmap=True)
    if tokens.dtype != torch.int32 or not torch.equal(tokens, expected_ids):
        raise ValueError(f"Shard {shard} token IDs differ from its source snapshot")
    for path in layer_paths:
        tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if tensor.shape != (n_rows, 1, d_model) or tensor.dtype != dtype:
            raise ValueError(f"Unexpected activation shape or dtype in {path}")
    return True


def run_prompt_extraction(
    *, dataset, out_dir, model_name="meta-llama/Llama-3.1-8B", revision=None,
    layers=None, dtype="bfloat16", device="cuda",
    batch_size=8, shard_size=512, max_length=None,
    cache_dir="dalg-cache/huggingface/hub", preview=False,
):
    """Write resumable prompt representation shards and a source snapshot.

    Model/tokenizer loading supports a Hugging Face repository or a local Llama
    checkpoint. A finished run can be resumed without loading the model weights.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer, __version__ as transformers_version

    if batch_size <= 0 or shard_size <= 0:
        raise ValueError("batch_size and shard_size must be positive")
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    source_path = Path(dataset)
    if source_path.is_dir():
        source_path = source_path / "prompts.jsonl"
    records = load_prompt_records(source_path)
    metadata_path = source_path.parent / "metadata.json"
    source_metadata = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else "{}\n"
    json.loads(source_metadata)
    cache_dir = str(Path(cache_dir).resolve())
    local_model = Path(model_name).is_dir()
    model_name = str(Path(model_name).resolve()) if local_model else model_name
    model_config = AutoConfig.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    if model_config.model_type != "llama":
        raise ValueError(f"Unsupported model_type={model_config.model_type!r}; expected 'llama'")
    resolved_revision = getattr(model_config, "_commit_hash", None)
    if not local_model and not resolved_revision:
        raise ValueError("Could not resolve the model's immutable Hugging Face revision")
    load_revision = resolved_revision if not local_model else revision
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=load_revision, cache_dir=cache_dir, use_fast=True,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define PAD or EOS for masked batch padding")
    context_limit = model_config.max_position_embeddings
    max_length = context_limit if max_length is None else max_length
    if not 1 < max_length <= context_limit:
        raise ValueError(f"max_length must be in [2, {context_limit}]")
    layers = list(range(model_config.num_hidden_layers)) if layers is None else sorted(set(layers))
    if not layers or layers[0] < 0 or layers[-1] >= model_config.num_hidden_layers:
        raise ValueError("Requested layers are outside the model's decoder blocks")
    records = tokenize_prompt_records(records, tokenizer, max_length=max_length)
    targets = _target_metadata(records, tokenizer)
    # copy of the original records for reproducibility
    snapshots = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)
    d_model = model_config.hidden_size
    num_shards = math.ceil(len(records) / shard_size)
    config = {
        "format": FORMAT,
        "model": model_name, "model_revision": resolved_revision,
        "local_checkpoint_sha256": _local_checkpoint_hash(model_name) if local_model else None,
        "tokenizer_revision": resolved_revision,
        "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "mode": "residual", "representation": "decoder_block_output_before_final_norm",
        "layers": layers, "d_model": d_model, "window": 1, "drop_prefix": 0, "dtype": dtype,
        "token_selection": "last", "prepend_bos": True, "append_eos": False,
        "chat_template": None, "padding_side": "right", "pad_token_id": pad_token_id,
        "bos_token_id": tokenizer.bos_token_id, "max_length": max_length, "truncation": False,
        "dataset": str(source_path.resolve()), "dataset_sha256": _sha256(source_path),
        "source_records": "source/records.jsonl", "source_metadata": "source/dataset_metadata.json",
        "source_records_sha256": hashlib.sha256(snapshots.encode()).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(source_metadata.encode()).hexdigest(),
        "num_rows": len(records), "num_source_rows": len(records),
        "shard_size": shard_size, "num_shards": num_shards,
        "torch_version": torch.__version__, "transformers_version": transformers_version,
    }
    print(f"{len(records):,} prompts; {len(layers)} layers; D={d_model}; last token; {dtype}", flush=True)
    print(f"Input lengths including BOS: {min(len(r['tokenization']['input_ids']) for r in records)}"
          f"–{max(len(r['tokenization']['input_ids']) for r in records)}", flush=True)
    if preview:
        seen = set()
        for record, target in zip(records, targets):
            if record["subset"] not in seen:
                print(json.dumps({"prompt": record["prompt"], "labels": record["labels"], **target}, ensure_ascii=False))
                seen.add(record["subset"])
        return config

    root = Path(out_dir)
    config_path = root / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        changed = [key for key in config if key != "dataset" and previous.get(key) != config[key]]
        if changed:
            raise ValueError(f"Cannot resume with changed settings: {', '.join(changed)}. Use a new output directory.")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Output directory has files but no extraction config: {root}")
        _write_text(config_path, _json_text(config))
    for relative, text in [(config["source_records"], snapshots), (config["source_metadata"], source_metadata)]:
        path = root / relative
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Source snapshot has changed: {path}")
        if not path.exists():
            _write_text(path, text)
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"completed": [], "timings": []}
    shard_metadata, completed, todo = {}, [], []
    for shard in range(num_shards):
        start, end = shard * shard_size, min((shard + 1) * shard_size, len(records))
        meta = {"start": start, "end": end, "row_indices": list(range(start, end)), "rows": targets[start:end]}
        shard_metadata[shard] = meta
        if shard in progress["completed"] and _shard_complete(root, shard, layers, meta, d_model, DTYPES[dtype]):
            completed.append(shard)
        else:
            todo.append(shard)
    progress["completed"] = completed
    progress["complete"] = not todo
    _write_text(progress_path, _json_text(progress))
    if not todo:
        print(f"All {num_shards} shards are complete; no model weights loaded.", flush=True)
        return config
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on a GPU allocation or explicitly choose --device cpu.")
    print(f"Loading {model_name} on {device}; {len(todo)} shards to extract...", flush=True)
    model = AutoModel.from_pretrained(
        model_name, revision=load_revision, cache_dir=cache_dir, dtype=DTYPES[dtype],
    ).to(device)
    model.eval()
    # A batch groups prompts for one model forward pass.
    # A shard groups the resulting activations into a file on disk.
    for shard in todo:
        started = time.monotonic()
        meta = shard_metadata[shard]
        start, end = meta["start"], meta["end"]
        buffers = {layer: torch.empty((end - start, 1, d_model), dtype=DTYPES[dtype]) for layer in layers}
        for batch_start in range(start, end, batch_size):
            batch_end = min(batch_start + batch_size, end)
            captured = collect_prompt_activations(
                model, records[batch_start:batch_end], layers=layers, pad_token_id=pad_token_id,
                device=device, dtype=DTYPES[dtype],
            )
            for layer in layers:
                buffers[layer][batch_start-start:batch_end-start, 0] = captured[layer]
        filename = f"shard_{shard:05d}"
        for layer in layers:
            _write_tensor(root / f"layer{layer:02d}" / f"{filename}.pt", buffers[layer])
        tokens = torch.tensor([[row["token_id"]] for row in meta["rows"]], dtype=torch.int32)
        _write_tensor(root / "tokens" / f"{filename}.pt", tokens)
        _write_text(root / "meta" / f"{filename}.json", _json_text(meta))
        progress["completed"] = sorted([*progress["completed"], shard])
        progress["complete"] = len(progress["completed"]) == num_shards
        progress["timings"].append({"shard": shard, "rows": end-start, "batch_size": batch_size,
                                    "seconds": time.monotonic()-started})
        _write_text(progress_path, _json_text(progress))
        print(f"Saved shard {shard+1}/{num_shards}: {end-start} rows ({time.monotonic()-started:.1f}s)", flush=True)
    print(f"Complete: {root}", flush=True)
    return config
