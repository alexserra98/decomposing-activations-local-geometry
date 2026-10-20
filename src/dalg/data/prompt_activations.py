"""Prompt records and provenance lookup for selected-token activation shards."""

from __future__ import annotations

import json
from pathlib import Path

from dalg.data.shard_activations import load_meta_index


FORMAT = "dalg_prompt_activations_v1"


def load_prompt_records(path: str | Path) -> list[dict]:
    """Read JSONL records without normalizing text or changing label dictionaries.

    Required fields are a unique integer row_id, a nonempty prompt string, a
    subset string, and a labels dictionary. Extra source fields are preserved.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "prompts.jsonl"
    records, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            row_id = record.get("row_id")
            if type(row_id) is not int or row_id in seen:
                raise ValueError(f"{path}:{line_number}: row_id must be a unique integer")
            if not isinstance(record.get("prompt"), str) or not record["prompt"]:
                raise ValueError(f"row_id={row_id}: prompt must be a nonempty string")
            if not isinstance(record.get("subset"), str) or not record["subset"]:
                raise ValueError(f"row_id={row_id}: subset must be a nonempty string")
            if not isinstance(record.get("labels"), dict):
                raise ValueError(f"row_id={row_id}: labels must be a dictionary")
            if "tokenization" in record:
                raise ValueError("tokenization is reserved for the extraction snapshot")
            seen.add(row_id)
            records.append(record)
    if not records:
        raise ValueError(f"No prompt records in {path}")
    return records


def load_prompt_activation_index(activation_dir: str | Path, layer: int) -> list[dict]:
    """Join activation coordinates to exact source prompts, tokens, and labels.

    Requires a completed extraction. No activation tensors are loaded. Each
    result includes activation_path, row_in_shard, and token_slot=0 for tensor
    indexing, plus token_index in the full unpadded model input (including BOS).
    """
    root = Path(activation_dir)
    config = json.loads((root / "config.json").read_text())
    if config.get("format") != FORMAT:
        raise ValueError(f"{root} is not a prompt activation dataset")
    if layer not in config["layers"]:
        raise ValueError(f"Layer {layer} was not extracted")
    progress = json.loads((root / "progress.json").read_text())
    if not progress["complete"]:
        raise ValueError("Extraction is incomplete; resume it before loading the index")
    with (root / config["source_records"]).open(encoding="utf-8") as handle:
        sources = {record["row_id"]: record for record in map(json.loads, handle)}
    index, shard_metadata = [], {}
    for coordinate in load_meta_index(root, layer):
        shard = coordinate["shard"]
        if shard not in shard_metadata:
            shard_metadata[shard] = json.loads(
                (root / "meta" / f"shard_{shard:05d}.json").read_text()
            )["rows"]
        target = shard_metadata[shard][coordinate["row_in_shard"]]
        source = sources[target["source_row_id"]]
        ids = source["tokenization"]["input_ids"]
        if ids[target["token_index"]] != target["token_id"]:
            raise ValueError(f"Token provenance mismatch at row {coordinate['global_row']}")
        index.append({
            **coordinate, **target,
            "layer": layer, "token_slot": 0,
            "activation_path": str(root / f"layer{layer:02d}" / f"shard_{shard:05d}.pt"),
            "prompt": source["prompt"], "labels": source["labels"],
            "input_ids": ids,
        })
    if len(index) != config["num_rows"]:
        raise ValueError("Activation index does not cover the configured number of rows")
    return index
