"""Offline smoke checks for prompt extraction using a tiny native Llama model.

Run from the repository root with PYTHONPATH=src. All fixtures are temporary.
No downloaded weights, GPUs, or credentials are required.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, PreTrainedTokenizerFast

from dalg.data.prompt_activations import load_prompt_activation_index, load_prompt_records
from dalg.data.shard_activations import ActivationBatchDataset
from dalg.llm.prompt_extraction import collect_prompt_activations, run_prompt_extraction, tokenize_prompt_records


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    records = [
        {"row_id": 91, "prompt": "They are 7 years old. ", "subset": "age", "labels": {"age": 7.0}},
        {"row_id": 8, "prompt": "It's midnight on Monday", "subset": "days", "labels": {"day": "Monday", "time_idx": 7}},
        {"row_id": 44, "prompt": "Café, 日本語, and a longer sentence.", "subset": "text", "labels": {"nested": {"values": [1, None, "é"]}}},
        {"row_id": 19, "prompt": "Hi!", "subset": "text", "labels": {}},
        {"row_id": 22, "prompt": "They are 7 years old. ", "subset": "age", "labels": {"age": 7.0}},
    ]
    with tempfile.TemporaryDirectory(prefix="dalg_prompt_smoke_") as directory:
        root = Path(directory)
        checkpoint = root / "tiny_llama"
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(
            [record["prompt"] for record in records],
            trainers.BpeTrainer(vocab_size=280, special_tokens=["<unk>", "<bos>", "<eos>"],
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
        )
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                            bos_token="<bos>", eos_token="<eos>")
        tokenizer.save_pretrained(checkpoint)
        cfg = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                          num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=128, bos_token_id=tokenizer.bos_token_id,
                          eos_token_id=tokenizer.eos_token_id)
        model = LlamaModel(cfg).eval()
        model.save_pretrained(checkpoint)
        data = root / "input"
        data.mkdir()
        (data / "prompts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records))
        (data / "metadata.json").write_text(json.dumps({"origin": "offline validation", "nested": {"keep": True}}))
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        prepared = tokenize_prompt_records(load_prompt_records(data), tokenizer, max_length=128)
        common = dict(layers=[0, 1, 2], pad_token_id=tokenizer.eos_token_id, device="cpu", dtype=torch.float32)
        batched = collect_prompt_activations(model, prepared, **common)
        hook_counts = [len(layer._forward_hooks) for layer in model.layers]
        for i, record in enumerate(prepared):
            individual = collect_prompt_activations(model, [record], **common)
            for layer in common["layers"]:
                torch.testing.assert_close(batched[layer][i], individual[layer][0], atol=1e-6, rtol=1e-5)
        assert [len(layer._forward_hooks) for layer in model.layers] == hook_counts
        ids = torch.tensor([prepared[0]["tokenization"]["input_ids"]])
        reference = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        for layer in [0, 1]:
            torch.testing.assert_close(batched[layer][0], reference.hidden_states[layer+1][0, -1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model.norm(batched[2][0]), reference.last_hidden_state[0, -1], atol=1e-6, rtol=1e-5)
        assert not torch.allclose(batched[2][0], reference.last_hidden_state[0, -1])
        print("PASS: mixed-length batching agrees with individual forward passes; final block precedes final norm")

        out = root / "activations"
        kwargs = dict(dataset=data, out_dir=out, model_name=str(checkpoint), device="cpu", dtype="float32",
                      batch_size=2, shard_size=2, cache_dir=root / "cache")
        run_prompt_extraction(**kwargs, preview=True)
        assert not out.exists()
        run_prompt_extraction(**kwargs)
        index = load_prompt_activation_index(out, layer=2)
        assert [row["source_row_id"] for row in index] == [row["row_id"] for row in records]
        for original, actual in zip(records, index):
            assert actual["prompt"] == original["prompt"] and actual["labels"] == original["labels"]
            assert actual["token_index"] == len(actual["input_ids"]) - 1
            assert actual["input_ids"][actual["token_index"]] == actual["token_id"]
        assert index[0]["prompt"].endswith(" ")
        assert index[0]["decoded_token"] == " "
        for layer in common["layers"]:
            stream = ActivationBatchDataset(out, layer, batch_size=3, return_metadata=True,
                                            shuffle_shards=False, shuffle_within_shard=False)
            tensors, global_rows, slots = zip(*list(stream))
            torch.testing.assert_close(torch.cat(tensors), batched[layer], atol=1e-6, rtol=1e-5)
            assert torch.cat(global_rows).tolist() == list(range(len(records)))
            assert torch.cat(slots).tolist() == [0] * len(records)
        saved_metadata = json.loads((out / "source/dataset_metadata.json").read_text())
        assert saved_metadata["nested"] == {"keep": True}
        print("PASS: source/label/token round trip, preserved whitespace/duplicates, existing shard reader")

        before = {p: p.stat().st_mtime_ns for p in out.glob("layer*/*.pt")}
        with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("completed run must not load weights")):
            run_prompt_extraction(**{**kwargs, "batch_size": 1})
        assert before == {p: p.stat().st_mtime_ns for p in before}
        broken = out / "layer01/shard_00001.pt"
        broken.unlink()
        run_prompt_extraction(**kwargs)
        assert broken.exists()
        assert (out / "layer00/shard_00000.pt").stat().st_mtime_ns == before[out / "layer00/shard_00000.pt"]
        assert (out / "layer00/shard_00002.pt").stat().st_mtime_ns == before[out / "layer00/shard_00002.pt"]
        print("PASS: completed resume skips weights; incomplete shard is repaired without rewriting other shards")
        try:
            run_prompt_extraction(**{**kwargs, "shard_size": 3})
        except ValueError as exc:
            assert "changed settings" in str(exc)
        else:
            raise AssertionError("resume accepted changed shard boundaries")
        try:
            tokenize_prompt_records(records, tokenizer, max_length=3)
        except ValueError as exc:
            assert "never truncated" in str(exc)
        else:
            raise AssertionError("overlong prompt was accepted")
        print("PASS: incompatible resume and overlong prompts fail explicitly")
        print("All offline extraction smoke checks passed.")


if __name__ == "__main__":
    main()
