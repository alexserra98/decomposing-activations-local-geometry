"""Extract last-token prompt activations with source provenance."""

from __future__ import annotations

import argparse

from dalg.llm.prompt_extraction import DTYPES, run_prompt_extraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Prompt JSONL file or directory containing prompts.jsonl.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Hugging Face ID or local Llama checkpoint.")
    parser.add_argument("--revision", help="Model/tokenizer revision; resolved to an immutable commit.")
    parser.add_argument("--layers", type=int, nargs="+", help="Zero-based decoder blocks (default: all).")
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, help="Reject longer inputs, including BOS; never truncate.")
    parser.add_argument("--cache-dir", default="dalg-cache/huggingface/hub")
    parser.add_argument("--preview", action="store_true", help="Inspect token selection without loading model weights or writing outputs.")
    return parser


def main() -> None:
    args = vars(build_parser().parse_args())
    args["model_name"] = args.pop("model")
    args["batch_size"] = args.pop("extract_batch_size")
    run_prompt_extraction(**args)


if __name__ == "__main__":
    main()
