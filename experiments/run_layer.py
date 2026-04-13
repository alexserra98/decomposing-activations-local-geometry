"""
CLI experiment runner for MFA pipeline.

Subcommands:
    extract       Extract activations for a single layer and save to disk.
    train         Train MFA on pre-extracted activations.
    intrinsic-dim Compute intrinsic dimensionality per cluster.
    overlap       Compute pairwise overlap metrics between components.
    all           Run extract + train + intrinsic-dim + overlap in sequence.

Example usage:
    # Step 1: extract (expensive LLM forward pass)
    python experiments/run_layer.py extract \
        --model gpt2-small --dataset ./data/supervised.json \
        --layer 4 --out-dir results/gpt2-small/layer_04 \
        --max-tokens 250000 --device cuda

    # Step 2: train (can re-run with different K without re-extracting)
    python experiments/run_layer.py train \
        --data-dir results/gpt2-small/layer_04 \
        --K 500 --rank 10 --epochs 10 --device cuda

    # Step 3: analysis
    python experiments/run_layer.py overlap \
        --data-dir results/gpt2-small/layer_04 --device cuda
    python experiments/run_layer.py intrinsic-dim \
        --data-dir results/gpt2-small/layer_04 --device cuda
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch.utils.data import DataLoader, TensorDataset


# ── Subcommand implementations ──────────────────────────────────────────


def cmd_extract(args):
    """Extract activations for one layer and save to disk."""
    from llm_utils.activation_generator import ActivationGenerator, extract_token_ids
    from data_utils.concept_dataset import SupervisedConceptDataset, ConceptDataset

    os.makedirs(args.out_dir, exist_ok=True)

    # Try supervised first, fall back to unsupervised
    try:
        dataset = SupervisedConceptDataset(args.dataset)
    except Exception:
        dataset = ConceptDataset(args.dataset)
    print(f"Dataset: {len(dataset)} samples")

    act_gen = ActivationGenerator(
        args.model,
        model_device=args.device,
        data_device=args.device,
        mode=args.mode,
    )

    layer = args.layer
    activations, freq = act_gen.generate_activations(
        dataset, [layer], batch_size=args.extract_batch_size,
    )
    tokens, _, _ = extract_token_ids(dataset, act_gen, batch_size=args.extract_batch_size)

    X = activations[0]
    if args.max_tokens > 0 and X.shape[0] > args.max_tokens:
        X = X[: args.max_tokens]
        tokens = tokens[: args.max_tokens]
        freq = freq[: args.max_tokens]

    # Save
    torch.save(X.cpu(), os.path.join(args.out_dir, "activations.pt"))
    torch.save(tokens.cpu(), os.path.join(args.out_dir, "tokens.pt"))
    torch.save(freq.cpu(), os.path.join(args.out_dir, "freq.pt"))

    config = {
        "model": args.model, "dataset": args.dataset,
        "layer": layer, "mode": args.mode,
        "N": int(X.shape[0]), "D": int(X.shape[1]),
        "max_tokens": args.max_tokens,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved activations {X.shape} to {args.out_dir}")


def cmd_train(args):
    """Initialize centroids and train MFA on pre-extracted activations."""
    from initializations.projected_knn import ReservoirKMeans
    from modeling.mfa import MFA, save_mfa
    from modeling.train import train_nll

    data_dir = args.data_dir
    X = torch.load(os.path.join(data_dir, "activations.pt"), weights_only=True)
    tok = torch.load(os.path.join(data_dir, "tokens.pt"), weights_only=True)
    print(f"Loaded activations: {X.shape}")

    # Load config to get vocab_size if available
    config_path = os.path.join(data_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    full_ds = TensorDataset(X, tok)
    loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Try loading freq for weighted sampling
    freq_path = os.path.join(data_dir, "freq.pt")
    token_loader = None
    if os.path.exists(freq_path):
        token_loader = DataLoader(TensorDataset(tok), batch_size=args.batch_size)

    # Initialization
    pool_size = max(args.K * 2, len(full_ds) // 5)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    knn = ReservoirKMeans(
        n_clusters=args.K,
        pool_size=pool_size,
        vocab_size=args.vocab_size,
        device=args.device,
        proj_dim=args.proj_dim,
        seed=args.seed,
    )
    centroids = knn.fit(loader, token_loader=token_loader, refine_epochs=args.refine_epochs)
    torch.save(centroids.cpu(), os.path.join(data_dir, "centroids.pt"))
    print(f"Centroids: {centroids.shape}")

    # Train MFA
    model = MFA(centroids=centroids, rank=args.rank).to(args.device)

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    train_nll(
        model, loader,
        epochs=args.epochs, lr=args.lr,
        grad_clip=args.grad_clip,
        save_path=os.path.join(data_dir, "mfa_model.pt"),
        save_func=save_mfa,
    )

    # Save final model (unwrap compiled model if needed)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_mfa(raw_model, os.path.join(data_dir, "mfa_model.pt"))

    # Update config
    config.update({
        "K": args.K, "rank": args.rank,
        "epochs": args.epochs, "lr": args.lr,
    })
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {data_dir}/mfa_model.pt")


def cmd_overlap(args):
    """Compute pairwise overlap metrics between MFA components."""
    from experiments.cluster_overlap import compute_overlap

    data_dir = args.data_dir
    model_path = os.path.join(data_dir, "mfa_model.pt")
    results = compute_overlap(model_path, device=args.device, batch_pairs=args.batch_pairs)

    save_path = os.path.join(data_dir, "overlap.pt")
    torch.save(results, save_path)
    print(f"Overlap saved to {save_path}")


def cmd_intrinsic_dim(args):
    """Compute intrinsic dimensionality per cluster."""
    from experiments.cluster_intrinsic_dim import compute_intrinsic_dims

    data_dir = args.data_dir
    model_path = os.path.join(data_dir, "mfa_model.pt")
    act_path = os.path.join(data_dir, "activations.pt")
    tok_path = os.path.join(data_dir, "tokens.pt")

    results = compute_intrinsic_dims(
        model_path, act_path, tok_path,
        device=args.device,
        batch_size=args.batch_size,
        variance_threshold=args.variance_threshold,
        min_population=args.min_population,
        max_samples=args.max_samples_per_cluster,
    )

    save_path = os.path.join(data_dir, "intrinsic_dims.pt")
    torch.save(results, save_path)
    print(f"Intrinsic dims saved to {save_path}")


def cmd_all(args):
    """Run the full pipeline: extract → train → overlap → intrinsic-dim."""
    # For 'all', --out-dir is required and becomes --data-dir for subsequent steps
    args.data_dir = args.out_dir

    print("=" * 60)
    print("STEP 1: Extract activations")
    print("=" * 60)
    cmd_extract(args)

    print("\n" + "=" * 60)
    print("STEP 2: Train MFA")
    print("=" * 60)
    cmd_train(args)

    print("\n" + "=" * 60)
    print("STEP 3: Compute overlap")
    print("=" * 60)
    cmd_overlap(args)

    print("\n" + "=" * 60)
    print("STEP 4: Compute intrinsic dimensions")
    print("=" * 60)
    cmd_intrinsic_dim(args)

    print("\n" + "=" * 60)
    print("DONE — all results in", args.out_dir)
    print("=" * 60)


# ── Argument parsing ────────────────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(description="MFA experiment runner (per-layer)")
    sub = p.add_subparsers(dest="command", required=True)

    # -- shared args --
    def add_common(sp):
        sp.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--batch-size", type=int, default=128)

    # -- extract --
    sp = sub.add_parser("extract", help="Extract activations for one layer")
    add_common(sp)
    sp.add_argument("--model", required=True, help="TransformerLens model name")
    sp.add_argument("--dataset", required=True, help="Path to dataset (JSON/JSONL/CSV)")
    sp.add_argument("--layer", type=int, required=True)
    sp.add_argument("--out-dir", required=True, help="Output directory")
    sp.add_argument("--max-tokens", type=int, default=0, help="Cap on tokens (0=no cap)")
    sp.add_argument("--mode", default="residual", help="Activation stream")
    sp.add_argument("--extract-batch-size", type=int, default=5, help="Batch size for LLM forward pass")
    sp.set_defaults(func=cmd_extract)

    # -- train --
    sp = sub.add_parser("train", help="Train MFA on pre-extracted activations")
    add_common(sp)
    sp.add_argument("--data-dir", required=True, help="Directory with activations.pt, tokens.pt")
    sp.add_argument("--K", type=int, required=True, help="Number of components")
    sp.add_argument("--rank", type=int, default=10, help="MFA rank (q)")
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--grad-clip", type=float, default=None)
    sp.add_argument("--proj-dim", type=int, default=32, help="Projection dim for KMeans init")
    sp.add_argument("--refine-epochs", type=int, default=25, help="Lloyd refinement epochs")
    sp.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size for weighted sampling")
    sp.add_argument("--compile", action="store_true", help="Use torch.compile")
    sp.set_defaults(func=cmd_train)

    # -- overlap --
    sp = sub.add_parser("overlap", help="Compute pairwise overlap metrics")
    add_common(sp)
    sp.add_argument("--data-dir", required=True)
    sp.add_argument("--batch-pairs", type=int, default=4096, help="Pairs per batch for vectorized overlap")
    sp.set_defaults(func=cmd_overlap)

    # -- intrinsic-dim --
    sp = sub.add_parser("intrinsic-dim", help="Compute intrinsic dim per cluster")
    add_common(sp)
    sp.add_argument("--data-dir", required=True)
    sp.add_argument("--variance-threshold", type=float, default=0.90)
    sp.add_argument("--min-population", type=int, default=100)
    sp.add_argument("--max-samples-per-cluster", type=int, default=10000)
    sp.set_defaults(func=cmd_intrinsic_dim)

    # -- all --
    sp = sub.add_parser("all", help="Run full pipeline: extract → train → overlap → intrinsic-dim")
    add_common(sp)
    # extract args
    sp.add_argument("--model", required=True)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--layer", type=int, required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--max-tokens", type=int, default=0)
    sp.add_argument("--mode", default="residual")
    sp.add_argument("--extract-batch-size", type=int, default=5)
    # train args
    sp.add_argument("--K", type=int, required=True)
    sp.add_argument("--rank", type=int, default=10)
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--grad-clip", type=float, default=None)
    sp.add_argument("--proj-dim", type=int, default=32)
    sp.add_argument("--refine-epochs", type=int, default=25)
    sp.add_argument("--vocab-size", type=int, default=50257)
    sp.add_argument("--compile", action="store_true")
    # overlap args
    sp.add_argument("--batch-pairs", type=int, default=4096)
    # intrinsic-dim args
    sp.add_argument("--variance-threshold", type=float, default=0.90)
    sp.add_argument("--min-population", type=int, default=100)
    sp.add_argument("--max-samples-per-cluster", type=int, default=10000)
    sp.set_defaults(func=cmd_all)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
