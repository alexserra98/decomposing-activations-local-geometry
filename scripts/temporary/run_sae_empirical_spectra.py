"""Empirical spectra for the eight years/temperature K=50, threshold=0.1 runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def selected_runs():
    """Resolve one completed run per dataset, layer, and model from saved configs."""
    runs = []
    for dataset in ("years_neutral", "temperature_02d"):
        for layer in (4, 28):
            for method in ("hddc_em", "kmeans_pca"):
                root = REPO / f"dalg-cache/sae_manifold_{dataset}_llama31_8b_last/sae_manifold_{dataset}_{method}"
                matches = []
                for config in sorted(root.glob("*/config.json")):
                    cfg = json.loads(config.read_text())
                    if (cfg["K"], cfg["layer"], cfg["surgery_threshold"]) == (50, layer, 0.1):
                        matches.append(config.parent)
                if len(matches) != 1:
                    raise ValueError(f"Expected one {dataset}/{method}/layer{layer} run; found {matches}")
                run = matches[0]
                checkpoint = run / ("mfa_model.pt" if method == "hddc_em" else "kmeans_model.pt")
                if not checkpoint.is_file() or not (run / "TRAINING_COMPLETED.json").is_file():
                    raise FileNotFoundError(f"Incomplete training run: {run}")
                runs.append(dict(dataset=dataset, layer=layer, method=method,
                                 checkpoint=checkpoint, output=run / "empirical_spectra.pt"))
    return runs


def main():
    from compute_empirical_spectra import export_spectra

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, choices=range(8))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eig-batch-size", type=int, default=1)
    args = parser.parse_args()
    runs = selected_runs()
    if args.task_index is not None:
        runs = [runs[args.task_index]]
    for run in runs:
        print(f"Analyzing {run['dataset']} / {run['method']} / layer {run['layer']}", flush=True)
        export_spectra(run["checkpoint"], run["output"], device=args.device,
                       batch_size=args.batch_size, eig_batch_size=args.eig_batch_size)


if __name__ == "__main__":
    main()
