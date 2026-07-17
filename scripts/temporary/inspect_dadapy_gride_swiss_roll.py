"""Inspect DADApy's multiscale GRIDE output on a Swiss roll."""

import argparse

import numpy as np
from dadapy.data import Data
from sklearn.datasets import make_swiss_roll


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--range-max", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.n_samples < 3:
        raise SystemExit("--n-samples must be at least 3")
    if args.range_max < 2:
        raise SystemExit("--range-max must be at least 2")

    X, _ = make_swiss_roll(
        n_samples=args.n_samples,
        noise=args.noise,
        random_state=args.seed,
    )
    effective_range_max = min(args.range_max, args.n_samples - 1)
    ids, errors, scales = Data(coordinates=X, n_jobs=1).return_id_scaling_gride(
        range_max=effective_range_max
    )

    np.set_printoptions(precision=4, suppress=True)
    print(f"X: shape={X.shape} dtype={X.dtype}")
    print(f"range_max: requested={args.range_max} effective={effective_range_max}")
    for name, values in (
        ("ids", ids),
        ("errors", errors),
        ("scales", scales),
    ):
        values = np.asarray(values)
        print(f"{name}: type={type(values).__name__} shape={values.shape} dtype={values.dtype}")
        print(values)
    print(f"median ID: {np.median(ids):.4f} (Swiss-roll reference: 2)")


if __name__ == "__main__":
    main()
