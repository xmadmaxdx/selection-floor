"""Download the small-image datasets. Colab-only entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.constants import DATASET_CLASS_COUNTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Download small-image datasets")
    parser.add_argument("--root", default="data")
    parser.add_argument("--sets", default="mnist,fashionmnist")
    args = parser.parse_args()
    import torchvision.datasets as tvd

    mapping = {"mnist": tvd.MNIST, "fashionmnist": tvd.FashionMNIST,
               "kmnist": tvd.KMNIST, "cifar10": tvd.CIFAR10}
    Path(args.root).mkdir(parents=True, exist_ok=True)
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        if name not in mapping or name not in DATASET_CLASS_COUNTS:
            raise SystemExit(f"unknown dataset: {name}")
        for train in (True, False):
            mapping[name](root=args.root, train=train, download=True)
        print(f"ok: {name} -> {args.root}", flush=True)


if __name__ == "__main__":
    main()