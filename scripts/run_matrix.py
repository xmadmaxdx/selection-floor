"""Sweep datasets x losses x backbones x seeds. Checkpointed per cell."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.constants import DATASET_CLASS_COUNTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the selection-floor training matrix")
    parser.add_argument("--datasets", default="mnist,fashionmnist")
    parser.add_argument("--losses", default="ce,label_smoothing,focal")
    parser.add_argument("--models", default="simplecnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Persist per-epoch weights; required by evaluate.py")
    parser.add_argument("--floor-stop", action="store_true", help="Halt when the curve is unresolvable")
    parser.add_argument("--ckpt-tag", default="",
                        help="Namespace this arm's outputs, e.g. 'fs'. Two arms must not share a path.")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class CellPaths:
    """Every output path one matrix cell owns.

    A second arm of the matrix, such as the FloorStop arm, must not share a
    path with the first. When it did, the arm found every cell already
    complete, skipped all of them, and wrote a manifest reporting a success
    that never happened. The tag keeps the two arms in separate namespaces.
    """

    stem: str
    checkpoint_dir: str
    config_name: str


def manifest_path(tag: str = "") -> Path:
    """Where the manifest for one arm is written."""
    name = f"matrix-manifest-{tag}.json" if tag else "matrix-manifest.json"
    return Path("outputs") / name


def cell_paths(dataset: str, model: str, loss: str, seed: int, multi_seed: bool,
               tag: str = "") -> CellPaths:
    """Output paths for one cell of one arm."""
    suffix = f"-s{seed}" if multi_seed else ""
    stem = f"{dataset}-{model}-{loss}{suffix}"
    labelled = f"{tag}-{stem}" if tag else stem
    return CellPaths(
        stem=stem,
        checkpoint_dir=f"outputs/ckpt-{labelled}",
        config_name=f"{labelled}.yaml",
    )


def _write_manifest(completed: list[str], failed: dict[str, int], path: Path) -> Path:
    """Persist which cells finished and which failed, so a short run is visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "completed": sorted(completed),
        "failed": failed,
        "n_completed": len(completed),
        "n_failed": len(failed),
    }, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    datasets = _csv(args.datasets)
    losses = _csv(args.losses)
    models = _csv(args.models)
    seeds = [int(s) for s in _csv(args.seeds)]
    if not seeds:
        raise SystemExit("empty --seeds")
    for name in datasets:
        if name not in DATASET_CLASS_COUNTS:
            raise SystemExit(f"unknown dataset: {name}")
    multi_seed = len(seeds) > 1 or seeds[0] != 0
    manifest = manifest_path(args.ckpt_tag)
    tmp = Path("outputs/tmp-configs")
    tmp.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    failed: dict[str, int] = {}
    for dataset, loss, model, seed in itertools.product(datasets, losses, models, seeds):
        paths = cell_paths(dataset, model, loss, seed, multi_seed, args.ckpt_tag)
        checkpoint_dir = paths.checkpoint_dir
        label_smoothing = 0.05 if loss == "label_smoothing" else 0.0
        config = {
            "data": {"dataset": dataset, "root": "data", "batch_size": 128,
                     "num_workers": 2, "image_size": 32 if dataset == "cifar10" else 28},
            "model": {"name": model, "num_classes": DATASET_CLASS_COUNTS[dataset]},
            "loss": {"name": loss, "gamma": 2.0, "label_smoothing": label_smoothing},
            "train": {"epochs": args.epochs, "lr": 0.001, "optimizer": "adamw", "seed": seed,
                      "keep_checkpoints": bool(args.keep_checkpoints),
                      "floor_stop": bool(args.floor_stop),
                      "checkpoint_dir": checkpoint_dir},
        }
        path = tmp / paths.config_name
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        command = [sys.executable, "-m", "scripts.train", "--config", str(path)]
        if args.download:
            command.append("--download")
        print("RUN:", " ".join(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            completed.append(paths.stem)
        else:
            failed[paths.stem] = result.returncode
            print(f"FAILED cell {paths.stem} (exit {result.returncode}), continuing", flush=True)
        _write_manifest(completed, failed, manifest)

    # A partial matrix must not look like a complete one.
    manifest = _write_manifest(completed, failed, manifest)
    if failed:
        print(f"WARNING: {len(failed)} cell(s) failed; see {manifest}", flush=True)
        raise SystemExit(1)
    print(f"all {len(completed)} cells completed; manifest at {manifest}", flush=True)


if __name__ == "__main__":
    main()