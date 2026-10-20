"""Selection across several runs: what a bigger candidate pool costs (H2, H4, T5).

Two claims in the paper needed this and nothing else produced them.

H2: running more seeds does not reduce the winner's curse. A practitioner who
trains three seeds and keeps the best one has not escaped the selection floor.
They have moved an argmax over twenty epochs to an argmax over sixty.

H4: selection *across* runs costs more floor than selection *within* one run,
and the cost grows with the number of runs.

Both fall out of one measurement. Stacking the recorded correctness matrices of
K runs produces a candidate pool of K x n_epochs checkpoints, which is exactly
the set a practitioner selects from, and replaying the selection step over it
measures the floor directly. No retraining is involved: the matrices are
already on disk.

Two implementation choices carry the argument. Every K sees the same sequence
of validation draws, so the comparison between K values is paired and the
difference cannot be a difference of luck. And the runs are stacked in seed
order, so the comparison is deterministic and reproducible.

`stack_correctness` from the selector module stacks the pool axis and is used
here unchanged. The test axis is the mirror of the same operation over the same
directories, and the two are asserted to have the same shape, so a row can
never refer to one checkpoint on the pool axis and a different one on the test
axis.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.constants import REPLAY_SEED, REPLAY_TRIALS, VALIDATION_SUBSET_SIZES
from src.engine.selector import (load_correctness, replay_select, stack_correctness)
from src.utils.common import setup_logging

log = logging.getLogger("selection-floor")

SELECTION_RULES = ("argmax", "one_se")


@dataclass(frozen=True)
class Run:
    """One trained cell: where it lives and how it was identified."""

    stem: str
    directory: Path
    dataset: str
    model: str
    loss: str
    seed: int


def identity_from_config(config_path: Path) -> dict:
    """Read a cell's dataset, model, loss and seed from the config that produced it.

    The checkpoint directory name carries the identity too, but only the config
    is authoritative: a tagged arm prefixes its directories, and parsing a name
    would misread the tag as a dataset.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = raw.get("data", {}) or {}
    model = raw.get("model", {}) or {}
    loss = raw.get("loss", {}) or {}
    train = raw.get("train", {}) or {}
    return {
        "dataset": str(data.get("dataset", "unknown")),
        "model": str(model.get("name", "unknown")),
        "loss": str(loss.get("name", "unknown")),
        "seed": int(train.get("seed", 0)),
    }


def discover_runs(ckpt_glob: str, config_glob: str) -> list[Run]:
    """Pair every trained cell with the config that produced it."""
    configs: dict[str, Path] = {}
    for path in sorted(Path().glob(config_glob)):
        configs[path.stem] = path
    runs: list[Run] = []
    for directory in sorted(Path().glob(ckpt_glob)):
        if not directory.is_dir():
            continue
        name = directory.name
        stem = name[len("ckpt-"):] if name.startswith("ckpt-") else name
        config_path = configs.get(stem)
        if config_path is None:
            log.warning("no config for %s; skipping", directory)
            continue
        identity = identity_from_config(config_path)
        runs.append(Run(stem=stem, directory=directory, **identity))
    return runs


def group_runs(runs: list[Run]) -> list[tuple[tuple[str, str, str], list[Run]]]:
    """Group cells by dataset, model and loss, ordered by seed inside each group."""
    groups: dict[tuple[str, str, str], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.dataset, run.model, run.loss), []).append(run)
    ordered = []
    for key in sorted(groups):
        ordered.append((key, sorted(groups[key], key=lambda r: r.seed)))
    return ordered


def stack_test(directories: list[Path]) -> np.ndarray:
    """Row-stack the test correctness matrices of the given runs."""
    rows = []
    for directory in directories:
        _, test = load_correctness(directory)
        rows.append(test)
    return np.concatenate(rows, axis=0)


def measure_group(group: list[Run], sizes: list[int], trials: int, seed: int) -> list[dict]:
    """Measure the selection floor for K = 1, 2, ... runs, on a shared draw stream."""
    rows: list[dict] = []
    for k in range(1, len(group) + 1):
        subset = group[:k]
        pool = stack_correctness([run.directory for run in subset])
        test = stack_test([run.directory for run in subset])
        if pool.shape[0] != test.shape[0]:
            raise ValueError(
                f"pool has {pool.shape[0]} candidate rows but test has {test.shape[0]}; "
                "the two axes must describe the same checkpoints")
        for n_val in sizes:
            for rule in SELECTION_RULES:
                result = replay_select(pool, test, rule, n_val, trials, seed + n_val)
                rows.append({
                    "k_runs": k,
                    "n_candidates": int(pool.shape[0]),
                    "n_val": result.n_val,
                    "rule": result.rule,
                    "mean_selected": result.mean_selected,
                    "sem_selected": result.sem_selected,
                    "optimism": result.optimism,
                    "regret": result.regret,
                    "mean_val_selected": result.mean_val_selected,
                    "oracle_test": result.oracle_test,
                    "frac_best": result.frac_best,
                })
    single = {(row["n_val"], row["rule"]): row for row in rows if row["k_runs"] == 1}
    for row in rows:
        base = single.get((row["n_val"], row["rule"]))
        row["optimism_vs_single_run"] = (row["optimism"] - base["optimism"]) if base else 0.0
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the floor across several runs")
    parser.add_argument("--ckpt-glob", default="outputs/ckpt-*")
    parser.add_argument("--config-glob", default="outputs/tmp-configs/*.yaml")
    parser.add_argument("--out", default="outputs/seed-studies/seed.json")
    parser.add_argument("--trials", type=int, default=REPLAY_TRIALS)
    parser.add_argument("--seed", type=int, default=REPLAY_SEED)
    parser.add_argument("--sizes", default=",".join(str(s) for s in VALIDATION_SUBSET_SIZES))
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    runs = discover_runs(args.ckpt_glob, args.config_glob)
    if not runs:
        raise SystemExit(f"no trained cells matched {args.ckpt_glob}")
    groups = group_runs(runs)
    payload_groups = []
    for key, group in groups:
        rows = measure_group(group, sizes, args.trials, args.seed)
        payload_groups.append({
            "dataset": key[0], "model": key[1], "loss": key[2],
            "seeds": [run.seed for run in group],
            "rows": rows,
        })
        log.info("%s/%s/%s: %d run(s), %d row(s)", key[0], key[1], key[2], len(group), len(rows))
    payload = {
        "trials": args.trials,
        "sizes": sizes,
        "n_groups": len(payload_groups),
        "groups": payload_groups,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", out)

    print(f"{'group':34s} {'k':>2s} {'n_val':>6s} {'rule':>8s} {'optim':>9s} {'vs 1 run':>9s} {'cand':>5s}")
    for group in payload_groups:
        label = f"{group['dataset']}/{group['model']}/{group['loss']}"
        for row in group["rows"]:
            print(f"{label:34s} {row['k_runs']:2d} {row['n_val']:6d} {row['rule']:>8s} "
                  f"{100 * row['optimism']:9.3f} {100 * row['optimism_vs_single_run']:9.3f} "
                  f"{row['n_candidates']:5d}")


if __name__ == "__main__":
    main()
