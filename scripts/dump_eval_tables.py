"""Per-cell metric tables (T4) and the FloorStop cost table (T6).

`evaluate.py` writes one JSON per trained cell, each holding the test metrics of
every checkpoint a selection rule can pick. Nothing rendered them, so the
paper's per-cell table would have been filled in by hand, which is the one
thing this repository exists to prevent.

T6 compares the main arm against the FloorStop arm, and needs both. The claim
is that stopping early costs no measurable accuracy, and that is a comparison,
not a property of either arm alone.

The two arms share a trajectory up to the stopping epoch: same seed, same data
order, same schedule. So the main arm's test accuracy at the FloorStop arm's
selected epoch is the fair comparator, and the difference between the two
should be zero up to GPU nondeterminism. Reporting it is a check on the arms
being comparable at all, which matters because a silent difference would mean
the arms are not the same experiment.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.seed_study import identity_from_config
from src.engine.selector import load_correctness
from src.utils.common import setup_logging

log = logging.getLogger("selection-floor")


def _configs(config_glob: str) -> dict[str, Path]:
    return {path.stem: path for path in sorted(Path().glob(config_glob))}


def load_payloads(glob_pattern: str, configs: dict[str, Path]) -> list[dict]:
    """Read every evaluation JSON and attach the identity of the cell it describes."""
    payloads: list[dict] = []
    for path in sorted(Path().glob(glob_pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        directory = Path(payload.get("checkpoint_dir", ""))
        stem = directory.name[len("ckpt-"):] if directory.name.startswith("ckpt-") else directory.name
        config_path = configs.get(stem)
        identity = (identity_from_config(config_path) if config_path is not None
                    else {"dataset": "unknown", "model": "unknown", "loss": "unknown",
                          "seed": int(payload.get("seed", 0))})
        if config_path is None:
            log.warning("no config for %s; identity left unknown", path)
        payload["_stem"] = stem
        payload["_identity"] = identity
        payloads.append(payload)
    return payloads


def eval_rows(payloads: list[dict]) -> list[dict]:
    """One row per cell and selection variant."""
    rows: list[dict] = []
    for payload in payloads:
        identity = payload["_identity"]
        for variant, metrics in payload.get("variants", {}).items():
            rows.append({
                "dataset": identity["dataset"],
                "model": identity["model"],
                "loss": identity["loss"],
                "seed": identity["seed"],
                "variant": variant,
                "epoch": metrics.get("epoch"),
                "acc": metrics.get("acc"),
                "macro_f1": metrics.get("macro_f1"),
                "bal_acc": metrics.get("bal_acc"),
                "selected_by": metrics.get("selected_by"),
                "n_epochs": payload.get("n_epochs"),
                "floor_stop_epoch": payload.get("floor_stop_epoch"),
            })
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Mean and spread of each variant's accuracy over the seeds of one cell group."""
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["loss"], row["variant"])
        groups.setdefault(key, []).append(row)
    aggregated: list[dict] = []
    for key in sorted(groups):
        members = groups[key]
        accuracies = np.asarray([m["acc"] for m in members if m["acc"] is not None], dtype=float)
        if accuracies.size == 0:
            continue
        aggregated.append({
            "dataset": key[0], "model": key[1], "loss": key[2], "variant": key[3],
            "n_seeds": int(accuracies.size),
            "mean_acc": float(accuracies.mean()),
            "sd_acc": float(accuracies.std(ddof=1)) if accuracies.size > 1 else 0.0,
            "min_acc": float(accuracies.min()),
            "max_acc": float(accuracies.max()),
        })
    return aggregated


def floor_stop_rows(main_payloads: list[dict], floor_payloads: list[dict]) -> list[dict]:
    """Compare the FloorStop arm against the main arm, cell by cell."""
    main_by_key = {(_key(p)): p for p in main_payloads}
    rows: list[dict] = []
    for payload in floor_payloads:
        key = _key(payload)
        main = main_by_key.get(key)
        if main is None:
            log.warning("no main-arm cell matches %s; skipping", key)
            continue
        variants = payload.get("variants", {})
        if "argmax" not in variants:
            log.warning("FloorStop cell %s has no argmax variant; skipping", key)
            continue
        selected = variants["argmax"]
        epoch = selected.get("epoch")
        main_epochs = int(main.get("n_epochs", 0))
        fs_epochs = int(payload.get("n_epochs", 0))
        row = {
            "dataset": key[0], "model": key[1], "loss": key[2], "seed": key[3],
            "main_epochs": main_epochs,
            "fs_epochs": fs_epochs,
            "epochs_saved": main_epochs - fs_epochs,
            "fs_selected_epoch": epoch,
            "fs_selected_acc": selected.get("acc"),
            "main_argmax_acc": main.get("variants", {}).get("argmax", {}).get("acc"),
        }
        if epoch is not None and 1 <= int(epoch) <= main_epochs:
            _, test_correct = load_correctness(main.get("checkpoint_dir", ""))
            same_epoch_acc = float(test_correct[int(epoch) - 1].mean())
            oracle_acc = float(test_correct.mean(axis=1).max())
            row["main_acc_at_same_epoch"] = same_epoch_acc
            row["main_oracle_acc"] = oracle_acc
            row["delta_same_epoch"] = float(selected.get("acc")) - same_epoch_acc
            row["delta_vs_oracle"] = float(selected.get("acc")) - oracle_acc
            if row["main_argmax_acc"] is not None:
                row["delta_vs_main_argmax"] = float(selected.get("acc")) - float(row["main_argmax_acc"])
        rows.append(row)
    return rows


def _key(payload: dict) -> tuple[str, str, str, int]:
    identity = payload["_identity"]
    return (identity["dataset"], identity["model"], identity["loss"], int(identity["seed"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the per-cell evaluation tables")
    parser.add_argument("--glob", default="outputs/evals/*.json")
    parser.add_argument("--config-glob", default="outputs/tmp-configs/*.yaml")
    parser.add_argument("--floor-stop-glob", default="outputs/evals/fs-*.json",
                        help="Evaluation JSONs of the FloorStop arm; empty disables T6")
    parser.add_argument("--out", default="outputs/tables/evals.json")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    configs = _configs(args.config_glob)
    payloads = load_payloads(args.glob, configs)
    if not payloads:
        raise SystemExit(f"no evaluation JSONs matched {args.glob}")
    rows = eval_rows(payloads)
    aggregated = aggregate_rows(rows)

    floor_payloads = load_payloads(args.floor_stop_glob, configs)
    floor_rows = floor_stop_rows(payloads, floor_payloads) if floor_payloads else []

    payload = {
        "n_cells": len(payloads),
        "rows": rows,
        "aggregate": aggregated,
        "floor_stop": floor_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", out)

    print(f"{'dataset':14s} {'loss':16s} {'variant':22s} {'epoch':>6s} {'acc':>8s} {'macroF1':>8s} {'balAcc':>8s}")
    for row in rows:
        epoch = "-" if row["epoch"] is None else str(row["epoch"])
        macro = "-" if row["macro_f1"] is None else f"{100 * row['macro_f1']:.3f}"
        bal = "-" if row["bal_acc"] is None else f"{100 * row['bal_acc']:.3f}"
        print(f"{row['dataset']:14s} {row['loss']:16s} {row['variant']:22s} {epoch:>6s} "
              f"{100 * row['acc']:.3f} {macro:>8s} {bal:>8s}")

    print("\nmean over seeds")
    print(f"{'dataset':14s} {'loss':16s} {'variant':22s} {'n':>2s} {'mean acc':>9s} {'sd':>7s}")
    for row in aggregated:
        print(f"{row['dataset']:14s} {row['loss']:16s} {row['variant']:22s} {row['n_seeds']:2d} "
              f"{100 * row['mean_acc']:9.3f} {100 * row['sd_acc']:7.3f}")

    if floor_rows:
        print("\nFloorStop arm")
        print(f"{'dataset':14s} {'loss':16s} {'seed':>4s} {'saved':>6s} {'fs acc':>8s} "
              f"{'main@same':>10s} {'delta':>8s} {'oracle':>8s} {'d oracle':>9s}")
        for row in floor_rows:
            print(f"{row['dataset']:14s} {row['loss']:16s} {row['seed']:4d} {row['epochs_saved']:6d} "
                  f"{100 * row['fs_selected_acc']:8.3f} {100 * row['main_acc_at_same_epoch']:10.3f} "
                  f"{100 * row['delta_same_epoch']:8.3f} {100 * row['main_oracle_acc']:8.3f} "
                  f"{100 * row['delta_vs_oracle']:9.3f}")
    else:
        print("\nno FloorStop arm found; T6 not produced")


if __name__ == "__main__":
    main()
