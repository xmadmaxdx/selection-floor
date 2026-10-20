"""Per-cell validation curves with the rule picks marked (F4).

`plot_validation_curve` existed with no caller, which is why F4 was missing.

The figure exists to make one argument visible: the three rules disagree
exactly where the validation curve is flat, and a flat curve is where the
validation sample stops carrying information. The argmax pick rides the
highest point of a plateau, the one-standard-error pick retreats to the
earliest epoch that is statistically indistinguishable from it, and the
FloorStop epoch is where no later epoch could beat the best so far by more
than one standard error. Marking all three on the same curve shows why they
land in different places, and therefore why the reported number depends on
which rule a practitioner happened to use.

The curve is the recorded pool accuracy, so the figure is regenerated from
disk without touching a model.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.engine.selector import select_argmax, select_one_se
from src.engine.trainer import floor_stop_epoch
from src.utils.common import setup_logging
from src.viz.plots import plot_validation_curve

log = logging.getLogger("selection-floor")


def curve_markers(pool_acc: list[float], n_pool: int) -> dict:
    """Where each rule picks on one validation curve.

    `n_pool` is the number of examples the curve was measured on, because the
    one-standard-error band and the FloorStop rule are both widths in standard
    errors of that measurement.
    """
    curve = np.asarray(pool_acc, dtype=float)
    if curve.size == 0:
        raise ValueError("empty validation curve")
    floor = floor_stop_epoch([float(v) for v in curve], n_pool)
    return {
        "argmax_index": select_argmax(curve, n_pool),
        "one_se_index": select_one_se(curve, n_pool),
        "floor_stop_index": None if floor is None else int(floor),
    }


def cell_curve(checkpoint_dir: Path) -> tuple[np.ndarray, int, dict]:
    """Read the recorded pool-accuracy curve and the rule picks on it."""
    history = json.loads((checkpoint_dir / "history.json").read_text(encoding="utf-8"))
    pool_acc = np.asarray(history["pool_acc"], dtype=float)
    n_pool = int(history["n_pool"])
    return pool_acc, n_pool, curve_markers(list(pool_acc), n_pool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validation curves with rule picks marked")
    parser.add_argument("--ckpt-glob", default="outputs/ckpt-*")
    parser.add_argument("--outdir", default="outputs/curves")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    outdir = Path(args.outdir)
    made = []
    for directory in sorted(Path().glob(args.ckpt_glob)):
        if not (directory / "history.json").is_file():
            log.warning("no history.json in %s; skipping", directory)
            continue
        pool_acc, n_pool, markers = cell_curve(directory)
        stem = directory.name[len("ckpt-"):] if directory.name.startswith("ckpt-") else directory.name
        path = outdir / f"{stem}.curve.png"
        plot_validation_curve(pool_acc, path,
                              one_se_index=markers["one_se_index"],
                              floor_index=markers["floor_stop_index"])
        made.append((stem, markers))
        log.info("%s: argmax=%s one_se=%s floor_stop=%s", stem, markers["argmax_index"],
                 markers["one_se_index"], markers["floor_stop_index"])
    if not made:
        raise SystemExit(f"no trained cells matched {args.ckpt_glob}")
    print(f"{'cell':44s} {'argmax':>7s} {'1-SE':>7s} {'floor stop':>11s}")
    for stem, markers in made:
        floor = "-" if markers["floor_stop_index"] is None else str(markers["floor_stop_index"] + 1)
        print(f"{stem:44s} {markers['argmax_index'] + 1:7d} {markers['one_se_index'] + 1:7d} {floor:>11s}")
    print(f"\nwrote {len(made)} curve(s) to {outdir}")


if __name__ == "__main__":
    main()
