"""Regenerate figures from saved study JSONs. No data, no GPU, no training.

The oracle used for the spread figure is taken from the stored JSON, so a figure
regenerated from disk is identical to the one produced during the study. The
first version recomputed the oracle here and disagreed with `selection_study.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.viz.plots import plot_floor_vs_measured, plot_rule_bias, plot_selection_spread


def main() -> None:
    parser = argparse.ArgumentParser(description="Regen figures from study JSONs")
    parser.add_argument("--json", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    path = Path(args.json)
    data = json.loads(path.read_text(encoding="utf-8"))
    outdir = Path(args.outdir) if args.outdir else path.parent
    stem = path.stem
    made = [
        plot_floor_vs_measured(data["floor"], outdir / f"{stem}.floor.png"),
        plot_rule_bias(data["replay"], outdir / f"{stem}.bias.png"),
        plot_selection_spread(data["replay"], outdir / f"{stem}.spread.png"),
    ]
    for item in made:
        print(f"wrote {item} + {item.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()