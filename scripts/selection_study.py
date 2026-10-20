"""Replay the model-selection step over many validation draws. No retraining.

Two distinct measurements are produced, and they answer different questions. The
first version conflated them and plotted a dispersion against a threshold.

`floor` rows test the theory. They compare the closed-form validation-gap
standard deviation `sqrt(2 p(1-p) / n_val)` against the standard deviation of
*measured* validation accuracy differences between epoch pairs on the same draw.
Those are the same physical quantity, so the ratio is a real test.

`replay` rows characterise the reported number: the winner's-curse bias against
the best test accuracy any fixed epoch achieves, plus the selection spread and
the separate test-set sampling term.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.constants import REPLAY_SEED, REPLAY_TRIALS, VALIDATION_SUBSET_SIZES
from src.engine.selector import (load_correctness, paired_rule_difference, replay_all_rules,
                                 validation_gap_dispersion)
from src.stats.inference import (gap_for_correct_ranking, minimum_detectable_effect,
                                 selection_floor_table, validation_gap_se)
from src.utils.common import setup_logging
from src.utils.drive import mirror_file
from src.viz.plots import (plot_floor_vs_measured, plot_rule_bias, plot_selection_spread)

log = logging.getLogger("selection-floor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay selection and measure the floor")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=REPLAY_TRIALS)
    parser.add_argument("--seed", type=int, default=REPLAY_SEED)
    parser.add_argument("--gap-draws", type=int, default=400,
                        help="validation draws for the closed-form test")
    parser.add_argument("--sizes", default=",".join(str(s) for s in VALIDATION_SUBSET_SIZES))
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    pool_correct, test_correct = load_correctness(args.checkpoint_dir)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    n_pool = pool_correct.shape[1]
    valid_sizes = [s for s in sizes if 0 < s < n_pool]
    if not valid_sizes:
        raise SystemExit(f"no usable validation size; pool has {n_pool} examples")

    # The accuracy that enters the floor formula is the accuracy of the models
    # being compared, i.e. the best epochs, not the mean over all epochs.
    pool_acc = pool_correct.mean(axis=1)
    compare_acc = float(pool_acc.max())
    mean_epoch_acc = float(pool_acc.mean())

    floor_rows = []
    for n_val in valid_sizes:
        rows = selection_floor_table(compare_acc, (n_val,))
        measured = validation_gap_dispersion(pool_correct, n_val, args.gap_draws, args.seed + n_val)
        theory_se = validation_gap_se(compare_acc, n_val)
        floor_rows.append({
            "n_val": n_val,
            "se_theory": theory_se,
            "se_measured": measured["observed_sd"],
            "ratio_measured_over_theory": measured["observed_sd"] / theory_se if theory_se > 0 else float("nan"),
            "se_sampling": measured["sampling_sd"],
            "se_theory_paired": measured["paired_theory_sd"],
            "ratio_paired": measured["ratio_paired"],
            "ratio_independence": measured["ratio_independence"],
            "between_pair_sd": measured["between_pair_sd"],
            "finite_population_factor": measured["finite_population_factor"],
            "abs_gap_mean": measured["observed_abs_mean"],
            "ranking_gap": rows[0]["ranking"],
            "mde": rows[0]["mde"],
            "n_pairs": measured["n_pairs"],
        })
        log.info("n_val=%d theory_se=%.5f measured_se=%.5f pooled_ratio=%.3f sampling_se=%.5f "
                 "paired_ratio=%.3f independence_ratio=%.3f",
                 n_val, theory_se, measured["observed_sd"],
                 floor_rows[-1]["ratio_measured_over_theory"], measured["sampling_sd"],
                 measured["ratio_paired"], measured["ratio_independence"])

    replay_rows = []
    for n_val in valid_sizes:
        for outcome in replay_all_rules(pool_correct, test_correct, n_val, args.trials, args.seed + n_val):
            replay_rows.append(outcome.as_dict())

    differences = [paired_rule_difference(pool_correct, test_correct, "one_se", "argmax", n_val,
                                          min(args.trials, 1000), args.seed + n_val)
                   for n_val in valid_sizes]

    payload = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "n_epochs": int(pool_correct.shape[0]),
        "n_pool": n_pool,
        "n_test": int(test_correct.shape[1]),
        "compare_acc": compare_acc,
        "mean_epoch_acc": mean_epoch_acc,
        "trials": args.trials,
        "gap_draws": args.gap_draws,
        "gap_for_correct_ranking": gap_for_correct_ranking(compare_acc, valid_sizes[0]),
        "minimum_detectable_effect": minimum_detectable_effect(compare_acc, valid_sizes[0]),
        "floor": floor_rows,
        "replay": replay_rows,
        "rule_differences": differences,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mirror_file(out)
    plot_floor_vs_measured(floor_rows, out.parent / f"{out.stem}.floor.png")
    plot_rule_bias(replay_rows, out.parent / f"{out.stem}.bias.png")
    plot_selection_spread(replay_rows, out.parent / f"{out.stem}.spread.png")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()