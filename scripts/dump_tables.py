"""Summarize replay and evaluation JSONs into the tables the paper transcribes.

`--mdi-table` prints the closed-form floor table so the manuscript's headline
numbers are generated from code rather than transcribed by hand. Two rows of the
first hand-written table were not reproducible from the stated formula.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from src.stats.inference import selection_floor_table

REPORT_DATASETS = (
    ("MNIST (SimpleCNN)", 0.99, 12000),
    ("MNIST (ResNet-18)", 0.99, 12000),
    ("FashionMNIST", 0.90, 12000),
    ("CIFAR-10", 0.85, 12000),
)


def print_mdi_table() -> None:
    sizes = (250, 500, 1000, 2000, 5000)
    print(f"{'setting':22s} {'p':>6s} {'n_val':>7s} {'SE':>8s} {'ranking':>9s} {'MDE':>9s}")
    for label, accuracy, n_val in REPORT_DATASETS:
        row = selection_floor_table(accuracy, (n_val,))[0]
        print(f"{label:22s} {accuracy:6.2f} {n_val:7d} {100 * row['se']:8.3f} "
              f"{100 * row['ranking']:9.3f} {100 * row['mde']:9.3f}")
    print("\nAll values in percentage points. 'ranking' = z_power * SE, 'MDE' = (z_{1-alpha/2} + z_power) * SE.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump selection-study tables")
    parser.add_argument("--glob", default="outputs/studies/*.json")
    parser.add_argument("--floor", action="store_true", help="Print the closed-form test rows")
    parser.add_argument("--mdi-table", action="store_true", help="Print the headline floor table")
    args = parser.parse_args()
    if args.mdi_table:
        print_mdi_table()
        return
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob}")
    for path in files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        print("=" * 26, Path(path).stem)
        print(f"epochs={data['n_epochs']} pool={data['n_pool']} test={data['n_test']} "
              f"compare_acc={data['compare_acc']:.4f} trials={data['trials']}")
        if args.floor:
            print(f"{'n_val':>7s} {'se_theory':>10s} {'se_meas':>9s} {'pooled':>7s} "
                  f"{'se_samp':>9s} {'se_pair':>9s} {'paired':>7s} {'indep':>7s} {'betw_sd':>9s} {'fpc':>6s}")
            for row in data["floor"]:
                print(f"{row['n_val']:7d} {row['se_theory']:10.5f} {row['se_measured']:9.5f} "
                      f"{row['ratio_measured_over_theory']:7.3f} {row.get('se_sampling', float('nan')):9.5f} "
                      f"{row.get('se_theory_paired', float('nan')):9.5f} "
                      f"{row.get('ratio_paired', float('nan')):7.3f} "
                      f"{row.get('ratio_independence', float('nan')):7.3f} "
                      f"{row.get('between_pair_sd', float('nan')):9.5f} "
                      f"{row.get('finite_population_factor', float('nan')):6.3f}")
            continue
        print(f"{'rule':>8s} {'n_val':>6s} {'mean_sel':>9s} {'std':>8s} {'sem':>8s} "
              f"{'optim':>8s} {'regret':>8s} {'val_sel':>8s} {'oracle':>8s} {'frac_best':>9s}")
        for row in sorted(data["replay"], key=lambda r: (r["n_val"], r["rule"])):
            print(f"{row['rule']:>8s} {row['n_val']:6d} {row['mean_selected']:9.4f} "
                  f"{row['std_selected']:8.4f} {row['sem_selected']:8.5f} {row['optimism']:8.4f} "
                  f"{row['regret']:8.4f} {row['mean_val_selected']:8.4f} {row['oracle_test']:8.4f} "
                  f"{row['frac_best']:9.3f}")
        if "rule_differences" in data:
            print("\npaired rule differences (one_se - argmax), same draws:")
            for row in data["rule_differences"]:
                print(f"  n_val={row['n_val']:6d} mean={100 * row['mean_difference']:+.4f} pp "
                      f"+/- {100 * row['sem_difference']:.4f}  a_better={row['fraction_a_better']:.3f}")


if __name__ == "__main__":
    main()