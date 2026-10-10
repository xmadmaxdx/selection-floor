"""Figures for the selection study.

Every axis states its own units. The first version plotted a standard deviation
(the spread of the reported number across validation draws) on the same log-log
axis as the closed-form minimum detectable effect (a gap threshold), under the
label "resolvable gap". Those are different quantities, so the two figures are now
separate and separately labelled: one tests the theory, one characterises the
reported number.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
_MARKERS = ["o", "s", "^", "D"]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def _by_rule(rows: list[dict]) -> tuple[list[str], list[int]]:
    rules = sorted({row["rule"] for row in rows})
    sizes = sorted({row["n_val"] for row in rows})
    return rules, sizes


def _series(rows: list[dict], rule: str, size: int, key: str) -> float:
    return float(next(row[key] for row in rows if row["rule"] == rule and row["n_val"] == size))


def plot_floor_vs_measured(rows: list[dict], path: Path) -> Path:
    """Closed-form validation-gap SE against the measured one. Like for like.

    Both curves are standard deviations of the same quantity: the difference in
    validation accuracy between two checkpoints measured on the same n_val
    examples. Their ratio is the falsifiable content of the theory.
    """
    xs = [row["n_val"] for row in rows]
    theory = [100.0 * row["se_theory"] for row in rows]
    measured = [100.0 * row["se_measured"] for row in rows]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot(xs, theory, color=_COLORS[0], marker="o", markersize=4, linewidth=1,
            label=r"closed form $\sqrt{2p(1-p)/n_{val}}$")
    ax.plot(xs, measured, color=_COLORS[1], marker="s", markersize=4, linewidth=1,
            label="measured SD of val. gap")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"validation size $n_{val}$")
    ax.set_ylabel("validation-gap SD (pp)")
    ax.legend(fontsize=6, loc="best")
    return _save(fig, path)


def plot_rule_bias(rows: list[dict], path: Path) -> Path:
    """Winner's-curse optimism per rule, against validation size.

    Plotted quantity is the validation accuracy of the selected checkpoint minus
    its accuracy on the untouched test set. Positive values mean the evidence
    used to make the selection overstates what the selection delivers.
    """
    rules, sizes = _by_rule(rows)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for index, rule in enumerate(rules):
        xs = [s for s in sizes if any(r["rule"] == rule and r["n_val"] == s for r in rows)]
        ys = [100.0 * _series(rows, rule, s, "optimism") for s in xs]
        ax.plot(xs, ys, color=_COLORS[index % len(_COLORS)], marker=_MARKERS[index % len(_MARKERS)],
                markersize=4, linewidth=1, label=rule)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"validation size $n_{val}$")
    ax.set_ylabel("winner's-curse optimism (pp)")
    ax.legend(fontsize=7, loc="best")
    return _save(fig, path)


def plot_selection_spread(rows: list[dict], path: Path) -> Path:
    """Dispersion of the reported number, with the test-set sampling term shown separately."""
    rules, sizes = _by_rule(rows)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for index, rule in enumerate(rules):
        xs = [s for s in sizes if any(r["rule"] == rule and r["n_val"] == s for r in rows)]
        ys = [100.0 * _series(rows, rule, s, "std_selected") for s in xs]
        ax.plot(xs, ys, color=_COLORS[index % len(_COLORS)], marker=_MARKERS[index % len(_MARKERS)],
                markersize=4, linewidth=1, label=f"{rule} (selection)")
    first = rows[0] if rows else None
    if first is not None:
        ax.axhline(100.0 * float(first["test_sampling_sd"]), color=_COLORS[3], linestyle=":",
                   linewidth=1, label="test-set sampling")
    ax.set_xlabel(r"validation size $n_{val}$")
    ax.set_ylabel("spread of reported number (pp)")
    ax.legend(fontsize=6, loc="best")
    return _save(fig, path)


def plot_validation_curve(curve: np.ndarray, path: Path, one_se_index: int | None = None,
                          floor_index: int | None = None) -> Path:
    """Pool accuracy per epoch, with the 1-SE pick and the floor-stop epoch marked."""
    curve = np.asarray(curve, dtype=float)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    epochs = np.arange(1, curve.size + 1)
    ax.plot(epochs, 100.0 * curve, color=_COLORS[0], marker="o", markersize=3, linewidth=1,
            label="pool accuracy")
    ax.axvline(int(np.argmax(curve)) + 1, color=_COLORS[3], linestyle=":", linewidth=1, label="argmax")
    if one_se_index is not None:
        ax.axvline(int(one_se_index) + 1, color=_COLORS[2], linestyle="--", linewidth=1, label="1-SE")
    if floor_index is not None:
        ax.axvline(int(floor_index) + 1, color=_COLORS[1], linestyle="-.", linewidth=1, label="floor stop")
    ax.set_xlabel("epoch")
    ax.set_ylabel("pool accuracy (%)")
    ax.legend(fontsize=6, loc="lower right")
    return _save(fig, path)