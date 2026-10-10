"""Selection rules and the replay engine that quantifies the selection floor.

`replay_select` is the piece the literature does not have. Given the per-epoch,
per-example correctness matrix recorded during one training run, it samples
validation subsets, applies a selection rule to each, and scores the selected
checkpoint on an untouched test set. Thousands of trials cost seconds and no
retraining, so the bias and spread of the *reported number* become directly
measurable rather than guessed.

Two things changed after the first referee pass, and both are load-bearing.

1. The `paired` selection rule is gone. It was proved to return `argmax` for
   every input: it tested each challenger against the incumbent and then
   required the challenger to beat the incumbent on point accuracy, which is
   impossible when the incumbent is already the accuracy argmax. The McNemar
   statistic it computed was discarded. A paired test is an inference tool, not
   a ranking rule; it now lives in `stats.inference.paired_test` and is used to
   decide what may be *reported*, not what to *select*.

2. `bias` now measures the winner's curse against an independent reference: the
   best accuracy achievable on the untouched test set. The previous version
   subtracted a per-trial maximum taken over the same held-out examples used for
   scoring, which is non-positive by construction and is a selection regret, not
   a bias. Regret is still reported, under its own name.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.constants import ONE_SE_MULTIPLIER
from src.stats.inference import standard_error

log = logging.getLogger("selection-floor")

SELECTION_RULES = ("argmax", "one_se")


def load_correctness(checkpoint_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the recorded pool and test correctness matrices, both (n_epochs, n_examples)."""
    directory = Path(checkpoint_dir)
    pool_path = directory / "pool_correctness.npy"
    test_path = directory / "test_correctness.npy"
    if not pool_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"missing correctness arrays in {directory}; train first")
    pool = np.load(pool_path)
    test = np.load(test_path)
    for name, matrix in (("pool", pool), ("test", test)):
        if matrix.ndim != 2:
            raise ValueError(f"{name} correctness must be 2D (n_epochs, n_examples); got {matrix.shape}")
        if matrix.shape[0] != pool.shape[0]:
            raise ValueError(f"{name} has {matrix.shape[0]} epochs, pool has {pool.shape[0]}")
    return pool, test


def stack_correctness(directories: list[str | Path]) -> np.ndarray:
    """Row-stack correctness matrices from several runs into one candidate pool.

    Selection across seeds, losses, or configurations is an argmax over the union
    of all their epochs, which is the decision a practitioner actually makes.
    All runs must share the same pool split, since columns index examples; the
    caller is responsible for passing runs produced with the same config.
    """
    if not directories:
        raise ValueError("no checkpoint directories given")
    rows: list[np.ndarray] = []
    expected: int | None = None
    for directory in directories:
        pool, _ = load_correctness(directory)
        if expected is None:
            expected = pool.shape[1]
        elif pool.shape[1] != expected:
            raise ValueError(
                f"{directory} has {pool.shape[1]} pool examples but earlier runs have {expected}; "
                "runs must share the same pool split for their columns to be comparable")
        rows.append(pool)
    return np.concatenate(rows, axis=0)


def select_argmax(val_acc: np.ndarray, n_val: int) -> int:
    """Baseline: earliest epoch attaining the best validation accuracy."""
    best = float(val_acc.max())
    return int(np.flatnonzero(val_acc >= best - 1e-12)[0])


def select_one_se(val_acc: np.ndarray, n_val: int) -> int:
    """Earliest epoch within one standard error of the best.

    Hastie, Tibshirani and Friedman's one-standard-error rule (ESL 7.10), applied
    to the epoch curve. It spends a little expected accuracy to buy a large
    reduction in selection variance, which is the right trade when the curve's
    differences sit under the noise floor. Verified correct by hand against the
    reference text; this is the one selection rule that was fully implemented as
    specified on the first pass.
    """
    best = float(val_acc.max())
    se = standard_error(best, n_val)
    return int(np.flatnonzero(val_acc >= best - ONE_SE_MULTIPLIER * se)[0])


def select_with(rule: str, epoch_correct: np.ndarray, n_val: int) -> int:
    if rule == "argmax":
        return select_argmax(epoch_correct.mean(axis=1), n_val)
    if rule == "one_se":
        return select_one_se(epoch_correct.mean(axis=1), n_val)
    raise ValueError(f"unknown rule: {rule}; available: {SELECTION_RULES}")


@dataclass(frozen=True)
class ReplayResult:
    """Aggregate outcome of replaying one selection rule across many draws.

    `optimism` is the winner's curse and is the paper's headline quantity: the
    mean difference between the validation accuracy of the checkpoint a rule
    selects and that checkpoint's accuracy on the untouched test set. Because
    the selected checkpoint is chosen for scoring well on the validation draw,
    this difference is positive by construction. It is the amount by which the
    evidence used to make the decision overstates what the decision delivers.

    `regret` is a different and also reported quantity: the mean shortfall of
    the selected checkpoint against the best checkpoint on the same test set. It
    is non-positive by construction, because the selection can only ever pick
    from the candidate set, and it answers "how far from optimal was the pick",
    not "how optimistic was the justification". An earlier version reported this
    as `bias` and drew a bias-versus-validation-size figure from it; every
    value was negative and the figure implied the opposite of the paper's claim.
    """

    rule: str
    n_val: int
    mean_selected: float
    std_selected: float
    sem_selected: float
    optimism: float
    regret: float
    mean_val_selected: float
    oracle_test: float
    test_sampling_sd: float
    frac_best: float

    def as_dict(self) -> dict:
        return {
            "rule": self.rule, "n_val": self.n_val,
            "mean_selected": self.mean_selected, "std_selected": self.std_selected,
            "sem_selected": self.sem_selected, "optimism": self.optimism,
            "regret": self.regret, "mean_val_selected": self.mean_val_selected,
            "oracle_test": self.oracle_test, "test_sampling_sd": self.test_sampling_sd,
            "frac_best": self.frac_best,
        }


def _test_sampling_sd(test_correct: np.ndarray) -> float:
    """Binomial SD of a test-set accuracy: the second, separate noise source."""
    acc = float(test_correct.mean())
    return float(np.sqrt(acc * (1.0 - acc) / test_correct.shape[1]))


def replay_select(pool_correct: np.ndarray, test_correct: np.ndarray, rule: str,
                  n_val: int, n_trials: int, seed: int = 0) -> ReplayResult:
    """Replay one selection rule over many validation draws.

    Each trial draws a validation subset of the pool, selects an epoch using only
    that subset, then scores the selected epoch twice: on the untouched test set
    (for the winner's-curse bias) and on the held-out pool remainder (for regret).
    """
    if pool_correct.ndim != 2 or test_correct.ndim != 2:
        raise ValueError("correctness matrices must be 2D")
    if pool_correct.shape[1] == 0:
        raise ValueError("pool is empty")
    if pool_correct.shape[0] != test_correct.shape[0]:
        raise ValueError("pool and test must record the same number of epochs")
    n_epochs, n_pool = pool_correct.shape
    if n_val >= n_pool:
        raise ValueError(f"n_val ({n_val}) must be smaller than the pool ({n_pool})")
    rng = np.random.default_rng(seed)
    selected_test = np.empty(n_trials, dtype=float)
    selected_val = np.empty(n_trials, dtype=float)
    chosen_epochs = np.empty(n_trials, dtype=int)
    pool_cols = np.arange(n_pool)
    for trial in range(n_trials):
        val_cols = rng.permutation(pool_cols)[:n_val]
        chosen = select_with(rule, pool_correct[:, val_cols], n_val)
        chosen_epochs[trial] = chosen
        selected_test[trial] = test_correct[chosen].mean()
        selected_val[trial] = pool_correct[chosen, val_cols].mean()
    oracle_test = float(test_correct.mean(axis=1).max())
    pool_mean = pool_correct.mean(axis=1)
    best_epochs = np.flatnonzero(pool_mean >= pool_mean.max() - 1e-12)
    regret = float(selected_test.mean() - oracle_test)
    return ReplayResult(
        rule=rule,
        n_val=int(n_val),
        mean_selected=float(selected_test.mean()),
        std_selected=float(selected_test.std(ddof=1)) if n_trials > 1 else 0.0,
        sem_selected=float(selected_test.std(ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
        optimism=float((selected_val - selected_test).mean()),
        regret=regret,
        mean_val_selected=float(selected_val.mean()),
        oracle_test=oracle_test,
        test_sampling_sd=_test_sampling_sd(test_correct),
        frac_best=float(np.count_nonzero(np.isin(chosen_epochs, best_epochs)) / n_trials),
    )


def replay_all_rules(pool_correct: np.ndarray, test_correct: np.ndarray, n_val: int,
                     n_trials: int, seed: int = 0) -> list[ReplayResult]:
    """Replay every selection rule at one validation size, on a SHARED random stream.

    The previous version advanced the seed per rule, so the rules were scored on
    disjoint draws and differences between them mixed the rule effect with the
    draw effect. Every rule now sees the identical sequence of validation subsets,
    which makes the comparison paired.
    """
    return [replay_select(pool_correct, test_correct, rule, n_val, n_trials, seed)
            for rule in SELECTION_RULES]


def paired_rule_difference(pool_correct: np.ndarray, test_correct: np.ndarray, rule_a: str,
                           rule_b: str, n_val: int, n_trials: int, seed: int = 0) -> dict:
    """Compare two rules within each draw, so trial-to-trial noise cancels.

    For every validation draw both rules select a checkpoint and the difference
    of their selected test accuracies is recorded. Reporting the mean of those
    paired differences is the only honest way to claim one rule beats another;
    comparing two independently-seeded means is not.
    """
    rng = np.random.default_rng(seed)
    n_epochs, n_pool = pool_correct.shape
    selected_a = np.empty(n_trials, dtype=float)
    selected_b = np.empty(n_trials, dtype=float)
    pool_cols = np.arange(n_pool)
    for trial in range(n_trials):
        val_cols = rng.permutation(pool_cols)[:n_val]
        subset = pool_correct[:, val_cols]
        selected_a[trial] = test_correct[select_with(rule_a, subset, n_val)].mean()
        selected_b[trial] = test_correct[select_with(rule_b, subset, n_val)].mean()
    delta = selected_a - selected_b
    sem = float(delta.std(ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0
    return {
        "rule_a": rule_a,
        "rule_b": rule_b,
        "n_val": int(n_val),
        "trials": int(n_trials),
        "mean_difference": float(delta.mean()),
        "sem_difference": sem,
        "fraction_a_better": float(np.count_nonzero(delta > 0) / n_trials),
        "n_draws_a_better": int(np.count_nonzero(delta > 0)),
        "n_draws_b_better": int(np.count_nonzero(delta < 0)),
        "n_draws_tied": int(np.count_nonzero(delta == 0)),
    }


def validation_gap_dispersion(pool_correct: np.ndarray, n_val: int, n_draws: int,
                              seed: int = 0, max_pairs: int = 64) -> dict:
    """Measure the quantity the closed-form floor actually describes.

    The closed form `sqrt(2 p(1-p) / n_val)` is the standard deviation, under the
    null of equal true accuracy, of the *validation accuracy difference* between
    two checkpoints measured on the same n_val examples. This samples that
    difference directly, so the theory can be tested against a like-for-like
    measurement.

    An earlier version plotted the standard deviation of the *selected test
    accuracy* against the closed form. Those are different physical quantities --
    one is a dispersion over epochs of a fixed test set, the other a gap
    threshold -- and the comparison was meaningless.
    """
    n_epochs, n_pool = pool_correct.shape
    if n_epochs < 2:
        raise ValueError("need at least two epochs to measure a gap")
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    step = max(1, n_epochs // max_pairs)
    for i in range(0, n_epochs - 1, step):
        for j in range(i + 1, min(i + 1 + step, n_epochs)):
            pairs.append((i, j))
    if not pairs:
        pairs = [(i, i + 1) for i in range(n_epochs - 1)]
    pool_cols = np.arange(n_pool)
    differences: list[float] = []
    for _ in range(n_draws):
        val_cols = rng.permutation(pool_cols)[:n_val]
        subset = pool_correct[:, val_cols].mean(axis=1)
        for i, j in pairs:
            differences.append(float(subset[i] - subset[j]))
    values = np.asarray(differences, dtype=float)
    return {
        "n_val": int(n_val),
        "n_pairs": len(pairs),
        "n_draws": int(n_draws),
        "observed_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "observed_abs_mean": float(np.abs(values).mean()),
        "pool_acc": float(pool_correct.mean()),
    }