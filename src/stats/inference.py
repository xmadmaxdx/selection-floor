"""Inference primitives behind the selection floor.

Design note. The first version of this module shipped a function called
`minimum_detectable_improvement` that used `z_{1-alpha/2}` alone. That constant
corresponds to neither standard estimand: it is not the one-sided gap needed to
rank two models correctly with probability `power` (which needs `z_power`), and
it is not the conventional two-sided minimum detectable effect (which needs
`z_{1-alpha/2} + z_power`). Both estimands are now implemented explicitly and
named for what they compute. The docstrings state which one is which, because
the ambiguity was the defect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.95

_NORMAL_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_NORMAL_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
_NORMAL_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_NORMAL_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF via Acklam's rational approximation.

    Verified against scipy.stats.norm.ppf to 6+ decimals in `tests/test_smoke.py`.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_NORMAL_C[0] * q + _NORMAL_C[1]) * q + _NORMAL_C[2]) * q + _NORMAL_C[3]) * q + _NORMAL_C[4]) * q + _NORMAL_C[5]) / \
               ((((_NORMAL_D[0] * q + _NORMAL_D[1]) * q + _NORMAL_D[2]) * q + _NORMAL_D[3]) * q + 1.0)
    if p > high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_NORMAL_C[0] * q + _NORMAL_C[1]) * q + _NORMAL_C[2]) * q + _NORMAL_C[3]) * q + _NORMAL_C[4]) * q + _NORMAL_C[5]) / \
                ((((_NORMAL_D[0] * q + _NORMAL_D[1]) * q + _NORMAL_D[2]) * q + _NORMAL_D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((_NORMAL_A[0] * r + _NORMAL_A[1]) * r + _NORMAL_A[2]) * r + _NORMAL_A[3]) * r + _NORMAL_A[4]) * r + _NORMAL_A[5]) * q / \
           (((((_NORMAL_B[0] * r + _NORMAL_B[1]) * r + _NORMAL_B[2]) * r + _NORMAL_B[3]) * r + _NORMAL_B[4]) * r + 1.0)


def standard_error(p: float, n: int) -> float:
    """Binomial standard error of an accuracy estimate."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if n < 1:
        raise ValueError("n must be >= 1")
    return math.sqrt(p * (1.0 - p) / n)


def paired_gap_se(acc_a: float, acc_b: float, n: int, agreement: float) -> float:
    """Standard error of the paired accuracy difference between two models.

    `agreement` is P(both models are correct on the same example), i.e. the
    number of shared successes, NOT a covariance. For Bernoulli indicators
    X ~ Ber(a), Y ~ Ber(b) with P(X=1, Y=1) = c, the covariance is `c - a*b`,
    and Var(X - Y) = a(1-a) + b(1-b) - 2(c - a*b). An earlier version assigned
    `c` directly to a variable named `covariance`, dropped the `+2ab` term, and
    clamped negative variances to zero -- which silently returned exactly 0.0
    for almost every realistic input.
    """
    for value, name in ((acc_a, "acc_a"), (acc_b, "acc_b"), (agreement, "agreement")):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if n < 1:
        raise ValueError("n must be >= 1")
    lower = max(0.0, acc_a + acc_b - 1.0)
    upper = min(acc_a, acc_b)
    if not lower - 1e-9 <= agreement <= upper + 1e-9:
        raise ValueError(
            f"agreement {agreement} violates the Frechet bounds [{lower}, {upper}] "
            f"for accuracies ({acc_a}, {acc_b}); refusing to return a clamped variance")
    covariance = agreement - acc_a * acc_b
    variance = acc_a * (1.0 - acc_a) + acc_b * (1.0 - acc_b) - 2.0 * covariance
    if variance < -1e-12:
        raise ValueError(f"negative paired variance {variance}; agreement is inconsistent")
    return math.sqrt(max(0.0, variance) / n)


def validation_gap_se(acc: float, n_val: int) -> float:
    """SE of the accuracy *difference* between two independent checkpoints.

    This is the quantity the closed-form floor is built on. Under the null of
    equal accuracy, the difference of two accuracies measured on the SAME
    n_val examples has variance 2 p(1-p) / n_val, because the shared test-set
    variance cancels in the difference only if the models are independent, and
    independence is what the null assumes.
    """
    return math.sqrt(2.0 * acc * (1.0 - acc) / n_val)


def gap_for_correct_ranking(acc: float, n_val: int, confidence: float = DEFAULT_POWER) -> float:
    """One-sided gap needed to rank two checkpoints correctly with probability `confidence`.

    P(correct ranking) = Phi(Delta / SE), so Delta = z_confidence * SE.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return _normal_quantile(confidence) * validation_gap_se(acc, n_val)


def minimum_detectable_effect(acc: float, n_val: int, alpha: float = DEFAULT_ALPHA,
                              power: float = DEFAULT_POWER) -> float:
    """Conventional two-sided minimum detectable effect.

    Delta = (z_{1-alpha/2} + z_{power}) * SE, the standard sample-size-inverting
    quantity. This is the headline number the paper reports because it is the
    one a statistics-literate reader will check, and because it is the
    conservative of the two available readings.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    return (_normal_quantile(1.0 - alpha / 2.0) + _normal_quantile(power)) * validation_gap_se(acc, n_val)


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of an exact paired comparison between two correctness vectors."""

    b: int
    c: int
    p_value: float
    significant: bool

    def as_dict(self) -> dict:
        return {"b": self.b, "c": self.c, "p_value": self.p_value, "significant": self.significant}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts b and c.

    Integer arithmetic throughout. An earlier version divided by `2.0 ** n`,
    which raises OverflowError once b + c >= 1024 -- reachable at n_val = 500
    with strongly disagreeing checkpoints, i.e. exactly the regime this project
    studies.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Exact integer sums throughout. `2.0 * total` converts the big integer to a
    # float first and raises OverflowError once n grows; integer true division
    # is correctly rounded and never overflows as long as the quotient is small.
    total = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(1.0, (2 * total) / (1 << n)))


def paired_test(correct_a: np.ndarray, correct_b: np.ndarray, alpha: float = DEFAULT_ALPHA) -> McNemarResult:
    """Exact McNemar test between two paired correctness vectors.

    This is where the paired test belongs. An earlier version used McNemar as a
    *selection rule* over epochs; that rule provably returned `argmax` for every
    input, because the guard required a challenger to beat the incumbent on
    point accuracy while the incumbent was already the argmax. A paired test is
    an inference tool, not a ranking rule, and is used here only to decide
    whether an observed difference is reportable.
    """
    if correct_a.shape != correct_b.shape:
        raise ValueError("correctness vectors must have the same shape")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    b = int(np.count_nonzero((~correct_a) & correct_b))
    c = int(np.count_nonzero(correct_a & (~correct_b)))
    p_value = mcnemar_exact(b, c)
    return McNemarResult(b=b, c=c, p_value=p_value, significant=p_value < alpha)


def selection_floor_table(acc: float, n_vals: tuple[int, ...], alpha: float = DEFAULT_ALPHA,
                          power: float = DEFAULT_POWER) -> list[dict]:
    """Floor values across validation sizes, generated from the functions above.

    The gap-analysis and README tables are produced by this function rather
    than written by hand. Two hand-computed rows in the first draft were not
    reproducible from the stated formula; generating them removes the class of
    error permanently.
    """
    return [{
        "n_val": int(n),
        "se": validation_gap_se(acc, int(n)),
        "ranking": gap_for_correct_ranking(acc, int(n), power),
        "mde": minimum_detectable_effect(acc, int(n), alpha, power),
    } for n in n_vals]