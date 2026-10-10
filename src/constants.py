"""Shared named constants. No magic numbers anywhere else in the code."""
from __future__ import annotations

IMAGE_SIZE = 28
CIFAR_IMAGE_SIZE = 32

# Fraction of the non-test data held out as the selection pool.
# The pool is never trained on; replay draws validation subsets from it.
POOL_FRACTION = 0.2

# Simulation settings for selection replay.
REPLAY_TRIALS = 2000
REPLAY_SEED = 0

# Validation draws used when testing the closed form against measurement.
GAP_DRAWS = 400

# Significance and power for the floor formulas. Both are swept in the paper's
# sensitivity table rather than being fixed here and bypassed at the call site.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.95

# One-standard-error rule multiplier, following Hastie, Tibshirani and Friedman.
ONE_SE_MULTIPLIER = 1.0

# Number of top epochs averaged in weight space before selection.
DENSE_AVERAGE_TOP_Q = 5

# Numeric guards.
LOG_EPS = 1e-8

VALIDATION_SUBSET_SIZES = (250, 500, 1000, 2000)

DATASET_CLASS_COUNTS = {
    "mnist": 10,
    "fashionmnist": 10,
    "kmnist": 10,
    "cifar10": 10,
    "fake": 10,
}