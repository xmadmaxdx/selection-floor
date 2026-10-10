"""Determinism, device, and logging helpers."""
from __future__ import annotations

import logging
import random

import numpy as np
import torch

LOGGER_NAME = "selection-floor"


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger(LOGGER_NAME)


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()