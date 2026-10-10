"""Loss functions used as *candidates* for checkpoint selection.

The study is not about losses. Three losses are included because that is what a
practitioner realistically selects between, so the candidate set has the shape of
a real experiment rather than an artificial one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import LOG_EPS


class FocalLoss(nn.Module):
    """Focal loss with stable numerics."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1).clamp(min=-30.0)
        ce = F.nll_loss(logp, target, reduction="none")
        pt = (-ce).exp().clamp(min=LOG_EPS, max=1.0)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def build_loss(cfg) -> nn.Module:
    """Loss factory with a typed contract. Raises ValueError on an unknown name."""
    name = cfg.loss.name
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=cfg.loss.gamma)
    raise ValueError(f"unknown loss: {name}")


def candidate_loss_names() -> tuple[str, ...]:
    """The candidate set a practitioner would actually choose between."""
    return ("ce", "label_smoothing", "focal")