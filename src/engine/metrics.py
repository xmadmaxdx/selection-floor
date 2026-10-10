"""Classification metrics beyond accuracy, so the floor argument is not single-metric."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)


def summarize(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """All scalar metrics. JSON-safe floats, with NaN allowed for AUROC failures."""
    prob = torch.softmax(logits, dim=1).numpy()
    y = labels.numpy()
    pred = prob.argmax(axis=1)
    out = {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
    }
    try:
        out["auroc_ovr"] = float(roc_auc_score(y, prob, multi_class="ovr"))
    except ValueError:
        out["auroc_ovr"] = float("nan")
    return out


def confusion(logits: torch.Tensor, labels: torch.Tensor) -> list[list[int]]:
    return confusion_matrix(labels.numpy(), torch.softmax(logits, dim=1).numpy().argmax(axis=1)).tolist()