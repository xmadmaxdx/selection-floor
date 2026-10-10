"""Dense weight averaging of the top-q epochs, applied before selection.

The rule is *top-q by validation loss*, matching the specification. Selecting by
accuracy instead is not equivalent: on a plateau, accuracy is flat while loss is
still falling, so the two rules pick different checkpoints.

Integer buffers such as BatchNorm's `num_batches_tracked` are excluded from the
average and copied from the best epoch, so the averaged state dict keeps its
original dtypes. Averaging them as floats is harmless in eval mode but leaves
the checkpoint subtly ill-typed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

INTEGER_BUFFER_NAMES = frozenset({"num_batches_tracked"})


def average_state_dicts(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Weight-space average of state dicts. Raises if the list is empty."""
    if not paths:
        raise ValueError("cannot average an empty list of checkpoints")
    accumulator: dict[str, torch.Tensor] = {}
    reference: dict[str, torch.Tensor] = {}
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"missing epoch checkpoint: {path}")
        state = torch.load(path, map_location="cpu")
        if index == 0:
            reference = state
            accumulator = {k: v.clone().float() for k, v in state.items() if k not in INTEGER_BUFFER_NAMES}
        else:
            if set(state) != set(reference):
                raise ValueError(f"checkpoint {path} has a different parameter set")
            for key, value in state.items():
                if key in INTEGER_BUFFER_NAMES:
                    continue
                accumulator[key] += value.float()
    count = float(len(paths))
    averaged = {k: (v / count).to(reference[k].dtype) for k, v in accumulator.items()}
    for key, value in reference.items():
        if key in INTEGER_BUFFER_NAMES:
            averaged[key] = value.clone()
    return averaged


def top_q_epoch_paths(checkpoint_dir: str | Path, val_loss: np.ndarray, q: int) -> list[Path]:
    """Paths of the q epochs with the lowest validation loss, best first."""
    directory = Path(checkpoint_dir) / "epochs"
    if q < 1:
        raise ValueError("q must be >= 1")
    losses = np.asarray(val_loss, dtype=float)
    if losses.size < q:
        raise ValueError(f"cannot take top {q} from {losses.size} epochs")
    order = np.argsort(losses, kind="stable")[:q]
    return [directory / f"epoch_{int(e) + 1:03d}.pt" for e in order]