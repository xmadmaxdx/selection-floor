"""Dataset builders and the fit / pool / test split that the selection study needs.

The whole project rests on one split design. Training uses `fit`. Model selection
uses a subset drawn from `pool`, and `pool` is never trained on, so every replayed
validation draw measures selection noise rather than memorisation. The official
test split is never used for anything except the final reported numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
import torchvision.datasets as tvd

from src.constants import DATASET_CLASS_COUNTS

SPLIT_SEED = 12345

_TORCHVISION_MAP = {
    "mnist": ("MNIST", (0.1307,), (0.3081,), 28, 1),
    "fashionmnist": ("FashionMNIST", (0.2860,), (0.3530,), 28, 1),
    "kmnist": ("KMNIST", (0.1918,), (0.3483,), 28, 1),
    "cifar10": ("CIFAR10", (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 32, 3),
}

_CHANNELS = {28: 1, 32: 3}


def _stats_for(name: str, image_size: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if name not in _TORCHVISION_MAP:
        raise ValueError(f"unknown dataset: {name}")
    _, mean, std, size, _ = _TORCHVISION_MAP[name]
    if size != image_size:
        raise ValueError(f"{name} has native size {size}, config asked for {image_size}")
    channels = _CHANNELS[image_size]
    mean_t = mean if isinstance(mean, tuple) else (mean,)
    std_t = std if isinstance(std, tuple) else (std,)
    if len(mean_t) == 1:
        return (mean_t[0],) * channels, (std_t[0],) * channels
    return mean_t, std_t


def train_transform(image_size: int, dataset_name: str) -> T.Compose:
    mean, std = _stats_for(dataset_name, image_size)
    return T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor(), T.Normalize(mean, std)])


def eval_transform(image_size: int, dataset_name: str) -> T.Compose:
    mean, std = _stats_for(dataset_name, image_size)
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


class FakeImages(Dataset):
    """Deterministic synthetic set for offline smoke tests. No downloads."""

    def __init__(self, n: int, num_classes: int, image_size: int, transform=None, seed: int = 0) -> None:
        self.n = int(n)
        self.num_classes = int(num_classes)
        self.transform = transform
        rng = np.random.default_rng(seed)
        channels = _CHANNELS[image_size]
        self.images = rng.integers(0, 256, size=(self.n, image_size, image_size, channels), dtype=np.uint8)
        self.labels = rng.integers(0, num_classes, size=(self.n,))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        from PIL import Image

        array = self.images[index]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        image = Image.fromarray(array)
        tensor = self.transform(image) if self.transform is not None else T.ToTensor()(image)
        return tensor, int(self.labels[index])


@dataclass(frozen=True)
class SplitIndices:
    """Index triples into the non-test data plus the official test set."""

    fit: np.ndarray
    pool: np.ndarray
    test: np.ndarray


def _stratified_pool_indices(labels: np.ndarray, pool_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out `pool_fraction` of each class for the selection pool."""
    rng = np.random.default_rng(seed)
    pool_parts: list[np.ndarray] = []
    fit_parts: list[np.ndarray] = []
    for class_id in np.unique(labels):
        member = np.flatnonzero(labels == class_id)
        member = member[rng.permutation(member.size)]
        n_pool = int(round(member.size * pool_fraction))
        n_pool = min(max(n_pool, 1), member.size - 1) if member.size > 1 else 0
        pool_parts.append(member[:n_pool])
        fit_parts.append(member[n_pool:])
    pool = np.sort(np.concatenate(pool_parts)) if pool_parts else np.empty(0, dtype=int)
    fit = np.sort(np.concatenate(fit_parts)) if fit_parts else np.empty(0, dtype=int)
    return fit, pool


def build_datasets(cfg, download: bool = False):
    """Build the fit / pool / test datasets described in the module docstring."""
    name = cfg.data.dataset
    image_size = cfg.data.image_size
    if name == "fake":
        classes = DATASET_CLASS_COUNTS["fake"]
        n_fit = cfg.data.subset_fit or 400
        n_pool = max(16, int(round(n_fit * cfg.data.pool_fraction)))
        n_test = 200
        # Real indices so the split invariant is exercised by the offline test
        # suite. Returning None here previously left the disjointness test body
        # unreachable in every environment where the suite actually runs.
        return {
            "fit": FakeImages(n_fit, classes, image_size, train_transform(image_size, "mnist"), seed=0),
            "pool": FakeImages(n_pool, classes, image_size, eval_transform(image_size, "mnist"), seed=1),
            "test": FakeImages(n_test, classes, image_size, eval_transform(image_size, "mnist"), seed=2),
            "indices": SplitIndices(
                fit=np.arange(n_fit),
                pool=np.arange(n_fit, n_fit + n_pool),
                test=np.arange(n_test),
            ),
        }
    if name not in _TORCHVISION_MAP:
        raise ValueError(f"unsupported dataset: {name}")
    cls_name, _, _, native_size, _ = _TORCHVISION_MAP[name]
    if native_size != image_size:
        raise ValueError(f"{name} needs image_size {native_size}, got {image_size}")
    dataset_cls = getattr(tvd, cls_name)
    root = Path(cfg.data.root)
    root.mkdir(parents=True, exist_ok=True)
    train_full = dataset_cls(root=str(root), train=True, download=download,
                             transform=train_transform(image_size, name))
    train_eval = dataset_cls(root=str(root), train=True, download=False,
                             transform=eval_transform(image_size, name))
    test_set = dataset_cls(root=str(root), train=False, download=download,
                           transform=eval_transform(image_size, name))
    labels = np.asarray(train_full.targets)
    fit_idx, pool_idx = _stratified_pool_indices(labels, cfg.data.pool_fraction, SPLIT_SEED)
    if cfg.data.subset_fit is not None:
        fit_idx = fit_idx[: min(int(cfg.data.subset_fit), fit_idx.size)]
    return {
        "fit": Subset(train_full, fit_idx.tolist()),
        "pool": Subset(train_eval, pool_idx.tolist()),
        "test": test_set,
        "indices": SplitIndices(fit=fit_idx, pool=pool_idx, test=np.arange(len(test_set))),
    }


def build_loaders(cfg, datasets: dict) -> dict[str, DataLoader]:
    """Build deterministic loaders. The pool and test loaders never shuffle."""
    common = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "fit": DataLoader(datasets["fit"], shuffle=True, **common),
        "pool": DataLoader(datasets["pool"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }