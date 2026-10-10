"""Typed config loading with fail-closed validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.constants import DATASET_CLASS_COUNTS, POOL_FRACTION


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "mnist"
    root: str = "data"
    image_size: int = 28
    batch_size: int = 128
    num_workers: int = 2
    subset_fit: int | None = None
    pool_fraction: float = POOL_FRACTION


@dataclass(frozen=True)
class ModelConfig:
    name: str = "simplecnn"
    num_classes: int = 10
    dropout: float = 0.0


@dataclass(frozen=True)
class LossConfig:
    name: str = "ce"
    gamma: float = 2.0
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adamw"
    seed: int = 0
    keep_checkpoints: bool = False
    floor_stop: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    smoke: bool = False


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_ALLOWED_DATASETS = set(DATASET_CLASS_COUNTS)
_ALLOWED_MODELS = {"resnet18_small", "simplecnn"}
_ALLOWED_LOSSES = {"ce", "focal", "label_smoothing"}
_ALLOWED_OPTIMIZERS = {"adamw", "sgd"}
_ALLOWED_IMAGE_SIZES = {28, 32}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_config(path: str | Path, smoke_override: bool = False) -> RunConfig:
    """Load YAML config with strict validation. Raises ValueError on any violation."""
    p = Path(path)
    _require(p.is_file(), f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    data = DataConfig(**raw.get("data", {}))
    model = ModelConfig(**raw.get("model", {}))
    loss = LossConfig(**raw.get("loss", {}))
    train = TrainConfig(**raw.get("train", {}))
    _require(data.dataset in _ALLOWED_DATASETS, f"bad data.dataset: {data.dataset}")
    _require(model.name in _ALLOWED_MODELS, f"bad model.name: {model.name}")
    _require(loss.name in _ALLOWED_LOSSES, f"bad loss.name: {loss.name}")
    _require(train.optimizer in _ALLOWED_OPTIMIZERS, f"bad optimizer: {train.optimizer}")
    _require(model.num_classes == DATASET_CLASS_COUNTS[data.dataset], "model.num_classes must match dataset")
    _require(data.image_size in _ALLOWED_IMAGE_SIZES, "data.image_size must be 28 or 32")
    _require(data.batch_size >= 1, "batch_size must be >= 1")
    _require(0.0 < data.pool_fraction < 1.0, "pool_fraction must be in (0, 1)")
    _require(train.epochs >= 1, "epochs must be >= 1")
    _require(train.lr > 0, "lr must be > 0")
    _require(train.weight_decay >= 0, "weight_decay must be >= 0")
    _require(0.0 <= loss.label_smoothing < 1.0, "label_smoothing in [0,1)")
    _require(loss.gamma >= 0, "loss.gamma must be >= 0")
    _require(model.dropout == 0.0, "model.dropout unsupported, must be 0.0")
    _require(data.subset_fit is None or data.subset_fit >= 1, "subset_fit must be >= 1")
    _require(loss.name == "label_smoothing" or loss.label_smoothing == 0.0, "label_smoothing only for its own loss")
    if smoke_override:
        train = TrainConfig(**{**train.__dict__, "smoke": True, "epochs": 1,
                               "keep_checkpoints": False, "floor_stop": False})
        data = DataConfig(**{**data.__dict__, "dataset": "fake", "subset_fit": 200})
    return RunConfig(data=data, model=model, loss=loss, train=train)