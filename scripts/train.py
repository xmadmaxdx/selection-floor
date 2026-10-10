"""Train one configuration and record everything the selection replay needs."""
from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.data.datasets import build_datasets, build_loaders
from src.engine.trainer import train_model
from src.losses.build import build_loss
from src.models.factory import build_model, count_parameters
from src.utils.common import get_device, set_seed, setup_logging
from src.utils.drive import ensure_drive_mounted

log = logging.getLogger("selection-floor")

GREYSCALE_DATASETS = {"mnist", "fashionmnist", "kmnist", "fake"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and record per-epoch correctness")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--smoke", action="store_true", help="Force a 1-epoch synthetic run")
    parser.add_argument("--download", action="store_true", help="Allow dataset download (Colab only)")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    ensure_drive_mounted()
    args = parse_args()
    cfg = load_config(args.config, smoke_override=args.smoke)
    set_seed(cfg.train.seed)
    device = get_device()
    log.info("device=%s config=%s dataset=%s model=%s loss=%s",
             device, args.config, cfg.data.dataset, cfg.model.name, cfg.loss.name)
    datasets = build_datasets(cfg, download=args.download)
    loaders = build_loaders(cfg, datasets)
    in_channels = 1 if cfg.data.dataset in GREYSCALE_DATASETS else 3
    model = build_model(cfg.model.name, cfg.model.num_classes, in_channels).to(device)
    params = count_parameters(model)
    log.info("parameters=%d", params)
    history = train_model(cfg, model, build_loss(cfg), loaders, device, param_count=params)
    if history.get("already_complete"):
        log.info("already complete, nothing to do")
        return
    log.info("done: %d epochs, final pool acc=%.4f", history["epochs_done"], history["pool_acc"][-1])


if __name__ == "__main__":
    main()