"""Evaluate the checkpoint each selection rule chooses, with full metrics.

Selection here uses the full pool as the validation signal, which is the honest
reading of "the model you would deploy". The replay study uses subsets of the
pool instead; the two answer different questions and are labelled separately.

The top-q epochs for dense averaging are chosen by *validation loss*, matching
the rule as specified, which requires `keep_checkpoints: true` and the recorded
`val_loss` series.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data.datasets import build_datasets, build_loaders
from src.engine.averaging import average_state_dicts, top_q_epoch_paths
from src.engine.metrics import summarize
from src.engine.selector import select_argmax, select_one_se
from src.engine.trainer import collect_outputs, floor_stop_epoch
from src.models.factory import build_model
from src.utils.common import get_device, set_seed, setup_logging
from src.utils.drive import ensure_drive_mounted, mirror_file

log = logging.getLogger("selection-floor")

GREYSCALE_DATASETS = {"mnist", "fashionmnist", "kmnist", "fake"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selection-rule checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-q", type=int, default=5)
    return parser.parse_args()


@torch.no_grad()
def _metrics_for(model: torch.nn.Module, loader, device: torch.device) -> dict:
    logits, labels = collect_outputs(model, loader, device)
    return summarize(logits, labels)


def _epoch_path(epoch_index: int) -> Path:
    return Path("epochs") / f"epoch_{int(epoch_index) + 1:03d}.pt"


def main() -> None:
    setup_logging()
    ensure_drive_mounted()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = get_device()
    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    if not (checkpoint_dir / "epochs").is_dir():
        raise SystemExit(f"no per-epoch checkpoints in {checkpoint_dir}; retrain with --keep-checkpoints")

    history = json.loads((checkpoint_dir / "history.json").read_text(encoding="utf-8"))
    pool_acc = np.asarray(history["pool_acc"], dtype=float)
    val_loss = np.asarray(history.get("val_loss", []), dtype=float)
    if val_loss.size != pool_acc.size:
        raise SystemExit(f"val_loss has {val_loss.size} entries but {pool_acc.size} epochs were trained")
    n_pool = int(history["n_pool"])

    datasets = build_datasets(cfg, download=args.download)
    loaders = build_loaders(cfg, datasets)
    in_channels = 1 if cfg.data.dataset in GREYSCALE_DATASETS else 3
    model = build_model(cfg.model.name, cfg.model.num_classes, in_channels).to(device)
    model.eval()

    def evaluate_epoch(index: int) -> dict:
        model.load_state_dict(torch.load(checkpoint_dir / _epoch_path(index), map_location=device))
        return _metrics_for(model, loaders["test"], device)

    variants: dict[str, dict] = {}
    argmax_epoch = select_argmax(pool_acc, n_pool)
    variants["argmax"] = {"epoch": argmax_epoch + 1, "selected_by": "full pool accuracy", **evaluate_epoch(argmax_epoch)}

    one_se_epoch = select_one_se(pool_acc, n_pool)
    if one_se_epoch != argmax_epoch:
        variants["one_se"] = {"epoch": one_se_epoch + 1, "selected_by": "1-SE band", **evaluate_epoch(one_se_epoch)}
    else:
        variants["one_se"] = {"epoch": one_se_epoch + 1, "selected_by": "1-SE band (same as argmax)",
                              **variants["argmax"]}

    last_epoch = int(pool_acc.size) - 1
    if last_epoch not in (argmax_epoch, one_se_epoch):
        variants["last"] = {"epoch": last_epoch + 1, "selected_by": "final epoch", **evaluate_epoch(last_epoch)}

    q = min(args.top_q, int(pool_acc.size))
    averaged = average_state_dicts(top_q_epoch_paths(checkpoint_dir, val_loss, q))
    model.load_state_dict({k: v.to(device) for k, v in averaged.items()})
    variants[f"dense_avg_top{q}_by_loss"] = {
        "epoch": None,
        "selected_by": f"weight-space average of top-{q} epochs by validation loss",
        **_metrics_for(model, loaders["test"], device),
    }

    stop_epoch = floor_stop_epoch([float(v) for v in pool_acc], n_pool)
    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "n_epochs": int(pool_acc.size),
        "config_digest": history.get("config_digest"),
        "seed": history.get("seed"),
        "param_count": history.get("param_count"),
        "floor_stop_epoch": None if stop_epoch is None else stop_epoch + 1,
        "variants": variants,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mirror_file(out)
    log.info("wrote %s", out)
    for name, row in variants.items():
        log.info("%-24s epoch=%s acc=%.4f macro_f1=%.4f bal_acc=%.4f",
                 name, row["epoch"], row["acc"], row["macro_f1"], row["bal_acc"])


if __name__ == "__main__":
    main()