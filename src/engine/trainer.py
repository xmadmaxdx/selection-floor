"""Training loop that records everything the selection study needs.

The key outputs are the per-epoch correctness matrices `pool_correctness` and
`test_correctness`, of shape (n_epochs, n_examples). Those matrices contain the
entire accuracy-based, single-run selection experiment, so the replay never
retrains anything.

Recorded per epoch, alongside correctness:
  * pool loss      -- required by the dense-averaging rule, which selects the
                      top-q epochs by validation loss. The first version never
                      stored it, making that rule unimplementable after the fact.
  * train loss     -- for the training curve.
  * parameter count, config digest, seed -- so every number is attributable.

FloorStop is implemented here as an optional early exit: training halts once the
best-so-far validation improvement falls below one standard error, because from
that point the validation curve carries no resolvable signal.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.stats.inference import standard_error
from src.utils.common import cleanup_cuda
from src.utils.drive import mirror_file

log = logging.getLogger("selection-floor")

_RECORDED_CONFIGS = ("data", "model", "loss", "train")


def _make_optimizer(cfg, model: torch.nn.Module):
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)


@torch.no_grad()
def correctness_matrix(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Per-example correctness as a boolean vector, in loader order."""
    model.eval()
    flags: list[np.ndarray] = []
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        flags.append((pred == y).numpy())
    if not flags:
        raise ValueError("loader produced no batches")
    return np.concatenate(flags).astype(bool)


@torch.no_grad()
def pool_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mean cross-entropy on the selection pool, used to rank epochs for averaging."""
    model.eval()
    total, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += float(torch.nn.functional.cross_entropy(model(x), y, reduction="sum").item())
        seen += x.size(0)
    if seen == 0:
        raise ValueError("pool loader produced no batches")
    return total / seen


@torch.no_grad()
def collect_outputs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Full logits and labels, used for metrics beyond correctness."""
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def floor_stop_epoch(pool_acc: list[float], n_pool: int) -> int | None:
    """Earliest epoch from which no later epoch could beat the best so far.

    A later epoch is worth training for only if it might exceed the current best
    by more than one standard error. So the scan returns the first index after
    which every remaining epoch sits inside the band around the best-so-far, and
    returns None while the curve is still resolving.

    The previous version tested whether every *past* value sat below the band,
    which for a curve that climbs and then flattens never fires, because the
    rising prefix is never inside the band of the final maximum.
    """
    if len(pool_acc) < 2:
        return None
    for index in range(len(pool_acc) - 1):
        best_so_far = max(pool_acc[:index + 1])
        best_later = max(pool_acc[index + 1:])
        if best_later <= best_so_far + standard_error(best_so_far, n_pool):
            return index
    return None


def _config_digest(cfg) -> str:
    payload = json.dumps({name: asdict(getattr(cfg, name)) for name in _RECORDED_CONFIGS}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _persist_arrays(ckpt_dir: Path, pool: list[np.ndarray], test: list[np.ndarray]) -> None:
    for name, rows in (("pool_correctness", pool), ("test_correctness", test)):
        if not rows:
            continue
        path = ckpt_dir / f"{name}.npy"
        np.save(path, np.stack(rows).astype(bool))
        mirror_file(path)


def train_model(cfg, model: torch.nn.Module, loss_fn, loaders: dict, device: torch.device,
                param_count: int | None = None) -> dict:
    """Train, recording per-epoch correctness matrices, losses, and provenance.

    Checkpoint selection is deliberately not done here. Selection is replayed
    offline from the recorded matrices, which is the whole point of the study.
    """
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "history.json"
    pool_path = ckpt_dir / "pool_correctness.npy"
    test_path = ckpt_dir / "test_correctness.npy"
    last_path = ckpt_dir / "last.pt"

    if history_path.is_file() and pool_path.is_file() and test_path.is_file():
        existing = json.loads(history_path.read_text(encoding="utf-8"))
        if int(existing.get("epochs_done", 0)) >= cfg.train.epochs:
            log.info("already complete (%s), preserving history", ckpt_dir)
            existing["already_complete"] = True
            return existing

    optimizer = _make_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.train.epochs // 3), gamma=0.5)

    start_epoch = 0
    train_loss: list[float] = []
    val_loss: list[float] = []
    pool_rows: list[np.ndarray] = []
    test_rows: list[np.ndarray] = []
    epoch_seconds: list[float] = []

    if last_path.is_file() and history_path.is_file() and pool_path.is_file() and test_path.is_file():
        state = None
        try:
            state = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["opt"])
            scheduler.load_state_dict(state["sched"])
            start_epoch = int(state["epoch"])
            train_loss = [float(v) for v in state["train_loss"]]
            val_loss = [float(v) for v in state.get("val_loss", [])]
            epoch_seconds = [float(v) for v in state.get("epoch_seconds", [])]
            pool_rows = [row.astype(bool) for row in np.load(pool_path)]
            test_rows = [row.astype(bool) for row in np.load(test_path)]
            if not (len(pool_rows) == len(test_rows) == len(train_loss) == len(val_loss)):
                raise ValueError("recorded arrays and history disagree in length")
            # Restore RNG so a resumed run follows the same trajectory.
            if "rng_state" in state:
                torch.set_rng_state(state["rng_state"].cpu().to(torch.uint8))
            log.info("resumed from %s at epoch %d", last_path, start_epoch)
        except Exception as exc:
            # A failed resume silently restarting would discard hours of Colab
            # compute with no trace, so it is logged loudly and re-raised.
            log.error("resume from %s failed (%s); refusing to silently restart", last_path, exc)
            raise

    def _history_payload(stopped: bool) -> dict:
        """Current training state, written after every epoch.

        `history.json` used to be written only once the final epoch finished,
        but the resume gate above requires it to exist. An interrupted cell
        therefore restarted from epoch 0 and a disconnect cost a whole cell.
        Writing it every epoch makes the recorded state and the resumable
        state the same file.
        """
        return {
            "train_loss": list(train_loss),
            "val_loss": list(val_loss),
            "pool_acc": [float(row.mean()) for row in pool_rows],
            "test_acc": [float(row.mean()) for row in test_rows],
            "epoch_seconds": list(epoch_seconds),
            "epochs_done": len(train_loss),
            "n_pool": int(pool_rows[0].size),
            "n_test": int(test_rows[0].size),
            "stopped_early": stopped,
            "param_count": int(param_count) if param_count is not None else None,
            "config_digest": _config_digest(cfg),
            "seed": int(cfg.train.seed),
            "device": str(device),
            "already_complete": False,
        }

    def _write_history(stopped: bool) -> dict:
        history = _history_payload(stopped)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        mirror_file(history_path)
        return history

    stopped_early = False
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            epoch_start = time.perf_counter()
            model.train()
            running, seen = 0.0, 0
            for x, y in loaders["fit"]:
                if x.size(0) == 0:
                    raise RuntimeError("fit loader yielded an empty batch")
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * x.size(0)
                seen += x.size(0)
            scheduler.step()
            train_loss.append(running / seen)
            pool_rows.append(correctness_matrix(model, loaders["pool"], device))
            test_rows.append(correctness_matrix(model, loaders["test"], device))
            val_loss.append(pool_loss(model, loaders["pool"], device))
            epoch_seconds.append(time.perf_counter() - epoch_start)
            log.info("epoch=%d loss=%.4f pool_loss=%.4f pool_acc=%.4f test_acc=%.4f (%.1fs)",
                     epoch + 1, train_loss[-1], val_loss[-1], float(pool_rows[-1].mean()),
                     float(test_rows[-1].mean()), epoch_seconds[-1])

            if cfg.train.keep_checkpoints:
                epoch_dir = ckpt_dir / "epochs"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), epoch_dir / f"epoch_{epoch + 1:03d}.pt")

            _persist_arrays(ckpt_dir, pool_rows, test_rows)
            torch.save({
                "model": model.state_dict(), "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(), "epoch": epoch + 1,
                "train_loss": train_loss, "val_loss": val_loss,
                "epoch_seconds": epoch_seconds, "rng_state": torch.get_rng_state(),
            }, last_path)
            mirror_file(last_path)
            _write_history(stopped_early)

            if cfg.train.floor_stop:
                stop_at = floor_stop_epoch([float(row.mean()) for row in pool_rows], pool_rows[0].size)
                if stop_at is not None and stop_at < len(pool_rows):
                    log.info("floor stop: validation curve unresolvable from epoch %d onward", stop_at + 1)
                    stopped_early = True
                    break
    finally:
        cleanup_cuda()

    if not pool_rows:
        raise RuntimeError("training produced no epochs")
    return _write_history(stopped_early)