"""Colab Drive autosave. No-op locally. Never fails a run."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER_NAME = "selection-floor"

log = logging.getLogger(LOGGER_NAME)

_MOUNT = Path("/content/drive")
_OUT = _MOUNT / "MyDrive" / "selection-floor-outputs"


def drive_mounted() -> bool:
    return _OUT.parent.is_dir()


def ensure_drive_mounted() -> bool:
    """Mount Drive on Colab if possible. Returns True when usable."""
    try:
        if drive_mounted():
            return True
        import google.colab  # noqa: F401

        from google.colab import drive

        drive.mount(str(_MOUNT), force_remount=False)
        _OUT.mkdir(parents=True, exist_ok=True)
        log.info("drive mounted at %s", _OUT)
        return True
    except Exception as exc:
        log.warning("drive not available, local-only mode: %s", exc)
        return False


def mirror_file(local: str | Path) -> Path | None:
    """Copy one file into the Drive mirror, preserving its parent folder name."""
    try:
        src = Path(local)
        if not src.is_file():
            return None
        if not drive_mounted():
            return None
        dest = _OUT / src.parent.name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest
    except Exception as exc:
        log.warning("drive mirror failed for %s: %s", local, exc)
        return None


def restore_mirror(subfolder: str, destination: str | Path) -> int:
    """Copy a Drive subfolder back into the local workspace. Returns file count."""
    src = _OUT / subfolder
    if not src.is_dir():
        return 0
    dst = Path(destination) / subfolder
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.rglob("*"):
        if item.is_file():
            target = dst / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            count += 1
    return count