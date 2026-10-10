"""Ensure repo root is importable for plain `pytest` (no -m) on Colab."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))