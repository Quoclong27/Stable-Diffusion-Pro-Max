"""Task 1 — Multi-Exposure Fusion using UFRetinex-MEF-ToneNet.

__init__ chỉ tập hợp public API; logic nằm trong:
  model.py      — Task1Model class + loading
  inference.py  — SIFT alignment + run()
  ui.py         — create_task1_tab()
"""

import threading

from model_manager import manager
from .model import model
from .inference import run  # noqa: F401
from .ui import create_task1_tab  # noqa: F401

__all__ = [
    "create_task1_tab",
    "preload_to_cpu",
    "run",
]


def preload_to_cpu():
    """Load Task 1 weights vào CPU RAM (background thread), rồi register với ModelManager."""
    def _do():
        if not model.loaded:
            model.load(target_device="cpu")
        if model.loaded:
            manager.register("task1", {"model": model})

    t = threading.Thread(target=_do, daemon=True, name="task1-preload")
    t.start()


# ─────────────────────────────────────────────────────────────────────────────
# Legacy / compat — phần còn lại của file gốc (patches, helpers, class) đã
# được chuyển sang model.py và inference.py. Giữ lại để không break import
# nếu có code bên ngoài dùng trực tiếp.
# ─────────────────────────────────────────────────────────────────────────────

