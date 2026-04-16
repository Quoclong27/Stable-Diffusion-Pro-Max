"""Task 2: Image Inpainting & Editing — thin aggregator."""

from .model import preload_to_cpu, load_to_ram, wait_until_loaded
from .inference import run_inference
from .ui import create_task2_tab

__all__ = ["create_task2_tab", "preload_to_cpu", "load_to_ram", "wait_until_loaded"]
