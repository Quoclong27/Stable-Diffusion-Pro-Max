"""Task 3: Image Outpainting — thin aggregator."""

from .model import preload_to_cpu, wait_until_loaded, set_lora_scale
from .inference import infer, preview
from .ui import create_task3_tab

__all__ = ["create_task3_tab", "preload_to_cpu", "wait_until_loaded"]
