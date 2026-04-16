"""Task 3 — Model loading + GPU management."""

import os
import threading
from pathlib import Path

import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict

from model_manager import manager

# ── Config ────────────────────────────────────────────────────────────────────
_T3_DIR = Path(__file__).parent

CONTROLNET_CONFIG  = os.environ.get("T3_CONTROLNET_CONFIG",  str(_T3_DIR / "config_promax.json"))
CONTROLNET_WEIGHTS = os.environ.get("T3_CONTROLNET_WEIGHTS", str(_T3_DIR / "diffusion_pytorch_model_promax.safetensors"))
BASE_MODEL         = os.environ.get("T3_BASE_MODEL",         "SG161222/RealVisXL_V5.0_Lightning")
VAE_MODEL          = os.environ.get("T3_VAE_MODEL",          "madebyollin/sdxl-vae-fp16-fix")
LORA_PATH          = os.environ.get("T3_LORA_PATH",          str(_T3_DIR / "lora_best"))
T3_DEVICE          = os.environ.get("T3_DEVICE",             "")   # auto-detected below
T3_DTYPE           = os.environ.get("T3_DTYPE",              "")

# ── Globals ───────────────────────────────────────────────────────────────────
_pipe    = None          # StableDiffusionXLFillPipeline on CPU RAM after load
_loading = False         # guard against concurrent loads


def _load_to_ram():
    """Load all weights from disk into CPU RAM (no VRAM used)."""
    global _pipe, _loading, T3_DEVICE, T3_DTYPE

    if _pipe is not None:
        return
    if _loading:
        return
    _loading = True

    try:
        if not T3_DEVICE:
            T3_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if not T3_DTYPE:
            T3_DTYPE = "fp16" if (T3_DEVICE == "cuda") else "fp32"

        missing = []
        if not Path(CONTROLNET_WEIGHTS).exists():
            missing.append(f"ControlNet weights: {CONTROLNET_WEIGHTS}")
        for mod_file in ("controlnet_union.py", "pipeline_fill_sd_xl.py"):
            if not (_T3_DIR / mod_file).exists():
                missing.append(mod_file)
        if missing:
            print("[Task3] Skipping model load — missing files:")
            for m in missing:
                print(f"  - {m}")
            return

        from .controlnet_union import ControlNetModel_Union
        from .pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

        dtype = (torch.float16 if T3_DTYPE == "fp16"
                 else torch.bfloat16 if T3_DTYPE == "bf16"
                 else torch.float32)

        print("[Task3] Loading ControlNet Union → CPU RAM…")
        config     = ControlNetModel_Union.load_config(CONTROLNET_CONFIG)
        controlnet = ControlNetModel_Union.from_config(config)
        sd         = load_state_dict(CONTROLNET_WEIGHTS)
        incompatible = controlnet.load_state_dict(sd, strict=False, assign=True)
        if incompatible.missing_keys:
            print(f"[Task3] ControlNet missing keys ({len(incompatible.missing_keys)}): "
                  f"{incompatible.missing_keys[:5]}")
        if incompatible.unexpected_keys:
            print(f"[Task3] ControlNet unexpected keys ({len(incompatible.unexpected_keys)}): "
                  f"{incompatible.unexpected_keys[:5]}")
        controlnet = controlnet.to(dtype=dtype)

        print("[Task3] Loading VAE → CPU RAM…")
        vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=dtype)

        print("[Task3] Loading SDXL Fill pipeline → CPU RAM…")
        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            vae=vae,
            controlnet=controlnet,
            variant="fp16" if (dtype == torch.float16) else None,
        )
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

        if LORA_PATH and Path(LORA_PATH).exists():
            try:
                from peft import PeftModel
                print(f"[Task3] Loading LoRA from {LORA_PATH}…")
                pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
                pipe.unet.to(dtype=dtype)
                print(f"[Task3] LoRA loaded, cast to {dtype} (matches CLI).")
            except ImportError:
                print("[Task3] peft not installed — skipping LoRA.")

        _pipe = pipe
        print("[Task3] Pipeline in CPU RAM — registering with ModelManager…")
        manager.register("task3", {"pipe": _pipe})
        print("[Task3] Registered with ModelManager.")
    except Exception as e:
        print(f"[Task3] ERROR loading pipeline: {e}")
        import traceback; traceback.print_exc()
    finally:
        _loading = False


def preload_to_cpu():
    """Start background preload into CPU RAM at startup."""
    t = threading.Thread(target=_load_to_ram, daemon=True, name="task3-preload")
    t.start()


def wait_until_loaded(timeout: float = 180.0):
    """Block until pipeline is in CPU RAM (or timeout)."""
    import time
    deadline = time.time() + timeout
    while _pipe is None and time.time() < deadline:
        time.sleep(0.5)
    return _pipe is not None


def set_lora_scale(scale: float):
    """Set LoRA adapter scaling.  0.0 = base model only, 1.0 = full LoRA."""
    if _pipe is None:
        return
    try:
        _pipe.unet.set_adapters(["default"], weights=[scale])
        print(f"[Task3] LoRA scale set to {scale}")
    except Exception as e:
        print(f"[Task3] Could not set LoRA scale: {e}")
