"""Task 2 — Model loading (SDXL Inpainting + ControlNet Depth + DPT + Qwen3-4B)."""

import os
import threading
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from transformers import DPTImageProcessor, DPTForDepthEstimation, AutoTokenizer, AutoModelForCausalLM

_T2_DIR = Path(__file__).parent

T2_DEVICE        = os.environ.get("T2_DEVICE",        "cuda:0")
SDXL_INPAINT_ID  = os.environ.get("T2_SDXL_ID",       "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
CONTROLNET_ID    = os.environ.get("T2_CONTROLNET_ID",  "diffusers/controlnet-depth-sdxl-1.0")
DPT_ID           = os.environ.get("T2_DPT_ID",         "Intel/dpt-hybrid-midas")
QWEN_ID          = os.environ.get("T2_QWEN_ID",         "Qwen/Qwen3-4B")
LORA_PATH        = os.environ.get("T2_LORA_PATH",      str(_T2_DIR / "ckp" / "pytorch_lora_weights.safetensors"))

# ── Globals ───────────────────────────────────────────────────────────────────
pipe           = None   # StableDiffusionXLControlNetInpaintPipeline
dp             = None   # DPTImageProcessor (CPU only, no .to() needed)
dm             = None   # DPTForDepthEstimation
qwen_tokenizer = None   # AutoTokenizer
qwen_model     = None   # AutoModelForCausalLM
_loading       = False


# ── LoRA direct merge (bypass peft to avoid meta tensor issues) ──────────────

def _merge_lora_direct(unet, lora_path: str, lora_scale: float = 0.8) -> None:
    from safetensors.torch import load_file as _load_sf
    sd = _load_sf(lora_path)
    pairs: dict = {}
    for key, tensor in sd.items():
        k = key
        for prefix in ("unet.", "base_model.model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        if ".lora_A.weight" in k:
            base = k[: k.rindex(".lora_A.weight")]
            pairs.setdefault(base, {})["A"] = tensor
        elif ".lora_B.weight" in k:
            base = k[: k.rindex(".lora_B.weight")]
            pairs.setdefault(base, {})["B"] = tensor
        elif ".lora_down.weight" in k:
            base = k[: k.rindex(".lora_down.weight")]
            pairs.setdefault(base, {})["A"] = tensor
        elif ".lora_up.weight" in k:
            base = k[: k.rindex(".lora_up.weight")]
            pairs.setdefault(base, {})["B"] = tensor
        elif ".lora_A." in k and k.endswith(".weight"):
            idx = k.rindex(".lora_A.")
            base = k[:idx]
            pairs.setdefault(base, {})["A"] = tensor
        elif ".lora_B." in k and k.endswith(".weight"):
            idx = k.rindex(".lora_B.")
            base = k[:idx]
            pairs.setdefault(base, {})["B"] = tensor

    merged = 0
    for path, mats in pairs.items():
        if "A" not in mats or "B" not in mats:
            continue
        try:
            mod = unet
            for p in path.split("."):
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        except (AttributeError, IndexError, TypeError, KeyError):
            continue
        if not hasattr(mod, "weight") or mod.weight is None:
            continue
        w = mod.weight.data
        A = mats["A"].to(device=w.device, dtype=w.dtype)
        B = mats["B"].to(device=w.device, dtype=w.dtype)
        if A.dim() >= 3:
            A = A.flatten(1)
        if B.dim() >= 3:
            B = B.flatten(1)
        delta = lora_scale * (B @ A)
        if w.dim() == 2 and delta.shape == w.shape:
            w.add_(delta)
            merged += 1
        elif w.dim() == 4 and delta.shape == w.shape[:2]:
            w[:, :, 0, 0].add_(delta)
            merged += 1

    print(f"[Task2] LoRA merged: {merged}/{len(pairs)} matrices (scale={lora_scale})")


# ── Loading ───────────────────────────────────────────────────────────────────

def load_to_ram():
    """Load tất cả models vào CPU RAM rồi register với ModelManager."""
    global pipe, dp, dm, qwen_tokenizer, qwen_model, _loading

    if pipe is not None:
        return
    if _loading:
        return
    _loading = True

    try:
        print("[Task2] Loading ControlNet Depth → CPU RAM…")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_ID, torch_dtype=torch.float16, low_cpu_mem_usage=False
        )

        print("[Task2] Loading SDXL Inpainting pipeline → CPU RAM…")
        _pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            SDXL_INPAINT_ID,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=False,
        )
        # The pipeline loads on CPU by default; avoid redundant .to('cpu') calls that
        # can fail on meta tensors during initialization.
        _pipe.vae.enable_slicing()
        _pipe.set_progress_bar_config(disable=True)
        _pipe.scheduler = DDIMScheduler.from_config(_pipe.scheduler.config)

        if LORA_PATH and Path(LORA_PATH).exists():
            print(f"[Task2] Merging LoRA from {LORA_PATH}…")
            _merge_lora_direct(_pipe.unet, LORA_PATH, lora_scale=0.8)
        else:
            print(f"[Task2] No LoRA found at {LORA_PATH}, skipping.")

        print("[Task2] Loading DPT depth model → CPU RAM…")
        _dp = DPTImageProcessor.from_pretrained(DPT_ID)
        _dm = DPTForDepthEstimation.from_pretrained(
            DPT_ID, torch_dtype=torch.float16, low_cpu_mem_usage=False
        )
        _dm.eval()

        print("[Task2] Loading Qwen3-4B → CPU RAM…")
        _qwen_tok = AutoTokenizer.from_pretrained(QWEN_ID)
        _qwen_mod = AutoModelForCausalLM.from_pretrained(
            QWEN_ID, torch_dtype=torch.float16, low_cpu_mem_usage=False,
        )
        _qwen_mod.eval()

        pipe           = _pipe
        dp             = _dp
        dm             = _dm
        qwen_tokenizer = _qwen_tok
        qwen_model     = _qwen_mod
        print("[Task2] All weights in CPU RAM — ready for fast GPU transfer.")

        # Register chỉ pipe với ModelManager (dm/qwen sẽ lazy load on-demand trong inference)
        from model_manager import manager

        class _PipelineWrapper:
            """Wrapper để ModelManager có thể move Diffusers pipeline."""
            def __init__(self, pipeline):
                self._pipe = pipeline

            def to(self, device, **kwargs):
                # Diffusers pipeline dùng cú pháp khác
                self._pipe.to(torch.device(device))
                return self

        manager.register("task2", {"pipe": _PipelineWrapper(pipe)})

    except Exception as e:
        print(f"[Task2] ERROR loading pipeline: {e}")
        import traceback; traceback.print_exc()
    finally:
        _loading = False


def preload_to_cpu():
    """Gọi lúc startup trong background thread."""
    t = threading.Thread(target=load_to_ram, daemon=True, name="task2-preload")
    t.start()


def wait_until_loaded(timeout: float = 180.0):
    """Block cho đến khi load xong (dùng bởi inference nếu cần)."""
    t0 = time.time()
    while _loading:
        if time.time() - t0 > timeout:
            raise RuntimeError("[Task2] Timeout waiting for model load.")
        time.sleep(0.5)
