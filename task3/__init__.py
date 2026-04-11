"""Task 3: Image Outpainting — dùng ControlNet Union + SDXL Fill.

Tham số UI được căn chỉnh khớp với inference_preserve_blend_v2.py:
  - Target resolution: SDXL bucket preset hoặc Auto detect
  - Alignment: Middle / Left / Right / Top / Bottom
  - Resize option: Full / 50% / 33% / 25% / Custom
  - Overlap mask, overlap sides
"""

import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── Model path config (override qua env vars) ───────────────────────────────
_T3_DIR = Path(__file__).parent
CONTROLNET_CONFIG  = os.environ.get("T3_CONTROLNET_CONFIG",  str(_T3_DIR / "config_promax.json"))
CONTROLNET_WEIGHTS = os.environ.get("T3_CONTROLNET_WEIGHTS", str(_T3_DIR / "diffusion_pytorch_model_promax.safetensors"))
BASE_MODEL         = os.environ.get("T3_BASE_MODEL",         "SG161222/RealVisXL_V5.0_Lightning")
VAE_MODEL          = os.environ.get("T3_VAE_MODEL",          "madebyollin/sdxl-vae-fp16-fix")
LORA_PATH          = os.environ.get("T3_LORA_PATH",          str(_T3_DIR / "lora_best"))
T3_DEVICE          = os.environ.get("T3_DEVICE",             "")   # auto-detect below
T3_DTYPE           = os.environ.get("T3_DTYPE",              "")

# SDXL bucket resolutions — must match training
SDXL_BUCKETS = {
    "1:1  — 1024×1024": (1024, 1024),
    "16:9 — 1344×768":  (1344, 768),
    "9:16 — 768×1344":  (768,  1344),
    "4:3  — 1152×896":  (1152, 896),
    "3:4  — 896×1152":  (896,  1152),
    "Customize":        (0,    0),
}

_RESIZE_OPTS = ["Full", "50%", "33%", "25%", "Custom"]
_ALIGN_OPTS  = ["Middle", "Left", "Right", "Top", "Bottom"]

# Hardcoded inference defaults (not exposed in UI)
_GUIDANCE_SCALE   = 2.5
_CONTROLNET_SCALE = 1.0
_SEED             = 1234
_BLEND_RADIUS     = 24
_COLOR_MATCH_STR  = 0.6

_pipe = None  # global pipeline handle


# ── Model loading ────────────────────────────────────────────────────────────

def load_model():
    global _pipe, T3_DEVICE, T3_DTYPE
    if _pipe is not None:
        return

    import torch

    # Auto-detect device and dtype
    if not T3_DEVICE:
        T3_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if not T3_DTYPE:
        T3_DTYPE = "fp16" if (T3_DEVICE == "cuda") else "fp32"
    print(f"[Task3] Device: {T3_DEVICE}, dtype: {T3_DTYPE}")

    # Verify required files exist before attempting full load
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

    from diffusers import AutoencoderKL, TCDScheduler
    from diffusers.models.model_loading_utils import load_state_dict
    from .controlnet_union import ControlNetModel_Union
    from .pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

    dtype = (torch.float16 if T3_DTYPE == "fp16"
             else torch.bfloat16 if T3_DTYPE == "bf16"
             else torch.float32)

    print("[Task3] Loading ControlNet Union…")
    config = ControlNetModel_Union.load_config(CONTROLNET_CONFIG)
    controlnet = ControlNetModel_Union.from_config(config)
    sd = load_state_dict(CONTROLNET_WEIGHTS)
    controlnet.load_state_dict(sd, strict=False)
    controlnet = controlnet.to(device=T3_DEVICE, dtype=dtype)

    print("[Task3] Loading VAE…")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=dtype).to(T3_DEVICE)

    print("[Task3] Loading SDXL Fill pipeline…")
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        vae=vae,
        controlnet=controlnet,
        variant="fp16" if (T3_DEVICE == "cuda" and dtype == torch.float16) else None,
    ).to(T3_DEVICE)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    if LORA_PATH and Path(LORA_PATH).exists():
        try:
            from peft import PeftModel
            print(f"[Task3] Loading LoRA from {LORA_PATH}…")
            pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
            pipe.unet.to(T3_DEVICE)
        except ImportError:
            print("[Task3] peft not installed — skipping LoRA.")

    _pipe = pipe
    print("[Task3] Pipeline ready.")


# ── Image helpers (mirrors inference_preserve_blend_v2.py) ──────────────────

def _prepare_canvas(
    image: Image.Image,
    width: int,
    height: int,
    resize_option: str,
    custom_resize_percentage: int,
    alignment: str,
    ovl_left_pct: float,
    ovl_right_pct: float,
    ovl_top_pct: float,
    ovl_bottom_pct: float,
):
    """Mirrors prepare_image_and_mask from inference_preserve_blend_v2.py.
    Per-side overlap: each side's percentage independently (0 = no overlap, use 2px gap).
    """
    target_size = (width, height)

    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    source = image.resize(
        (int(image.width * scale_factor), int(image.height * scale_factor)), Image.LANCZOS
    )

    pct_map = {"Full": 100, "50%": 50, "33%": 33, "25%": 25}
    resize_pct = pct_map.get(resize_option, custom_resize_percentage)
    resize_factor = resize_pct / 100.0
    new_width  = max(int(source.width  * resize_factor), 64)
    new_height = max(int(source.height * resize_factor), 64)
    source = source.resize((new_width, new_height), Image.LANCZOS)

    if alignment == "Middle":
        margin_x = (target_size[0] - new_width)  // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width)  // 2
        margin_y = 0
    else:  # Bottom
        margin_x = (target_size[0] - new_width)  // 2
        margin_y = target_size[1] - new_height

    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    app_mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(app_mask)

    GAP = 2

    def _px(pct, dim):
        return max(int(dim * pct / 100.0), 1) if pct > 0 else GAP

    left_ovl   = margin_x + _px(ovl_left_pct,   new_width)
    right_ovl  = margin_x + new_width  - _px(ovl_right_pct,  new_width)
    top_ovl    = margin_y + _px(ovl_top_pct,    new_height)
    bottom_ovl = margin_y + new_height - _px(ovl_bottom_pct, new_height)

    # Flush-edge correction: no overlap on the side the image is pressed against
    if alignment == "Left":
        left_ovl  = margin_x
    elif alignment == "Right":
        right_ovl = margin_x + new_width
    elif alignment == "Top":
        top_ovl   = margin_y
    elif alignment == "Bottom":
        bottom_ovl = margin_y + new_height

    left_ovl   = max(0, min(left_ovl,  target_size[0] - 1))
    right_ovl  = max(left_ovl + 1, min(right_ovl,  target_size[0]))
    top_ovl    = max(0, min(top_ovl,   target_size[1] - 1))
    bottom_ovl = max(top_ovl + 1,  min(bottom_ovl, target_size[1]))

    preserve_rect = (left_ovl, top_ovl, right_ovl, bottom_ovl)
    mask_draw.rectangle([(left_ovl, top_ovl), (right_ovl, bottom_ovl)], fill=0)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), app_mask)

    return background, app_mask, cnet_image, preserve_rect


def _make_blend_mask(size, preserve_rect, blend_radius: int) -> Image.Image:
    w, h = size
    x1, y1, x2, y2 = preserve_rect
    core = Image.new("L", (w, h), 0)
    ImageDraw.Draw(core).rectangle([(x1, y1), (x2, y2)], fill=255)
    if blend_radius > 0:
        return core.filter(ImageFilter.GaussianBlur(radius=blend_radius))
    return core


def _color_match(generated: Image.Image, background: Image.Image,
                 blend_mask: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return generated
    a = np.asarray(blend_mask).astype(np.float32) / 255.0
    band = (a > 0.05) & (a < 0.95)
    if band.sum() < 128:
        return generated
    gen = np.asarray(generated).astype(np.float32)
    bg  = np.asarray(background).astype(np.float32)
    out = gen.copy()
    for c in range(3):
        shift = (float(bg[..., c][band].mean()) - float(gen[..., c][band].mean())) * strength
        out[..., c] = np.clip(out[..., c] + shift, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


# ── Preview ──────────────────────────────────────────────────────────────────

def _resolve_res(target_res_label, custom_w, custom_h, image=None):
    """Return (width, height) for the chosen resolution option."""
    bw, bh = SDXL_BUCKETS.get(target_res_label, (0, 0))
    if bw == 0:  # Customize
        bw, bh = int(custom_w or 1024), int(custom_h or 1024)
    return bw, bh


def _preview(image, target_res_label, custom_w, custom_h,
             alignment, resize_option, custom_resize_pct,
             ovl_left_pct, ovl_right_pct, ovl_top_pct, ovl_bottom_pct):
    if image is None:
        return None
    bw, bh = _resolve_res(target_res_label, custom_w, custom_h)
    try:
        bg, app_mask, _cnet, _rect = _prepare_canvas(
            image, bw, bh, resize_option, custom_resize_pct, alignment,
            ovl_left_pct, ovl_right_pct, ovl_top_pct, ovl_bottom_pct,
        )
    except Exception:
        return image
    preview   = bg.copy().convert("RGBA")
    red_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    overlay   = Image.new("RGBA", bg.size, (255, 0, 0, 80))
    red_layer.paste(overlay, (0, 0), app_mask)
    return Image.alpha_composite(preview, red_layer)


# ── Inference ────────────────────────────────────────────────────────────────

def _infer(
    image,
    target_res_label, custom_w, custom_h,
    alignment, resize_option, custom_resize_pct,
    ovl_left_pct, ovl_right_pct, ovl_top_pct, ovl_bottom_pct,
    prompt, num_steps,
):
    import torch

    if image is None:
        raise gr.Error("Vui lòng upload ảnh trước.")
    if _pipe is None:
        raise gr.Error(
            "Pipeline Task 3 chưa được load. "
            "Kiểm tra log khởi động — có thể thiếu controlnet_union.py, "
            "pipeline_fill_sd_xl.py hoặc file weights."
        )

    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)

    bw, bh = _resolve_res(target_res_label, custom_w, custom_h)

    background, _app_mask, cnet_image, preserve_rect = _prepare_canvas(
        image, bw, bh, resize_option, custom_resize_pct, alignment,
        ovl_left_pct, ovl_right_pct, ovl_top_pct, ovl_bottom_pct,
    )
    blend_mask = _make_blend_mask(background.size, preserve_rect, _BLEND_RADIUS)

    prompt_text = f"{prompt}, high quality, 4k" if prompt else "high quality, 4k"
    (prompt_embeds, neg_pe,
     pooled_pe, neg_pooled_pe) = _pipe.encode_prompt(prompt_text, T3_DEVICE, True)

    last_image = None
    for out in _pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_pe,
        pooled_prompt_embeds=pooled_pe,
        negative_pooled_prompt_embeds=neg_pooled_pe,
        image=cnet_image,
        num_inference_steps=num_steps,
        guidance_scale=_GUIDANCE_SCALE,
        controlnet_conditioning_scale=_CONTROLNET_SCALE,
    ):
        last_image = out
        yield last_image   # stream intermediate steps

    generated_adj = _color_match(last_image, background, blend_mask, _COLOR_MATCH_STR)
    result = Image.composite(background, generated_adj, blend_mask)
    yield result


# ── UI ───────────────────────────────────────────────────────────────────────

def create_task3_tab():
    gr.Markdown(
        "**Mở rộng ảnh (Outpainting)** — Ảnh gốc sẽ được đặt lên canvas SDXL, "
        "vùng màu đỏ trong preview là nơi model tự vẽ thêm."
    )

    with gr.Row():
        # ── Left column ──────────────────────────────────────────────
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil", label="Ảnh đầu vào",
                sources=["upload", "clipboard"],
            )
            gr.Markdown(
                "<small style='color:#888'>Kéo thả ảnh, nhấn 📎 để chọn file hoặc 📋 để dán từ clipboard</small>"
            )

            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Prompt mô tả nội dung mở rộng (tuỳ chọn)", scale=2
                )
                run_button = gr.Button("Generate", variant="primary", scale=1)

            # Target resolution
            target_res = gr.Radio(
                choices=list(SDXL_BUCKETS.keys()),
                value="1:1  — 1024×1024",
                label="Kích thước đầu ra",
            )
            with gr.Row(visible=False) as custom_res_row:
                custom_w = gr.Number(value=1024, label="Width (px)",  precision=0, minimum=256, maximum=2048)
                custom_h = gr.Number(value=1024, label="Height (px)", precision=0, minimum=256, maximum=2048)

            with gr.Row():
                alignment = gr.Dropdown(
                    choices=_ALIGN_OPTS,
                    value="Middle",
                    label="Vị trí ảnh gốc trên canvas",
                    info="Middle=giữa, Left/Right/Top/Bottom=sát cạnh",
                    scale=1,
                )
                resize_option = gr.Radio(
                    choices=_RESIZE_OPTS,
                    value="Full",
                    label="Kích thước ảnh gốc",
                    scale=2,
                )

            custom_resize_pct = gr.Slider(
                10, 100, step=1, value=100,
                label="Custom resize (%)",
                visible=False,
            )

            gr.Markdown("**Overlap theo từng cạnh** — 0% = không overlap, >0% = số % của cạnh ảnh gốc")
            with gr.Row():
                ovl_top_pct    = gr.Slider(0, 50, step=1, value=10, label="↑ Trên (%)")
                ovl_bottom_pct = gr.Slider(0, 50, step=1, value=10, label="↓ Dưới (%)")
            with gr.Row():
                ovl_left_pct   = gr.Slider(0, 50, step=1, value=10, label="← Trái (%)")
                ovl_right_pct  = gr.Slider(0, 50, step=1, value=10, label="→ Phải (%)")

            with gr.Accordion("Cài đặt nâng cao", open=False):
                num_steps = gr.Slider(4, 20, step=1, value=8, label="Số bước inference")

        # ── Right column ──────────────────────────────────────────────
        with gr.Column(scale=1):
            output_tabs = gr.Tabs(selected="tab_preview")
            with output_tabs:
                with gr.Tab("Preview vùng mở rộng", id="tab_preview"):
                    preview_image = gr.Image(
                        interactive=False, container=False,
                        sources=[], show_label=False,
                    )
                with gr.Tab("Kết quả", id="tab_result"):
                    result_image = gr.Image(
                        interactive=False, container=False,
                        sources=[], show_label=False,
                    )
            use_as_input_btn = gr.Button("↩ Dùng làm ảnh đầu vào", visible=False)

    # ── Shared param lists ────────────────────────────────────────────────────
    _preview_inputs = [
        input_image, target_res, custom_w, custom_h,
        alignment, resize_option, custom_resize_pct,
        ovl_left_pct, ovl_right_pct, ovl_top_pct, ovl_bottom_pct,
    ]
    _all_inputs = _preview_inputs + [prompt_input, num_steps]

    # ── Show/hide custom resolution row ──────────────────────────────────────
    target_res.change(
        fn=lambda v: gr.update(visible=(v == "Customize")),
        inputs=target_res,
        outputs=custom_res_row,
        queue=False,
    )

    # ── Show/hide custom resize slider ────────────────────────────────────────
    resize_option.change(
        fn=lambda opt: gr.update(visible=(opt == "Custom")),
        inputs=resize_option,
        outputs=custom_resize_pct,
        queue=False,
    )

    # ── Auto-preview on any canvas param change ───────────────────────────────
    for _comp in _preview_inputs:
        _comp.change(
            fn=_preview,
            inputs=_preview_inputs,
            outputs=preview_image,
            queue=False,
        )

    # Upload ảnh → switch to preview tab
    input_image.change(
        fn=lambda: gr.update(selected="tab_preview"),
        outputs=output_tabs,
        queue=False,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    def _clear_result():
        return gr.update(value=None), gr.update(visible=False)

    def _run(*args):
        yield from _infer(*args)

    for trigger in (run_button.click, prompt_input.submit):
        trigger(
            fn=_clear_result,
            outputs=[result_image, use_as_input_btn],
        ).then(
            fn=lambda: gr.update(selected="tab_result"),
            outputs=output_tabs,
        ).then(
            fn=_run,
            inputs=_all_inputs,
            outputs=result_image,
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=use_as_input_btn,
        )

    use_as_input_btn.click(
        fn=lambda img: gr.update(value=img),
        inputs=result_image,
        outputs=input_image,
    )
