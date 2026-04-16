"""Task 3 — Inference helpers (canvas prep, blending, infer).

Must match training exactly (Long/outpainting/src/train.py →
prepare_image_and_mask_like_app).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

import gradio as gr

from . import model as M

# ── Constants ─────────────────────────────────────────────────────────────────
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

_GUIDANCE_SCALE   = 2.5      # match CLI inference_preserve_blend_v2.py
_CONTROLNET_SCALE = 1.0
_SEED             = 1234
_BLEND_RADIUS     = 24
_COLOR_MATCH_STR  = 0.6


def _resolve_res(target_res_label, custom_w, custom_h):
    bw, bh = SDXL_BUCKETS.get(target_res_label, (0, 0))
    if bw == 0:  # Customize
        bw, bh = int(custom_w or 1024), int(custom_h or 1024)
    return bw, bh


def can_expand(source_width, source_height, target_width, target_height, alignment):
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


def _prepare_canvas(
    image: Image.Image,
    width: int,
    height: int,
    overlap_percentage: float,
    resize_option: str,
    custom_resize_percentage: int,
    alignment: str,
    overlap_left: bool,
    overlap_right: bool,
    overlap_top: bool,
    overlap_bottom: bool,
):
    """Prepare canvas, mask, and cnet_image — matches training exactly."""
    target_size = (width, height)

    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    source = image.resize((new_width, new_height), Image.LANCZOS)

    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:
        resize_percentage = custom_resize_percentage

    resize_factor = resize_percentage / 100.0
    new_width = max(int(source.width * resize_factor), 64)
    new_height = max(int(source.height * resize_factor), 64)
    source = source.resize((new_width, new_height), Image.LANCZOS)

    overlap_x = max(int(new_width * (overlap_percentage / 100.0)), 1)
    overlap_y = max(int(new_height * (overlap_percentage / 100.0)), 1)

    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    else:  # Bottom
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch

    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

    left_overlap = max(0, min(left_overlap, target_size[0] - 1))
    right_overlap = max(left_overlap + 1, min(right_overlap, target_size[0]))
    top_overlap = max(0, min(top_overlap, target_size[1] - 1))
    bottom_overlap = max(top_overlap + 1, min(bottom_overlap, target_size[1]))

    preserve_rect = (left_overlap, top_overlap, right_overlap, bottom_overlap)
    mask_draw.rectangle([(left_overlap, top_overlap), (right_overlap, bottom_overlap)], fill=0)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    return background, mask, cnet_image, preserve_rect


def _make_blend_mask(size, preserve_rect, blend_radius: int) -> Image.Image:
    w, h    = size
    x1, y1, x2, y2 = preserve_rect
    core    = Image.new("L", (w, h), 0)
    ImageDraw.Draw(core).rectangle([(x1, y1), (x2, y2)], fill=255)
    if blend_radius > 0:
        return core.filter(ImageFilter.GaussianBlur(radius=blend_radius))
    return core


def _color_match(generated: Image.Image, background: Image.Image,
                 blend_mask: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return generated
    a    = np.asarray(blend_mask).astype(np.float32) / 255.0
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


def _sharpen_generated(
    generated: Image.Image,
    blend_mask: Image.Image,
    strength: float = 1.0,
) -> Image.Image:
    """Sharpen only the generated area to compensate for VAE decode softness.

    blend_mask: 255 = preserve (no sharpen), 0 = generated (full sharpen).
    strength: 0.0 = off, 1.0 = normal, 2.0 = strong.
    """
    if strength <= 0:
        return generated
    sharpened = generated.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=int(80 * strength), threshold=2)
    )
    gen_area_mask = Image.eval(blend_mask, lambda x: 255 - x)
    gen_area_mask = gen_area_mask.filter(ImageFilter.GaussianBlur(radius=3))
    return Image.composite(sharpened, generated, gen_area_mask)


def preview(image, target_res_label, custom_w, custom_h,
            alignment, resize_option, custom_resize_pct,
            overlap_percentage, overlap_left, overlap_right,
            overlap_top, overlap_bottom):
    if image is None:
        return None
    bw, bh = _resolve_res(target_res_label, custom_w, custom_h)
    try:
        bg, mask, _cnet, _rect = _prepare_canvas(
            image, bw, bh, overlap_percentage,
            resize_option, custom_resize_pct, alignment,
            overlap_left, overlap_right, overlap_top, overlap_bottom,
        )
    except Exception:
        return image
    vis       = bg.copy().convert("RGBA")
    red_layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    overlay   = Image.new("RGBA", bg.size, (255, 0, 0, 80))
    red_layer.paste(overlay, (0, 0), mask)
    return Image.alpha_composite(vis, red_layer)


def infer(
    image,
    target_res_label, custom_w, custom_h,
    alignment, resize_option, custom_resize_pct,
    overlap_percentage, overlap_left, overlap_right,
    overlap_top, overlap_bottom,
    prompt, num_steps, sharpen_strength, lora_scale,
):
    if image is None:
        raise gr.Error("Vui lòng upload ảnh trước.")

    from model_manager import manager

    if M._pipe is None:
        if M._loading:
            import time
            print("[Task3] Waiting for background preload…")
            M.wait_until_loaded()
        else:
            M._load_to_ram()

    if M._pipe is None:
        raise gr.Error(
            "Pipeline Task 3 chưa được load. "
            "Kiểm tra log khởi động — có thể thiếu controlnet_union.py, "
            "pipeline_fill_sd_xl.py hoặc file weights."
        )

    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)

    bw, bh = _resolve_res(target_res_label, custom_w, custom_h)

    if not can_expand(image.width, image.height, bw, bh, alignment):
        alignment = "Middle"

    background, _mask, cnet_image, preserve_rect = _prepare_canvas(
        image, bw, bh, overlap_percentage,
        resize_option, custom_resize_pct, alignment,
        overlap_left, overlap_right, overlap_top, overlap_bottom,
    )
    blend_mask = _make_blend_mask(background.size, preserve_rect, _BLEND_RADIUS)

    prompt_text = f"{prompt}, high quality, 4k" if prompt else "high quality, 4k"

    device = M.T3_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

    manager.activate("task3")
    manager.start_inference("task3")
    try:
        M.set_lora_scale(float(lora_scale))

        with torch.no_grad():
            (prompt_embeds, neg_pe,
             pooled_pe, neg_pooled_pe) = M._pipe.encode_prompt(prompt_text, device, True)

        last_image = None
        try:
            for out in M._pipe(
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
                yield cnet_image, out
        finally:
            del prompt_embeds, neg_pe, pooled_pe, neg_pooled_pe

        clean_base = last_image.copy()                   
        pr = preserve_rect
        clean_base.paste(                                 
            background.crop(pr), (pr[0], pr[1])
        )

        generated_adj = _color_match(last_image, clean_base, blend_mask, _COLOR_MATCH_STR)
        generated_adj = _sharpen_generated(generated_adj, blend_mask, float(sharpen_strength))
        result = Image.composite(clean_base, generated_adj, blend_mask)
        del last_image, blend_mask, clean_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        yield cnet_image, result
    finally:
        manager.end_inference()
