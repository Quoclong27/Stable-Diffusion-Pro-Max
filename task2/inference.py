"""Task 2 — Inference helpers: depth, mask, inpaint, enhance prompt."""

import re
import time

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from . import model as M


MAX_T2_LONG = 3840


# ── Prompt enhancement (Qwen3-4B) ────────────────────────────────────────────

def enhance_prompt(raw_prompt: str, max_clip_tokens: int = 77) -> str:
    if M.qwen_tokenizer is None or M.qwen_model is None:
        return raw_prompt
    system = (
        "You are an expert Stable Diffusion prompt writer for interior design inpainting. "
        "As an interior design expert, please help me enhance the simple descriptions "
        "of an object into a detailed, vivid presentation of that object. "
        "Given a description, rewrite it into a detailed, high-quality prompt. "
        "Rules:\n1. Output ONLY the enhanced prompt\n2. Under 60 words\n"
        "3. Add: material, lighting, style, quality tags\n4. Comma-separated\n5. No quotes\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Enhance this inpainting prompt: {raw_prompt}"},
    ]
    text = M.qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # Move chỉ Qwen lên GPU, chạy, rồi move về CPU ngay
    M.qwen_model.to("cuda:0")
    try:
        inputs = M.qwen_tokenizer(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = M.qwen_model.generate(
                **inputs, max_new_tokens=120, temperature=0.7,
                top_p=0.9, do_sample=True,
            )
        enhanced = M.qwen_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    finally:
        M.qwen_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    enhanced = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", enhanced, flags=re.DOTALL).strip()
    enhanced = re.sub(r"<\s*think\s*>.*", "", enhanced, flags=re.DOTALL).strip()
    enhanced = re.sub(r"<[^>]+>", "", enhanced).strip().strip('"\'')
    if not enhanced:
        return raw_prompt
    clip_ids = M.pipe.tokenizer.encode(enhanced)
    if len(clip_ids) > max_clip_tokens:
        enhanced = M.pipe.tokenizer.decode(clip_ids[:max_clip_tokens], skip_special_tokens=True)
    print(f"[Task2] Prompt enhanced: {enhanced}")
    return enhanced


# ── Depth estimation ──────────────────────────────────────────────────────────

def estimate_depth(pil_image: Image.Image) -> Image.Image:
    W, H = pil_image.size
    M.dm.to("cuda:0")
    try:
        with torch.no_grad():
            inputs = M.dp(images=pil_image.convert("RGB"), return_tensors="pt")
            inputs = {k: v.to("cuda:0", torch.float16) for k, v in inputs.items()}
            pred = M.dm(**inputs).predicted_depth
        dt = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze()
        dn = dt.cpu().numpy()
    finally:
        M.dm.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dn = (dn - dn.min()) / (dn.max() - dn.min() + 1e-8) * 255.0
    return Image.fromarray(np.stack([dn.astype(np.uint8)] * 3, axis=-1))


# ── Mask helpers ──────────────────────────────────────────────────────────────

def regularize_mask(mask_image, target_size, roundness_threshold=0.75):
    mask_np = np.array(mask_image.convert("L").resize(target_size, Image.NEAREST))
    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_image.convert("L").resize(target_size, Image.NEAREST)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area / perimeter**2) if perimeter > 0 else 0
    canvas = np.zeros_like(binary)
    if circularity >= roundness_threshold and len(contour) >= 5:
        (cx, cy), (ma, mb), angle = cv2.fitEllipse(contour)
        if min(ma, mb) / max(ma, mb) > 0.90:
            cv2.circle(canvas, (int(cx), int(cy)), int(max(ma, mb) / 2), 255, -1)
        else:
            cv2.ellipse(canvas, (int(cx), int(cy)), (int(ma / 2), int(mb / 2)),
                        angle, 0, 360, 255, -1)
    else:
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
        cv2.fillPoly(canvas, [box], 255)
    return Image.fromarray(canvas)


def extract_mask(editor_val) -> Image.Image | None:
    if editor_val is None:
        return None
    layers = editor_val.get("layers") or []
    if not layers or layers[0] is None:
        return None
    layer = layers[0]
    arr = np.array(layer)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        mask_arr = np.where(alpha > 10, 255, 0).astype(np.uint8)
        return Image.fromarray(mask_arr, mode="L")
    return layer if isinstance(layer, Image.Image) else Image.fromarray(arr)


def cap_image(img: Image.Image, max_long: int = MAX_T2_LONG) -> Image.Image:
    if img is None:
        return img
    if max(img.width, img.height) > max_long:
        s = max_long / max(img.width, img.height)
        return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return img


# ── Blending helpers ──────────────────────────────────────────────────────────

def _alpha_blend(src, dst, mask, feather=5):
    mask_f = cv2.GaussianBlur(mask.astype(float), (feather * 2 + 1, feather * 2 + 1), 0) / 255.0
    mask_f = mask_f[:, :, np.newaxis]
    return (src.astype(float) * mask_f + dst.astype(float) * (1 - mask_f)).astype(np.uint8)


def _safe_poisson_blend(src, dst, mask, margin=3):
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    H, W = mask.shape
    if (mask_eroded[0].max() > 0 or mask_eroded[-1].max() > 0 or
            mask_eroded[:, 0].max() > 0 or mask_eroded[:, -1].max() > 0):
        return _alpha_blend(src, dst, mask)
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _alpha_blend(src, dst, mask)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    center = (x + w // 2, y + h // 2)
    try:
        return cv2.seamlessClone(src, dst, mask_eroded, center, cv2.NORMAL_CLONE)
    except Exception:
        return _alpha_blend(src, dst, mask)


# ── Core inpaint ──────────────────────────────────────────────────────────────

def inpaint(init_image, mask_image, depth_map, prompt, negative_prompt, steps,
            strength=1.0, guidance_scale=12.0, cn_scale=0.3):
    W, H = init_image.size
    mask_np_hard = np.array(mask_image.convert("L"))
    ys, xs = np.where(mask_np_hard > 127)
    if len(xs) == 0:
        raise ValueError("Mask trống")

    mx1, mx2 = int(xs.min()), int(xs.max())
    my1, my2 = int(ys.min()), int(ys.max())
    mask_w, mask_h = mx2 - mx1, my2 - my1
    cx, cy = (mx1 + mx2) / 2, (my1 + my2) / 2

    inpaint_size = 1024
    target_mask_fraction = 0.5
    min_mask_px = 400
    max_context_ratio = 0.85

    target_mask_px = max(min_mask_px, inpaint_size * target_mask_fraction)
    scale = target_mask_px / max(mask_w, mask_h)
    crop_size = inpaint_size / scale
    max_crop = min(W, H) * max_context_ratio
    if crop_size > max_crop:
        crop_size = max_crop
        scale = inpaint_size / crop_size

    half = crop_size / 2
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(W, cx + half))
    y2 = int(min(H, cy + half))
    if x2 - x1 < crop_size:
        if x1 == 0: x2 = min(W, int(crop_size))
        else: x1 = max(0, int(x2 - crop_size))
    if y2 - y1 < crop_size:
        if y1 == 0: y2 = min(H, int(crop_size))
        else: y1 = max(0, int(y2 - crop_size))

    crop_w, crop_h = x2 - x1, y2 - y1
    inp_w = (int(crop_w * scale) // 64) * 64
    inp_h = (int(crop_h * scale) // 64) * 64

    crop_img   = init_image.convert("RGB").crop((x1, y1, x2, y2))
    crop_mask  = mask_image.convert("L").crop((x1, y1, x2, y2))
    crop_depth = depth_map.convert("RGB").crop((x1, y1, x2, y2))

    inp_img   = crop_img.resize((inp_w, inp_h), Image.LANCZOS)
    inp_mask  = crop_mask.resize((inp_w, inp_h), Image.NEAREST)
    inp_depth = crop_depth.resize((inp_w, inp_h), Image.LANCZOS)

    context_strip_w = 256
    context_thumb       = init_image.resize((context_strip_w, inp_h), Image.LANCZOS)
    context_depth_thumb = depth_map.resize((context_strip_w, inp_h), Image.LANCZOS)

    combined_w = inp_w + context_strip_w
    combined_inp = Image.new("RGB", (combined_w, inp_h))
    combined_inp.paste(context_thumb, (0, 0))
    combined_inp.paste(inp_img, (context_strip_w, 0))

    combined_mask = Image.new("L", (combined_w, inp_h), 0)
    combined_mask.paste(inp_mask, (context_strip_w, 0))

    combined_depth = Image.new("RGB", (combined_w, inp_h))
    combined_depth.paste(context_depth_thumb, (0, 0))
    combined_depth.paste(inp_depth, (context_strip_w, 0))

    print(f"  Crop: {crop_w}x{crop_h} | Inpaint: {inp_w}x{inp_h} | Combined: {combined_w}x{inp_h}")

    result_combined = M.pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=combined_inp,
        mask_image=combined_mask,
        control_image=combined_depth,
        height=inp_h,
        width=combined_w,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(cn_scale),
    ).images[0]

    result_small = result_combined.crop((context_strip_w, 0, combined_w, inp_h))
    result_crop  = result_small.resize((crop_w, crop_h), Image.LANCZOS)
    output = init_image.convert("RGB").copy()
    output.paste(result_crop, (x1, y1), mask=crop_mask.resize((crop_w, crop_h), Image.NEAREST))
    return output


def _poisson_blend(result_raw, init_image, mask_pil, label=""):
    dst = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
    src = cv2.cvtColor(np.array(result_raw), cv2.COLOR_RGB2BGR)
    mask_blend = np.array(mask_pil.convert("L"))
    contours, _ = cv2.findContours(mask_blend, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        center = (x + w // 2, y + h // 2)
        try:
            blended_cv = cv2.seamlessClone(src, dst, mask_blend, center, cv2.NORMAL_CLONE)
            return Image.fromarray(cv2.cvtColor(blended_cv, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"[Task2] {label}Poisson blend failed: {e}")
    return result_raw


# ── Public inference entry point ─────────────────────────────────────────────

def run_inference(editor_val, prompt, step, task_type,
                  strength=1.0, guidance_scale=12.0, cn_scale=0.3):
    if editor_val is None:
        raise gr.Error("Vui lòng upload ảnh trước.")
    bg = editor_val.get("background")
    if bg is None:
        raise gr.Error("Vui lòng upload ảnh trước.")
    input_img = cap_image(bg.convert("RGB"))
    if max(input_img.width, input_img.height) > MAX_T2_LONG:
        raise gr.Error(f"Ảnh quá lớn — tối đa {MAX_T2_LONG}px cạnh dài.")
    mask_pil = extract_mask(editor_val)
    if mask_pil is None or np.array(mask_pil).max() == 0:
        raise gr.Error("Vui lòng vẽ mask lên ảnh trước.")

    from model_manager import manager

    if M.pipe is None:
        gr.Info("Đang load model SDXL lên GPU…")
        M.load_to_ram()

    manager.activate("task2")
    manager.start_inference("task2")
    try:
        t0 = time.time()

        if task_type == "Delete":
            prompt = "clean empty background, seamless surface, natural lighting"
            neg_prompt = (
                "object, furniture, item, person, blurry, smudge, artifacts, "
                "distorted texture, mismatched pattern, text, watermark, low quality, "
                "messy, floating debris, ghosting"
            )
            strength = 0.95
            guidance_scale = 8.0
            cn_scale = 0.0
            step = 50
        else:
            if task_type == "Add" and not prompt.strip():
                prompt = "a beautiful interior design object, high quality"
            neg_prompt = (
                "additional objects, extra items, unwanted objects, "
                "multiple objects, cluttered, busy background, "
                "unrelated decoration, extra furniture, "
                "low quality, blurry, distorted, deformed, artifacts, "
                "3d render, CGI, plastic texture, flat shading, "
                "floating, wrong perspective, impossible geometry, "
                "cartoon, anime, painting, sketch, "
                "text, watermark, logo"
            )

        W, H = input_img.size
        if (mask_pil.width, mask_pil.height) != (W, H):
            mask_pil = mask_pil.resize((W, H), Image.NEAREST)
        mask_pil = regularize_mask(mask_pil, target_size=(W, H))
        depth_map = estimate_depth(input_img)

        result_raw = inpaint(
            input_img, mask_pil, depth_map,
            prompt=prompt, negative_prompt=neg_prompt,
            steps=int(step), strength=strength,
            guidance_scale=guidance_scale, cn_scale=cn_scale,
        )
        del depth_map  # free depth map tensor / PIL image before blending
        result = _poisson_blend(result_raw, input_img, mask_pil)
        del result_raw

        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
    finally:
        manager.end_inference()

    info = (f"Task: {task_type} | Steps: {int(step)} | Strength: {strength} | "
            f"CFG: {guidance_scale} | CN: {cn_scale} | Time: {elapsed:.1f}s")
    return result, info
