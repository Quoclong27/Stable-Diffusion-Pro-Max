#!/usr/bin/env python3
"""Inference script for ControlNet Union + LoRA outpainting model.

Matches training data flow exactly:
  - control_image = background with black (0) in generation area (cnet_image)
  - ControlNet Union slot 6 (inpaint/fill)
  - SDXL bucket resolutions for auto-resolution
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# Must match training buckets exactly
SDXL_BUCKETS: Dict[str, Tuple[int, int]] = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (1024, 1024),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "21:9": (1536, 640),
    "9:21": (640, 1536),
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def best_bucket_for_ratio(width: int, height: int) -> Tuple[str, int, int]:
    ratio = width / max(height, 1)
    best_name = "1:1"
    best_w, best_h = SDXL_BUCKETS[best_name]
    best_diff = float("inf")
    for name, (bw, bh) in SDXL_BUCKETS.items():
        diff = abs((bw / bh) - ratio)
        if diff < best_diff:
            best_diff = diff
            best_name = name
            best_w, best_h = bw, bh
    return best_name, best_w, best_h


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_controlnet_union(config_path: str, weights_path: str, dtype: torch.dtype, device: str):
    config = ControlNetModel_Union.load_config(config_path)
    controlnet_model = ControlNetModel_Union.from_config(config)
    state_dict = load_state_dict(weights_path)
    controlnet_model.load_state_dict(state_dict, strict=False)
    return controlnet_model.to(device=device, dtype=dtype)


def prepare_image_and_mask(
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
) -> Tuple[Image.Image, Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """Exactly matches training prepare_image_and_mask_like_app.

    Returns:
        background: RGB canvas with source pasted
        app_mask: L mask (255=generate, 0=preserve)
        cnet_image: RGB with preserved pixels, black in gen area
        preserve_rect: (x1, y1, x2, y2) of preserved area
    """
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
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height
    else:
        raise ValueError(f"Unknown alignment: {alignment}")

    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    app_mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(app_mask)

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

    # cnet_image: must match training exactly.
    # Training uses app_mask to black out both overlap AND outer canvas.
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), app_mask)

    return background, app_mask, cnet_image, preserve_rect


def make_blend_mask(size, preserve_rect, blend_radius: int) -> Image.Image:
    """Sharp preserve core -> Gaussian-blurred alpha for smooth blending.
    Returns L image: 255 = keep original, 0 = use generated.
    """
    w, h = size
    x1, y1, x2, y2 = preserve_rect
    core = Image.new("L", (w, h), 0)
    ImageDraw.Draw(core).rectangle([(x1, y1), (x2, y2)], fill=255)
    if blend_radius > 0:
        return core.filter(ImageFilter.GaussianBlur(radius=blend_radius))
    return core


def color_match_on_seam(
    generated: Image.Image,
    background: Image.Image,
    blend_mask: Image.Image,
    strength: float = 0.6,
) -> Image.Image:
    """Shift generated image's color to match background in the seam band."""
    if strength <= 0:
        return generated
    a = np.asarray(blend_mask).astype(np.float32) / 255.0
    band = (a > 0.05) & (a < 0.95)
    if band.sum() < 128:
        return generated

    gen = np.asarray(generated).astype(np.float32)
    bg = np.asarray(background).astype(np.float32)
    out = gen.copy()
    for c in range(3):
        shift = (float(bg[..., c][band].mean()) - float(gen[..., c][band].mean())) * strength
        out[..., c] = np.clip(out[..., c] + shift, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def parse_args():
    p = argparse.ArgumentParser(description="Outpainting inference with ControlNet Union + LoRA")
    # Model paths
    p.add_argument("--base_model", type=str, default="SG161222/RealVisXL_V5.0_Lightning")
    p.add_argument("--controlnet_config", type=str, required=True)
    p.add_argument("--controlnet_weights", type=str, required=True)
    p.add_argument("--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--lora_path", type=str, default="")
    # I/O
    p.add_argument("--input_image", type=str, required=True, help="Single image or directory of images")
    p.add_argument("--output_dir", type=str, default="", help="Output directory (for batch mode)")
    p.add_argument("--output_image", type=str, default="", help="Output path (single image mode)")
    p.add_argument("--debug_dir", type=str, default="")
    # Prompt
    p.add_argument("--prompt", type=str,
                    default="same scene, seamless continuation, same structure, same lighting, photorealistic")
    # Resolution: 0 = auto-select best SDXL bucket from input aspect ratio
    p.add_argument("--width", type=int, default=0, help="Target width (0=auto from SDXL bucket)")
    p.add_argument("--height", type=int, default=0, help="Target height (0=auto from SDXL bucket)")
    # Generation
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=1.5)
    p.add_argument("--controlnet_scale", type=float, default=1.0)
    # Outpainting config
    p.add_argument("--overlap_percentage", type=float, default=10)
    p.add_argument("--resize_option", type=str, default="Full",
                    choices=["Full", "50%", "33%", "25%", "Custom"])
    p.add_argument("--custom_resize_percentage", type=int, default=100)
    p.add_argument("--alignment", type=str, default="Middle",
                    choices=["Middle", "Left", "Right", "Top", "Bottom"])
    p.add_argument("--overlap_left", action="store_true")
    p.add_argument("--overlap_right", action="store_true")
    p.add_argument("--overlap_top", action="store_true")
    p.add_argument("--overlap_bottom", action="store_true")
    # Blending
    p.add_argument("--blend_radius", type=int, default=24)
    p.add_argument("--color_match_strength", type=float, default=0.6)
    # Misc
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    return p.parse_args()


@torch.no_grad()
def run_single(
    pipe,
    image: Image.Image,
    args,
    output_path: str,
    debug_dir: Optional[Path] = None,
) -> str:
    """Run outpainting on a single image."""
    # --- Resolve resolution ---
    if args.width > 0 and args.height > 0:
        width, height = args.width, args.height
    else:
        bucket_name, width, height = best_bucket_for_ratio(image.width, image.height)
        print(f"  Auto bucket: {bucket_name} -> {width}x{height}")

    # --- Prepare canvas + mask + cnet_image (matches training exactly) ---
    background, app_mask, cnet_image, preserve_rect = prepare_image_and_mask(
        image=image,
        width=width,
        height=height,
        overlap_percentage=args.overlap_percentage,
        resize_option=args.resize_option,
        custom_resize_percentage=args.custom_resize_percentage,
        alignment=args.alignment,
        overlap_left=args.overlap_left,
        overlap_right=args.overlap_right,
        overlap_top=args.overlap_top,
        overlap_bottom=args.overlap_bottom,
    )

    # --- Blend mask for compositing ---
    blend_mask = make_blend_mask(background.size, preserve_rect, args.blend_radius)

    # --- Encode prompt ---
    prompt = f"{args.prompt}, high quality, 4k" if args.prompt else "high quality, 4k"
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt, args.device, True
    )

    # --- Run pipeline ---
    last_image = None
    for out in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
    ):
        last_image = out

    if last_image is None or not isinstance(last_image, Image.Image):
        raise RuntimeError(f"Pipeline returned invalid output: {type(last_image)}")

    generated_raw = last_image

    # --- Color matching on seam band ---
    generated_adj = color_match_on_seam(
        generated_raw, background, blend_mask, args.color_match_strength
    )

    # --- Composite: keep original in preserve area, generated elsewhere ---
    result = Image.composite(background, generated_adj, blend_mask)

    # --- Save ---
    os.makedirs(Path(output_path).parent, exist_ok=True)
    result.save(output_path)

    # --- Save raw generated (without blending) ---
    raw_path = str(Path(output_path).with_name(Path(output_path).stem + "_raw.png"))
    generated_raw.save(raw_path)

    # --- Debug outputs ---
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(output_path).stem
        background.save(debug_dir / f"{stem}_canvas.png")
        app_mask.save(debug_dir / f"{stem}_mask.png")
        cnet_image.save(debug_dir / f"{stem}_cnet.png")
        blend_mask.save(debug_dir / f"{stem}_blend_mask.png")
        generated_raw.save(debug_dir / f"{stem}_generated_raw.png")
        generated_adj.save(debug_dir / f"{stem}_color_matched.png")
        result.save(debug_dir / f"{stem}_result.png")
        with open(debug_dir / f"{stem}_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "prompt": prompt,
                "target_resolution": [width, height],
                "preserve_rect": preserve_rect,
                "blend_radius": args.blend_radius,
                "color_match_strength": args.color_match_strength,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "controlnet_scale": args.controlnet_scale,
                "seed": args.seed,
            }, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {output_path}")
    return output_path


@torch.no_grad()
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = resolve_dtype(args.dtype)

    # --- Load models ---
    print("Loading ControlNet Union...")
    controlnet = load_controlnet_union(args.controlnet_config, args.controlnet_weights, dtype, args.device)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=dtype).to(args.device)

    print("Loading pipeline...")
    pipe = StableDiffusionXLFillPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        vae=vae,
        controlnet=controlnet,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(args.device)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    if args.lora_path:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed but --lora_path was provided.")
        print(f"Loading LoRA from {args.lora_path}...")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)
        pipe.unet.to(device=args.device, dtype=dtype)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    # --- Single image or batch ---
    input_path = Path(args.input_image)

    if input_path.is_file():
        output = args.output_image or str(Path(args.output_dir or ".") / f"{input_path.stem}_outpaint.png")
        image = Image.open(input_path).convert("RGB")
        print(f"Processing: {input_path.name} ({image.width}x{image.height})")
        run_single(pipe, image, args, output, debug_dir)

    elif input_path.is_dir():
        out_dir = Path(args.output_dir or str(input_path / "outpaint_results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        image_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        print(f"Found {len(image_files)} images in {input_path}")
        for i, img_path in enumerate(image_files, 1):
            image = Image.open(img_path).convert("RGB")
            output = str(out_dir / f"{img_path.stem}_outpaint.png")
            print(f"[{i}/{len(image_files)}] {img_path.name} ({image.width}x{image.height})")
            run_single(pipe, image, args, output, debug_dir)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Cleanup
    del pipe, controlnet, vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
