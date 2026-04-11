"""Task 1: Multi-Exposure Fusion using UFRetinex-MEF-ToneNet."""

import gc
import os
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .nets.net import FusionNet, L_net, SRE, ToneNet, compute_mef_quality
from .nets.restormer import Attention
from .utils import image_read

_TASK1_DIR = Path(__file__).parent
MODEL_PHASE_1 = os.environ.get("MODEL_PHASE_1", str(_TASK1_DIR / "model/phase_1/final.pth"))
MODEL_PHASE_2 = os.environ.get("MODEL_PHASE_2", str(_TASK1_DIR / "model/phase_2/final_with_tonenet.pth"))
MAX_LONG_SIDE = int(os.environ.get("MAX_IMAGE_SIZE", "3840"))
MAX_PIXELS_32BIT = (2**31 - 1) // 256


# ── Memory-optimisation patches ──────────────────────────────────────────────

def _attn_forward_opt(self, x, crossatt=None):
    from einops import rearrange
    b, c, h, w = x.shape
    k = self.k_dwconv(self.k(x))
    v = self.v_dwconv(self.v(x))
    q = self.q_dwconv(self.q(x if crossatt is None else crossatt))
    q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    attn = (q @ k.transpose(-2, -1)) * self.temperature
    attn = attn.softmax(dim=-1)
    out = attn @ v
    del q, k, attn
    out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                    head=self.num_heads, h=h, w=w)
    return self.project_out(out)


def _sre_forward_seq(self, images):
    B, N, C, H, W = images.shape
    feat_list = []
    for n in range(N):
        feat_list.append(self.encoder(self.patch_embed(images[:, n])))
    feats = torch.stack(feat_list, dim=1)
    del feat_list
    R_feat, attn_weights = self.aggregator(feats)
    del feats
    Rhat = self.act(self.output(R_feat))
    return Rhat, R_feat, attn_weights


def _fusionnet_forward_opt(self, images, L_maps=None, L_override=None):
    if L_override is not None:
        L_fused, L_weights = L_override, None
    else:
        assert L_maps is not None
        L_fused, L_weights = self.l_fusion(L_maps)
    Rhat_raw, _, _ = self.SRE(images)
    if self.use_retinex_prior:
        if L_maps is not None:
            quality = compute_mef_quality(L_maps)
            w = quality / (quality.sum(dim=1, keepdim=True) + 1e-6)
            I_mean = images.mean(dim=[3, 4], keepdim=True).clamp(min=0.01)
            target_I = (I_mean * w.mean(dim=[3, 4], keepdim=True)).sum(dim=1, keepdim=True) \
                       / (w.mean(dim=[3, 4], keepdim=True).sum(dim=1, keepdim=True) + 1e-6)
            images_comp = (images * (target_I / I_mean)).clamp(0, 1)
            retinex_prior = (images_comp * w).sum(dim=1).clamp(0, 1)
            del quality, w, images_comp
        else:
            mean_exposed = images.mean(dim=1)
            L_safe = L_fused.clamp(min=0.05)
            retinex_prior = (mean_exposed / L_safe).clamp(0, 1)
            del mean_exposed, L_safe
        Rhat = self.retinex_gate(Rhat_raw, retinex_prior)
        del Rhat_raw, retinex_prior
    else:
        Rhat = Rhat_raw
        del Rhat_raw
    L_for_crossatt = F.avg_pool2d(L_fused, kernel_size=21, stride=1, padding=10)
    R = self.patch_embed2(Rhat)
    for block in self.decoder:
        R = block(R, crossatt=L_for_crossatt)
    del L_for_crossatt
    R_refined = torch.clamp(Rhat + self.output2(R), 0, 1)
    del R
    output = R_refined * L_fused
    return output, R_refined, Rhat, L_fused, L_weights


def _apply_patches():
    Attention.forward = _attn_forward_opt
    SRE.forward = _sre_forward_seq
    FusionNet.forward = _fusionnet_forward_opt


# ── Resolution helpers ───────────────────────────────────────────────────────

def _safe_size(H, W):
    if H * W <= MAX_PIXELS_32BIT and max(H, W) <= MAX_LONG_SIDE:
        return None
    scale = min((MAX_PIXELS_32BIT / (H * W)) ** 0.5, MAX_LONG_SIDE / max(H, W), 1.0)
    return max(int(H * scale) // 8 * 8, 8), max(int(W * scale) // 8 * 8, 8)


def _downscale(images, size):
    B, N, C, H, W = images.shape
    flat = F.interpolate(images.reshape(B * N, C, H, W), size=size,
                         mode='bilinear', align_corners=False)
    return flat.reshape(B, N, C, *size)


# ── Model ────────────────────────────────────────────────────────────────────

class _Task1Model:
    def __init__(self):
        self.loaded = False
        self.net_fusion = self.net_L = self.tone_net = None
        self.device = self.fp16 = None

    def load(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fp16 = self.device == 'cuda'
        _apply_patches()

        p1 = Path(MODEL_PHASE_1)
        if not p1.exists():
            print(f"[Task1] WARNING: weights not found at {p1} — inference unavailable.")
            return

        self.net_fusion = FusionNet().to(self.device)
        self.net_L = L_net().to(self.device)

        ckpt = torch.load(p1, map_location=self.device, weights_only=False)
        if 'net_fusion' in ckpt:
            self.net_fusion.load_state_dict(ckpt['net_fusion'])
            if 'net_L' in ckpt:
                self.net_L.load_state_dict(ckpt['net_L'])
        else:
            self.net_fusion.load_pretrained(str(p1))

        self.net_fusion.eval()
        self.net_L.eval()

        self.tone_net = None
        p2 = Path(MODEL_PHASE_2)
        if p2.exists():
            tn = torch.load(p2, map_location=self.device, weights_only=False)
            if 'tone_net' in tn:
                self.tone_net = ToneNet().to(self.device)
                self.tone_net.load_state_dict(tn['tone_net'])
                self.tone_net.eval()
                if self.fp16:
                    self.tone_net.half()

        if self.fp16:
            self.net_fusion.half()
            self.net_L.half()

        self.loaded = True
        print(f"[Task1] Loaded on {self.device} (fp16={self.fp16})")

    def infer(self, paths: list[str], apply_phase2: bool = True) -> np.ndarray:
        imgs = []
        for p in paths:
            img = image_read(p, 'RGB').transpose(2, 0, 1)[np.newaxis] / 255.0
            imgs.append(torch.FloatTensor(img))

        dtype = torch.float16 if self.fp16 else torch.float32

        if len(paths) == 1:
            # ── Single image: bỏ qua Phase 1, chỉ chạy ToneNet (Phase 2) ──
            output = imgs[0].to(self.device)           # (1, 3, H, W)
            _, C, H, W = output.shape
            orig = (H, W)
            safe = _safe_size(H, W)
            if safe:
                output = F.interpolate(output, size=safe,
                                       mode='bilinear', align_corners=False)
            if apply_phase2 and self.tone_net is not None:
                with torch.no_grad():
                    with torch.autocast('cuda', dtype=dtype,
                                        enabled=(self.fp16 and self.device == 'cuda')):
                        inp = output.half() if self.fp16 else output
                        result, _ = self.tone_net(inp)
                output = result.float()
            if safe:
                output = F.interpolate(output, size=orig,
                                       mode='bicubic', align_corners=False)
            output = output.clamp(0, 1)

        else:
            # ── Multiple images: Phase 1 fusion ─────────────────────────
            images = torch.cat(imgs, 0).unsqueeze(0)  # (1, N, 3, H, W)
            B, N, C, H, W = images.shape
            orig = (H, W)
            safe = _safe_size(H, W)
            if safe:
                images = _downscale(images, safe)
            g = images.to(self.device)
            with torch.no_grad():
                with torch.autocast('cuda', dtype=dtype,
                                     enabled=(self.fp16 and self.device == 'cuda')):
                    L_maps = torch.stack([self.net_L(g[:, n]) for n in range(N)], dim=1)
                    output, *_ = self.net_fusion(g, L_maps=L_maps)
                    del g, L_maps
                    if apply_phase2 and self.tone_net is not None:
                        output, _ = self.tone_net(output)
            output = output.float()
            if safe:
                output = F.interpolate(output, size=orig,
                                       mode='bicubic', align_corners=False).clamp(0, 1)

        arr = np.squeeze(output.cpu().numpy()).transpose(1, 2, 0).clip(0, 1)
        del output
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return (arr * 255).astype(np.uint8)


_model = _Task1Model()


def load_model():
    _model.load()


# ── Gradio function ──────────────────────────────────────────────────────────

def _to_path(f) -> str:
    """Resolve Gradio file object to a file-system path string."""
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get('path') or f.get('name')
    return getattr(f, 'path', getattr(f, 'name', str(f)))


def run(files, apply_phase2: bool = True):
    if not files:
        raise gr.Error("Vui lòng chọn ảnh trước khi xử lý.")
    if not _model.loaded:
        raise gr.Error("Không tìm thấy file weights của model. Kiểm tra lại cài đặt.")

    paths = list(dict.fromkeys(_to_path(f) for f in files))  # deduplicate

    # Validate kích thước ảnh khi có nhiều hơn 1 ảnh
    if len(paths) > 1:
        from PIL import Image as _PIL
        sizes = []
        for p in paths:
            try:
                with _PIL.open(p) as im:
                    sizes.append(im.size)
            except Exception as e:
                raise gr.Error(f"Không thể đọc ảnh: {p}\n{e}")
        if len(set(sizes)) > 1:
            details = ", ".join(f"{w}×{h}" for w, h in sizes)
            raise gr.Error(
                f"Các ảnh phải có cùng kích thước để ghép.\n"
                f"Kích thước hiện tại: {details}"
            )

    if len(paths) == 1:
        if not apply_phase2:
            return Image.open(paths[0]).convert("RGB"), "1 ảnh · Bỏ qua tăng cường màu → trả về ảnh gốc"
        if _model.tone_net is None:
            raise gr.Error("Cần 2+ ảnh để ghép, hoặc cần file weights đầy đủ để xử lý ảnh đơn.")

    t0 = time.time()
    arr = _model.infer(paths, apply_phase2=apply_phase2)
    dt = round(time.time() - t0, 2)

    if len(paths) == 1:
        mode_info = "Tăng cường màu sắc"
    elif apply_phase2:
        mode_info = "Ghép ảnh + tăng cường màu sắc"
    else:
        mode_info = "Ghép ảnh thô"
    info = f"Hoàn thành trong {dt}s · {mode_info} · {len(paths)} ảnh · {arr.shape[1]}×{arr.shape[0]}px"
    return Image.fromarray(arr), info


# ── UI ───────────────────────────────────────────────────────────────────────

def create_task1_tab():
    gr.Markdown(
        "**Ghép nhiều ảnh phơi sáng** — Upload 1 ảnh để tăng cường chất lượng, "
        "hoặc 2+ ảnh chụp cùng một cảnh với mức phơi sáng khác nhau để ghép lại. "
        "Hỗ trợ ảnh tối đa 4K."
    )

    with gr.Row():
        # ── Left: inputs ─────────────────────────────────────────────────
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Chọn ảnh (1 hoặc nhiều ảnh)",
                file_count="multiple",
                file_types=["image"],
            )
            preview_gallery = gr.Gallery(
                label="Ảnh đã chọn",
                columns=4,
                object_fit="contain",
                interactive=False,
                show_label=False,
                allow_preview=True,
            )
            apply_phase2 = gr.Checkbox(
                label="Tăng cường màu sắc & ánh sáng sau khi ghép",
                value=True,
            )
            gr.Markdown(
                "_Bỏ tick nếu bạn chỉ muốn lấy kết quả ghép thô, không qua bước xử lý màu._"
            )
            run_btn = gr.Button("Xử lý", variant="primary", size="lg")

        # ── Right: output ─────────────────────────────────────────────────
        with gr.Column(scale=1):
            output_img = gr.Image(label="Kết quả", type="pil", sources=[])

    _info_state = gr.State()

    def _preview(files):
        if not files:
            return []
        # Pass file paths directly — Gallery renders them without loading pixels into Python
        seen, paths = set(), []
        for f in files:
            p = _to_path(f)
            if p not in seen:
                seen.add(p)
                paths.append(p)
        return paths

    file_input.change(fn=_preview, inputs=file_input, outputs=preview_gallery)
    run_btn.click(fn=run, inputs=[file_input, apply_phase2], outputs=[output_img, _info_state])
