"""Task 1 — Model definition + loading."""

import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .nets.net import FusionNet, L_net, ToneNet, compute_mef_quality
from .nets.restormer import Attention
from .utils import image_read

_TASK1_DIR = Path(__file__).parent
MODEL_PHASE_1 = os.environ.get("MODEL_PHASE_1", str(_TASK1_DIR / "model/phase_1/final.pth"))
MODEL_PHASE_2 = os.environ.get("MODEL_PHASE_2", str(_TASK1_DIR / "model/phase_2/final_with_tonenet.pth"))
MAX_LONG_SIDE = int(os.environ.get("MAX_IMAGE_SIZE", "3840"))
MAX_PIXELS_32BIT = (2**31 - 1) // 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    from .nets.net import SRE  # noqa: F401 — imported for type hint only
    B, N, C, H, W = images.shape
    feat_list = []
    for n in range(N):
        feat_list.append(self.encoder(self.patch_embed(images[:, n])))
    feats = torch.stack(feat_list, dim=1)
    del feat_list
    R_feat, attn_weights = self.aggregator(feats)
    del feats
    from .nets.net import SRE as _SRE  # noqa
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
    from .nets.net import SRE, FusionNet as _FN
    Attention.forward = _attn_forward_opt
    SRE.forward = _sre_forward_seq
    _FN.forward = _fusionnet_forward_opt


# ── Resolution helpers ───────────────────────────────────────────────────────

def safe_size(H, W):
    if H * W <= MAX_PIXELS_32BIT and max(H, W) <= MAX_LONG_SIDE:
        return None
    scale = min((MAX_PIXELS_32BIT / (H * W)) ** 0.5, MAX_LONG_SIDE / max(H, W), 1.0)
    return max(int(H * scale) // 8 * 8, 8), max(int(W * scale) // 8 * 8, 8)


def downscale(images, size):
    B, N, C, H, W = images.shape
    flat = F.interpolate(images.reshape(B * N, C, H, W), size=size,
                         mode='bilinear', align_corners=False)
    return flat.reshape(B, N, C, *size)


# ── Model class ──────────────────────────────────────────────────────────────

class Task1Model:
    def __init__(self):
        self.loaded = False
        self.net_fusion = self.net_L = self.tone_net = None
        self.device = DEVICE
        self.fp16 = (DEVICE == "cuda")

    # Support .to(device) so ModelManager can move it
    def to(self, device: str):
        for net in (self.net_fusion, self.net_L, self.tone_net):
            if net is not None:
                net.to(device)
        return self

    def load(self, target_device: str = "cpu"):
        _apply_patches()
        p1 = Path(MODEL_PHASE_1)
        if not p1.exists():
            print(f"[Task1] WARNING: weights not found at {p1} — inference unavailable.")
            return

        self.net_fusion = FusionNet().to(target_device)
        self.net_L = L_net().to(target_device)

        ckpt = torch.load(p1, map_location=target_device, weights_only=False)
        if 'net_fusion' in ckpt:
            try:
                self.net_fusion.load_state_dict(ckpt['net_fusion'])
            except RuntimeError:
                self.net_fusion.load_pretrained(str(p1))
            if 'net_L' in ckpt:
                self.net_L.load_state_dict(ckpt['net_L'])
        else:
            self.net_fusion.load_pretrained(str(p1))

        self.net_fusion.eval()
        self.net_L.eval()

        self.tone_net = None
        p2 = Path(MODEL_PHASE_2)
        if p2.exists():
            tn = torch.load(p2, map_location=target_device, weights_only=False)
            if 'tone_net' in tn:
                self.tone_net = ToneNet().to(target_device)
                self.tone_net.load_state_dict(tn['tone_net'])
                self.tone_net.eval()
                if self.fp16:
                    self.tone_net.half()

        if self.fp16:
            self.net_fusion.half()
            self.net_L.half()

        self.loaded = True
        print(f"[Task1] Loaded on {target_device} (fp16={self.fp16})")

    def infer(self, paths: list, apply_phase2: bool = True, align: bool = False) -> np.ndarray:
        from .inference import _align_images_sift
        imgs = []
        for p in paths:
            img = image_read(p, 'RGB').transpose(2, 0, 1)[np.newaxis] / 255.0
            imgs.append(torch.FloatTensor(img))

        dtype = torch.float16 if self.fp16 else torch.float32
        dev = next(self.net_fusion.parameters()).device  # actual device (CPU or GPU)

        if len(paths) == 1:
            output = imgs[0].to(dev)
            _, C, H, W = output.shape
            orig = (H, W)
            s = safe_size(H, W)
            if s:
                output = F.interpolate(output, size=s, mode='bilinear', align_corners=False)
            if apply_phase2 and self.tone_net is not None:
                with torch.no_grad():
                    with torch.autocast('cuda', dtype=dtype,
                                        enabled=(self.fp16 and str(dev) != 'cpu')):
                        inp = output.half() if self.fp16 else output
                        result, _ = self.tone_net(inp)
                output = result.float()
            if s:
                output = F.interpolate(output, size=orig, mode='bicubic', align_corners=False)
            output = output.clamp(0, 1)
        else:
            images = torch.cat(imgs, 0).unsqueeze(0)
            if align:
                images = _align_images_sift(images)
            B, N, C, H, W = images.shape
            orig = (H, W)
            s = safe_size(H, W)
            if s:
                images = downscale(images, s)
            g = images.to(dev)
            with torch.no_grad():
                with torch.autocast('cuda', dtype=dtype,
                                     enabled=(self.fp16 and str(dev) != 'cpu')):
                    L_maps = torch.stack([self.net_L(g[:, n]) for n in range(N)], dim=1)
                    output, *_ = self.net_fusion(g, L_maps=L_maps)
                    del g, L_maps
                    if apply_phase2 and self.tone_net is not None:
                        output, _ = self.tone_net(output)
            output = output.float()
            if s:
                output = F.interpolate(output, size=orig,
                                       mode='bicubic', align_corners=False).clamp(0, 1)

        arr = np.squeeze(output.cpu().numpy()).transpose(1, 2, 0).clip(0, 1)
        del output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (arr * 255).astype(np.uint8)


# ── Singleton ────────────────────────────────────────────────────────────────
model = Task1Model()
