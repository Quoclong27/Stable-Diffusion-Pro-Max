"""Task 1 — SIFT alignment + inference entry point."""

import time

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from .model import model
from .utils import image_read


# ── SIFT Alignment ────────────────────────────────────────────────────────────

def _validate_homography(H_mat, W, H, max_shift_frac=0.25, max_scale=2.0):
    if H_mat is None:
        return False
    det = np.linalg.det(H_mat)
    if det <= 0:
        return False
    h7, h8 = H_mat[2, 0], H_mat[2, 1]
    persp_magnitude = np.sqrt(h7 ** 2 + h8 ** 2) * max(W, H)
    if persp_magnitude > max_shift_frac * max(W, H):
        return False
    scale = np.sqrt(abs(np.linalg.det(H_mat[:2, :2])))
    if scale < 1.0 / max_scale or scale > max_scale:
        return False
    max_drift = max_shift_frac * max(W, H)
    corners = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]).reshape(-1, 1, 2)
    try:
        warped_corners = cv2.perspectiveTransform(corners, H_mat)
    except cv2.error:
        return False
    for (ox, oy), (wc,) in zip(corners[:, 0], warped_corners):
        wx, wy = wc
        if abs(wx - ox) > max_drift or abs(wy - oy) > max_drift:
            return False
    return True


def _align_images_sift(images_tensor, ratio_thresh=0.65, ransac_thresh=4.0,
                       min_inliers=15, min_inlier_ratio=0.25):
    """Align all exposures to reference using SIFT + validated homography.
    images_tensor: (1, N, 3, H, W) float32 [0,1] on CPU
    Returns: aligned (1, N, 3, H, W) float32 [0,1]
    """
    B, N, C, H, W = images_tensor.shape
    imgs_np = (images_tensor[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.04)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    gray_list, kp_list, des_list = [], [], []
    for i in range(N):
        g = cv2.cvtColor(imgs_np[i], cv2.COLOR_RGB2GRAY)
        g = cv2.equalizeHist(g)
        gray_list.append(g)
        kp, des = sift.detectAndCompute(g, None)
        kp_list.append(kp)
        des_list.append(des)

    counts = [len(k) if k is not None else 0 for k in kp_list]
    max_c, min_c = max(counts), min(counts)
    ref_idx = N // 2 if (max_c - min_c < 50) else int(np.argmax(counts))
    print(f"[Task1][align] Reference: exposure {ref_idx + 1}/{N} ({counts[ref_idx]} keypoints)")

    kp_ref, des_ref = kp_list[ref_idx], des_list[ref_idx]
    if des_ref is None or len(kp_ref) < min_inliers:
        print("[Task1][align] Reference has too few keypoints — skipping alignment")
        return images_tensor

    aligned = list(imgs_np)
    for i in range(N):
        if i == ref_idx:
            continue
        kp_src, des_src = kp_list[i], des_list[i]
        if des_src is None or len(kp_src) < min_inliers:
            continue
        raw_matches = matcher.knnMatch(des_src, des_ref, k=2)
        good = [m for pair in raw_matches if len(pair) == 2
                for m, n2 in [pair] if m.distance < ratio_thresh * n2.distance]
        if len(good) < min_inliers:
            continue
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H_mat, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh, confidence=0.999, maxIters=2000,
        )
        n_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = n_inliers / len(good)
        if not _validate_homography(H_mat, W, H):
            print(f"[Task1][align] Image {i + 1}: homography degenerate, skipping")
            continue
        if n_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
            continue
        aligned[i] = cv2.warpPerspective(
            imgs_np[i], H_mat, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        print(f"[Task1][align] Image {i + 1} aligned ({n_inliers} inliers, {inlier_ratio:.0%})")

    aligned_np = np.stack(aligned, axis=0)
    aligned_tensor = torch.from_numpy(aligned_np.astype(np.float32) / 255.0)
    return aligned_tensor.permute(0, 3, 1, 2).unsqueeze(0)


# ── Path helper ──────────────────────────────────────────────────────────────

def to_path(f) -> str:
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get('path') or f.get('name')
    return getattr(f, 'path', getattr(f, 'name', str(f)))


# ── Public inference entry point ─────────────────────────────────────────────

def run(files, apply_phase2: bool = True, align: bool = False):
    if not files:
        raise gr.Error("Vui lòng chọn ảnh trước khi xử lý.")
    if not model.loaded:
        raise gr.Error("Không tìm thấy file weights của model. Kiểm tra lại cài đặt.")

    from model_manager import manager

    paths = list(dict.fromkeys(to_path(f) for f in files))

    if len(paths) > 1:
        sizes = []
        for p in paths:
            try:
                with Image.open(p) as im:
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
        if model.tone_net is None:
            raise gr.Error("Cần 2+ ảnh để ghép, hoặc cần file weights đầy đủ để xử lý ảnh đơn.")

    manager.activate("task1")
    manager.start_inference("task1")
    try:
        t0 = time.time()
        arr = model.infer(paths, apply_phase2=apply_phase2, align=align)
        dt = round(time.time() - t0, 2)
    finally:
        manager.end_inference()

    if len(paths) == 1:
        mode_info = "Tăng cường màu sắc"
    elif apply_phase2 and align:
        mode_info = "Căn chỉnh + Ghép ảnh + Tăng cường màu sắc"
    elif apply_phase2:
        mode_info = "Ghép ảnh + Tăng cường màu sắc"
    elif align:
        mode_info = "Căn chỉnh + Ghép ảnh thô"
    else:
        mode_info = "Ghép ảnh thô"
    info = f"Hoàn thành trong {dt}s · {mode_info} · {len(paths)} ảnh · {arr.shape[1]}×{arr.shape[0]}px"
    return Image.fromarray(arr), info
