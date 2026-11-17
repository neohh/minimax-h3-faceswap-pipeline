#!/usr/bin/env python3
"""
CoTracker3 + Depth-Aware Smooth MLS Neural Projection Engine
Part of the VFX Studio Trifecta: Block 3 - Precision Temporal Texture Projection

Eliminates:
- 2D Optical Flow Accumulated Drift (subpixel drift over time)
- Delaunay Triangulation Facet Seams & Polygonal Cuts
- Jello Effect / Wavy Micro-Rippling from steep weight funnels
- Non-coplanar 3D parallax warping via Depth-Aware Bilateral Weighting
- Flow field micro-noise via Gaussian displacement smoothing

Authors: neohh & Antigravity DeepMind Team
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch

try:
    from scipy.signal import savgol_filter
except ImportError:
    def savgol_filter(x, window_length=7, polyorder=2):
        """Pure-numpy fallback for Savitzky-Golay filter (window=7, poly=2)."""
        if window_length == 7 and polyorder <= 3:
            coeffs = np.array([-2, 3, 6, 7, 6, 3, -2], dtype=np.float32) / 21.0
        elif window_length == 5 and polyorder <= 2:
            coeffs = np.array([-3, 12, 17, 12, -3], dtype=np.float32) / 35.0
        else:
            coeffs = np.ones(window_length, dtype=np.float32) / float(window_length)
        
        pad_w = window_length // 2
        padded = np.pad(x, pad_w, mode='edge')
        smoothed = np.convolve(padded, coeffs, mode='valid')
        return smoothed


def mls_smooth_field(
    p, q, v,
    alpha=0.65,
    r_reg=250.0,
    d_flat_gpu=None,
    pts_depth_gpu=None,
    sigma_d=25.0,
    device='cuda'
):
    """
    Computes a C1-continuous smooth deformation field using Moving Least Squares (MLS)
    with optional Depth-Aware Bilateral Weighting.
    
    p: (N, 2) source anchor points in Frame 0
    q: (N, 2) deformed anchor points in Frame t (from CoTracker3)
    v: (H, W, 2) coordinate grid in destination frame
    alpha: decay exponent (default: 0.65)
    r_reg: radius regularization constant (default: 250.0)
    d_flat_gpu: (H*W,) depth values of destination grid (optional)
    pts_depth_gpu: (N,) depth values of tracked points (optional)
    sigma_d: depth bilateral bandwidth (default: 25.0)
    """
    H, W, _ = v.shape
    v_flat = v.view(-1, 2) # (M, 2)
    
    diff = v_flat.unsqueeze(1) - q.unsqueeze(0) # (M, N, 2)
    dist_sq = torch.sum(diff ** 2, dim=-1) + r_reg
    w_space = 1.0 / (dist_sq ** alpha)
    
    if d_flat_gpu is not None and pts_depth_gpu is not None:
        d_diff = torch.abs(d_flat_gpu.unsqueeze(1) - pts_depth_gpu.unsqueeze(0))
        w_depth = torch.exp(- (d_diff ** 2) / (2.0 * (sigma_d ** 2)))
        w = w_space * w_depth + 1e-6
    else:
        w = w_space
        
    w_sum = torch.sum(w, dim=1, keepdim=True)
    
    # Centroids
    q_star = torch.matmul(w, q) / w_sum
    p_star = torch.matmul(w, p) / w_sum
    
    q_hat = q.unsqueeze(0) - q_star.unsqueeze(1)
    p_hat = p.unsqueeze(0) - p_star.unsqueeze(1)
    
    w_uns = w.unsqueeze(-1).unsqueeze(-1)
    q_hat_col = q_hat.unsqueeze(-1)
    q_hat_row = q_hat.unsqueeze(-2)
    
    # Local covariance matrix C: (M, 2, 2)
    C = torch.sum(w_uns * torch.matmul(q_hat_col, q_hat_row), dim=1)
    eye = torch.eye(2, device=device).unsqueeze(0) * 1e-3
    C_inv = torch.inverse(C + eye)
    
    v_diff = (v_flat - q_star).unsqueeze(1)
    temp = torch.matmul(v_diff, C_inv)
    temp_q = torch.sum(temp * q_hat, dim=-1)
    weighted_temp_q = (w * temp_q).unsqueeze(-1)
    
    mapped = p_star + torch.sum(weighted_temp_q * p_hat, dim=1)
    return mapped.view(H, W, 2)


def run_neural_projection(
    video_path: str,
    mask_path: str,
    hero_path: str,
    output_path: str,
    depth_path: str = None,
    grid_size: int = 35,
    alpha: float = 0.65,
    r_reg: float = 250.0,
    sigma_depth: float = 25.0,
    flow_blur: int = 31,
    savgol_window: int = 7,
    savgol_poly: int = 2,
    dilate_px: int = 7,
    feather_sigma: float = 1.8,
    save_gif: bool = True,
    device_str: str = 'cuda'
):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing CoTracker3 on {device}...")
    model = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device)

    print(f"[*] Reading source video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    orig_bgr = []
    orig_rgb = []
    while True:
        ret, f = cap.read()
        if not ret: break
        orig_bgr.append(f)
        orig_rgb.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    N = len(orig_bgr)
    if N == 0:
        raise ValueError(f"Could not read frames from {video_path}")
    h, w = orig_bgr[0].shape[:2]
    print(f"[+] Loaded {N} frames ({w}x{h}).")

    # Load Depth Video if provided
    depth_frames = []
    if depth_path and os.path.exists(depth_path):
        print(f"[*] Loading Depth Map Video: {depth_path}")
        cap_d = cv2.VideoCapture(depth_path)
        while True:
            ret, f = cap_d.read()
            if not ret: break
            depth_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap_d.release()
        print(f"[+] Loaded {len(depth_frames)} depth frames.")
    else:
        print("[*] No depth map provided. Running in pure 2D Spatial MLS mode.")

    print(f"[*] Reading props mask: {mask_path}")
    cap_m = cv2.VideoCapture(mask_path)
    masks = []
    while True:
        ret, f = cap_m.read()
        if not ret: break
        masks.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap_m.release()
    if len(masks) == 0:
        m_static = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m_static is None:
            raise ValueError(f"Could not read mask from {mask_path}")
        masks = [m_static] * N
    elif len(masks) < N:
        masks += [masks[-1]] * (N - len(masks))
    print(f"[+] Loaded {len(masks)} mask frames.")

    mask0 = (masks[0] > 128).astype(np.uint8)
    video_tensor = torch.from_numpy(np.array(orig_rgb)).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask0.astype(np.float32)).unsqueeze(0).unsqueeze(0).float().to(device)

    print(f"[*] Tracking neural anchor points with CoTracker3 (grid_size={grid_size})...")
    with torch.no_grad():
        pred_tracks, _ = model(video_tensor, grid_size=grid_size, segm_mask=mask_tensor, grid_query_frame=0)

    raw_tracks = pred_tracks[0].cpu().numpy() # (T, N_pts, 2)
    num_pts = raw_tracks.shape[1]
    print(f"[+] CoTracker3 locked {num_pts} ground-truth 3D points.")

    # Temporal Savitzky-Golay filter
    print(f"[*] Applying Savitzky-Golay temporal smoothing (window={savgol_window}, poly={savgol_poly})...")
    smoothed_tracks = np.zeros_like(raw_tracks)
    for i in range(num_pts):
        smoothed_tracks[:, i, 0] = savgol_filter(raw_tracks[:, i, 0], window_length=savgol_window, polyorder=savgol_poly)
        smoothed_tracks[:, i, 1] = savgol_filter(raw_tracks[:, i, 1], window_length=savgol_window, polyorder=savgol_poly)

    pts0_gpu = torch.from_numpy(smoothed_tracks[0]).float().to(device)

    # Dynamic bounding box encompassing all active mask regions and tracked points
    all_mask_ys, all_mask_xs = [], []
    for m in masks:
        ys, xs = np.where(m > 10)
        if len(xs) > 0:
            all_mask_xs.append(xs.min())
            all_mask_xs.append(xs.max())
            all_mask_ys.append(ys.min())
            all_mask_ys.append(ys.max())

    if len(all_mask_xs) > 0:
        mask_min_x, mask_max_x = min(all_mask_xs), max(all_mask_xs)
        mask_min_y, mask_max_y = min(all_mask_ys), max(all_mask_ys)
    else:
        mask_min_x, mask_max_x, mask_min_y, mask_max_y = 0, w, 0, h

    pts_all = smoothed_tracks.reshape(-1, 2)
    pts_min_x = int(np.min(pts_all[:, 0]))
    pts_max_x = int(np.max(pts_all[:, 0]))
    pts_min_y = int(np.min(pts_all[:, 1]))
    pts_max_y = int(np.max(pts_all[:, 1]))

    min_x = max(0, min(mask_min_x, pts_min_x) - 40)
    max_x = min(w, max(mask_max_x, pts_max_x) + 40)
    min_y = max(0, min(mask_min_y, pts_min_y) - 40)
    max_y = min(h, max(mask_max_y, pts_max_y) + 40)

    # If bounds approach image edges (within 50px), snap to frame boundary to prevent any crop seam
    if min_x <= 50: min_x = 0
    if max_x >= w - 50: max_x = w
    if min_y <= 50: min_y = 0
    if max_y >= h - 50: max_y = h

    grid_y, grid_x = torch.meshgrid(
        torch.arange(min_y, max_y, device=device, dtype=torch.float32),
        torch.arange(min_x, max_x, device=device, dtype=torch.float32),
        indexing='ij'
    )
    v_grid = torch.stack([grid_x, grid_y], dim=-1)

    print(f"[*] Loading Hero Frame 0: {hero_path}")
    rgb0 = cv2.imread(hero_path)
    if rgb0 is None:
        raise ValueError(f"Could not read Hero Frame from {hero_path}")
    rgb0_t = torch.from_numpy(rgb0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    final_frames = []

    print("[*] Projecting texture across all frames via Depth-Aware Smooth MLS & Alpha-Clipping...")
    for t in range(N):
        pts_t = smoothed_tracks[t]
        pts_t_gpu = torch.from_numpy(pts_t).float().to(device)
        
        # Depth maps if available
        d_flat_gpu = None
        pts_depth_gpu = None
        if len(depth_frames) > t:
            d_curr = depth_frames[t].astype(np.float32)
            pts_depth = [d_curr[int(np.clip(pt[1], 0, h - 1)), int(np.clip(pt[0], 0, w - 1))] for pt in pts_t]
            pts_depth_gpu = torch.tensor(pts_depth, device=device, dtype=torch.float32)
            d_crop = d_curr[min_y:max_y, min_x:max_x]
            d_flat_gpu = torch.from_numpy(d_crop.reshape(-1)).to(device)
            
        # Deformation field
        V_field = mls_smooth_field(
            pts0_gpu, pts_t_gpu, v_grid,
            alpha=alpha, r_reg=r_reg,
            d_flat_gpu=d_flat_gpu, pts_depth_gpu=pts_depth_gpu, sigma_d=sigma_depth,
            device=device
        )
        
        # Flow Field Gaussian Smoothing if enabled
        if flow_blur > 0:
            dx = (V_field[..., 0] - v_grid[..., 0]).cpu().numpy()
            dy = (V_field[..., 1] - v_grid[..., 1]).cpu().numpy()
            dx_b = cv2.GaussianBlur(dx, (flow_blur, flow_blur), 0)
            dy_b = cv2.GaussianBlur(dy, (flow_blur, flow_blur), 0)
            V_smooth_x = torch.from_numpy(v_grid[..., 0].cpu().numpy() + dx_b).to(device)
            V_smooth_y = torch.from_numpy(v_grid[..., 1].cpu().numpy() + dy_b).to(device)
        else:
            V_smooth_x = V_field[..., 0]
            V_smooth_y = V_field[..., 1]
            
        norm_x = (V_smooth_x / (w - 1.0)) * 2.0 - 1.0
        norm_y = (V_smooth_y / (h - 1.0)) * 2.0 - 1.0
        grid_norm = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
        
        warped_crop = torch.nn.functional.grid_sample(
            rgb0_t, grid_norm, mode='bicubic', padding_mode='reflection', align_corners=True
        )
        warped_crop_np = warped_crop[0].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        # Organic mask clipping
        m_t = (masks[t] > 128).astype(np.uint8)
        dil_m_t = cv2.dilate(m_t, kernel_dilate, iterations=1)
        m_crop = dil_m_t[min_y:max_y, min_x:max_x]
        feather_crop = cv2.GaussianBlur(m_crop.astype(np.float32), (dilate_px, dilate_px), feather_sigma)[..., None]
        
        # Seamless composite
        full_frame = orig_bgr[t].copy()
        bg_crop = orig_bgr[t][min_y:max_y, min_x:max_x].astype(np.float32)
        fg_crop = warped_crop_np.astype(np.float32)
        
        comp_crop = (fg_crop * feather_crop + bg_crop * (1.0 - feather_crop)).clip(0, 255).astype(np.uint8)
        full_frame[min_y:max_y, min_x:max_x] = comp_crop
        final_frames.append(full_frame)

        if (t + 1) % 15 == 0 or t == N - 1:
            print(f"    -> Rendered {t + 1}/{N} frames")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 24.0, (w, h))
    for f in final_frames:
        writer.write(f)
    writer.release()
    print(f"[+] Master MP4 saved to: {output_path}")

    if save_gif:
        import imageio
        gif_path = os.path.splitext(output_path)[0] + "_preview.gif"
        crop_zoom = slice(min_y, max_y), slice(min_x, max_x)
        crops = [cv2.cvtColor(f[crop_zoom], cv2.COLOR_BGR2RGB) for f in final_frames]
        imageio.mimsave(gif_path, crops, fps=12)
        print(f"[+] Preview GIF saved to: {gif_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CoTracker3 + Depth-Aware Smooth MLS Neural Texture Projection Engine")
    parser.add_argument('--video', type=str, required=True, help="Input plate video MP4")
    parser.add_argument('--mask', type=str, required=True, help="Props mask video MP4 or image")
    parser.add_argument('--hero', type=str, required=True, help="Master Hero Frame 0 image")
    parser.add_argument('--output', type=str, required=True, help="Output master video MP4")
    parser.add_argument('--depth', type=str, default=None, help="Depth Map video MP4 for Bilateral 3D separation")
    parser.add_argument('--grid_size', type=int, default=35, help="CoTracker3 grid query density")
    parser.add_argument('--alpha', type=float, default=0.65, help="MLS weight decay factor")
    parser.add_argument('--r_reg', type=float, default=250.0, help="MLS radius regularization (R^2)")
    parser.add_argument('--sigma_d', type=float, default=25.0, help="Bilateral depth bandwidth")
    parser.add_argument('--flow_blur', type=int, default=31, help="Displacement field Gaussian smoothing kernel")
    parser.add_argument('--savgol_window', type=int, default=7, help="Savitzky-Golay window length")
    parser.add_argument('--savgol_poly', type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument('--dilate', type=int, default=7, help="Mask boundary dilation kernel size")
    parser.add_argument('--no_gif', action='store_true', help="Disable GIF preview generation")

    args = parser.parse_args()
    run_neural_projection(
        video_path=args.video,
        mask_path=args.mask,
        hero_path=args.hero,
        output_path=args.output,
        depth_path=args.depth,
        grid_size=args.grid_size,
        alpha=args.alpha,
        r_reg=args.r_reg,
        sigma_depth=args.sigma_d,
        flow_blur=args.flow_blur,
        savgol_window=args.savgol_window,
        savgol_poly=args.savgol_poly,
        dilate_px=args.dilate,
        save_gif=not args.no_gif
    )
