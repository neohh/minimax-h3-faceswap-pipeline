"""
MiniMax H3 Head-Crop + Rock-Solid InSwapper Pipeline (v2.0.0)
=============================================================
Cinematic actor replacement and face swapping with VFX-grade temporal stability.

Upgrades in v2.0.0:
- Dense Optical Flow Temporal Warping (OpenCV DIS Flow 65/35 blend).
- Smoothstep S-Curve Face Gate for extreme 3D tilts (pitch > 35°).
- Elimination of GPEN plastic distortion in favor of optical unsharp sharpening.
- Dynamic 3D mask centroid tracking for tight framing.
"""

import os
import cv2
import json
import time
import shutil
import random
import argparse
import urllib.request
import numpy as np
import insightface


def step1_prepare_crop(source_video, mask_video, start_frame=0, num_frames=56, crop_sz=220, dynamic_track=True, comfy_input_dir=r"C:\Work\docker\ComfyUI\input"):
    """
    Extracts a tight crop around the user's mask (+ hair room) and scales to 768x768.
    Supports dynamic centroid tracking or static bounding box.
    """
    print("\n[Step 1] Preparing tight crop & user mask (768x768)...")
    cap_s = cv2.VideoCapture(source_video)
    cap_m = cv2.VideoCapture(mask_video)
    
    cap_s.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_m.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fps = cap_s.get(cv2.CAP_PROP_FPS) or 24.0
    w_orig = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_s, frames_m = [], []
    centers = []
    
    for _ in range(num_frames):
        r1, fs = cap_s.read()
        r2, fm = cap_m.read()
        if not r1: break
        frames_s.append(fs)
        mask_frame = fm if r2 else np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        frames_m.append(mask_frame)
        
        gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 30)
        if len(xs) > 0:
            centers.append((float(np.mean(xs)), float(np.mean(ys))))
        else:
            centers.append(centers[-1] if centers else (w_orig / 2.0, h_orig / 3.0))
            
    cap_s.release()
    cap_m.release()
    
    n = len(frames_s)
    
    # Smooth centroid trajectory
    smooth_cx, smooth_cy = [], []
    for i in range(n):
        s_i = max(0, i - 2)
        e_i = min(n, i + 3)
        smooth_cx.append(np.mean([centers[k][0] for k in range(s_i, e_i)]))
        smooth_cy.append(np.mean([centers[k][1] for k in range(s_i, e_i)]))
        
    crop_coords = []
    tmp_s_raw = os.path.join(comfy_input_dir, "temp_crop_s_raw.mp4")
    tmp_m_raw = os.path.join(comfy_input_dir, "temp_crop_m_raw.mp4")
    
    wr_s = cv2.VideoWriter(tmp_s_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (768, 768))
    wr_m = cv2.VideoWriter(tmp_m_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (768, 768))
    
    for i in range(n):
        if dynamic_track:
            cx, cy = smooth_cx[i], smooth_cy[i]
        else:
            cx, cy = np.mean(smooth_cx), np.mean(smooth_cy)
            
        x1 = int(np.clip(cx - crop_sz // 2, 0, w_orig - crop_sz))
        y1 = int(np.clip(cy - crop_sz // 2, 0, h_orig - crop_sz))
        crop_coords.append((x1, y1, crop_sz))
        
        c_s = cv2.resize(frames_s[i][y1:y1+crop_sz, x1:x1+crop_sz], (768, 768), interpolation=cv2.INTER_LANCZOS4)
        c_m_raw = frames_m[i][y1:y1+crop_sz, x1:x1+crop_sz]
        
        # Soft-feather mask edges for clean diffusion integration
        gray_m = cv2.cvtColor(c_m_raw, cv2.COLOR_BGR2GRAY)
        dilated = cv2.dilate(gray_m, np.ones((7, 7), np.uint8), iterations=1)
        soft_m = cv2.GaussianBlur(dilated, (15, 15), 5)
        c_m = cv2.resize(cv2.cvtColor(soft_m, cv2.COLOR_GRAY2BGR), (768, 768), interpolation=cv2.INTER_LANCZOS4)
        
        wr_s.write(c_s)
        wr_m.write(c_m)
        
    wr_s.release()
    wr_m.release()
    
    os.system("docker exec comfyui ffmpeg -y -i /opt/comfyui/input/temp_crop_s_raw.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart /opt/comfyui/input/source.mp4")
    os.system("docker exec comfyui ffmpeg -y -i /opt/comfyui/input/temp_crop_m_raw.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart /opt/comfyui/input/mask.mp4")
    print(f"Prepared 768p inputs with {'dynamic' if dynamic_track else 'static'} framing.")
    return crop_coords, (w_orig, h_orig), fps


def step2_generate_minimax_h3(workflow_json, comfy_url="http://127.0.0.1:8188", comfy_output_dir=r"C:\Work\docker\ComfyUI\output\h3-relay", prefix="h3-relay/crop_run"):
    """
    Submits prompt to ComfyUI and waits for output.
    """
    print("\n[Step 2] Queuing MiniMax H3 generation...")
    with open(workflow_json, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)
        
    prompt_dict["save"]["inputs"]["filename_prefix"] = prefix
    seed = random.randint(100000, 999999999)
    if "plan" in prompt_dict and "seed" in prompt_dict["plan"]["inputs"]:
        prompt_dict["plan"]["inputs"]["seed"] = seed
    
    req_data = json.dumps({"prompt": prompt_dict}).encode("utf-8")
    req = urllib.request.Request(f"{comfy_url}/prompt", data=req_data, headers={"Content-Type": "application/json"})
    res = urllib.request.urlopen(req).read()
    prompt_id = json.loads(res).get("prompt_id")
    print(f"Queued prompt ID: {prompt_id}. Waiting for completion...")
    
    while True:
        time.sleep(5)
        try:
            url = f"{comfy_url}/history/{prompt_id}"
            res = urllib.request.urlopen(url).read()
            data = json.loads(res)
            if prompt_id in data:
                print("Generation complete!")
                break
        except Exception:
            pass
            
    files = [f for f in os.listdir(comfy_output_dir) if f.startswith(os.path.basename(prefix)) and f.endswith(".mp4")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(comfy_output_dir, x)))
    return os.path.join(comfy_output_dir, files[-1])


def step3_rock_solid_inswapper_v2(input_video, ref_image, output_swapped_video,
                                  insightface_root=r"C:\Work\docker\ComfyUI\models\insightface",
                                  inswapper_onnx=r"C:\Work\docker\ComfyUI\models\insightface\inswapper_128.onnx",
                                  enable_optical_flow=True,
                                  enable_pitch_gate=True):
    """
    VFX-Grade Rock-Solid InSwapper with:
    1. Zero-drop linear interpolation.
    2. 5-frame temporal landmark smoothing.
    3. Dense Optical Flow Temporal Warping (DIS Flow, 65/35 blend).
    4. Smoothstep S-curve tilt gate on extreme pitch angles (>35°).
    5. Clean optical unsharp mask (NO GPEN distortion).
    """
    print("\n[Step 3] Running VFX Rock-Solid InSwapper v2.0...")
    app = insightface.app.FaceAnalysis(name='buffalo_l', root=insightface_root)
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.20)
    swapper = insightface.model_zoo.get_model(inswapper_onnx)
    
    ref = cv2.imread(ref_image)
    source_face = sorted(app.get(ref), key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    
    n = len(frames)
    raw_kps, raw_bbox, det_mask = [], [], []
    for f in frames:
        faces = app.get(f)
        if faces:
            tf = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
            raw_kps.append(tf.kps.copy())
            raw_bbox.append(tf.bbox.copy())
            det_mask.append(True)
        else:
            raw_kps.append(None)
            raw_bbox.append(None)
            det_mask.append(False)
            
    print(f"Tracking coverage: {det_mask.count(True)}/{n} frames detected.")
    
    # 1. Zero-drop continuous interpolation
    for i in range(n):
        if not det_mask[i]:
            p = next((k for k in range(i-1, -1, -1) if det_mask[k]), None)
            nxt = next((k for k in range(i+1, n) if det_mask[nxt]), None)
            if p is not None and nxt is not None:
                w_f = (i - p) / (nxt - p)
                raw_kps[i] = raw_kps[p] * (1.0 - w_f) + raw_kps[nxt] * w_f
                raw_bbox[i] = raw_bbox[p] * (1.0 - w_f) + raw_bbox[nxt] * w_f
            elif p is not None:
                raw_kps[i] = raw_kps[p].copy()
                raw_bbox[i] = raw_bbox[p].copy()
            elif nxt is not None:
                raw_kps[i] = raw_kps[nxt].copy()
                raw_bbox[i] = raw_bbox[nxt].copy()
                
    # 2. 5-frame moving-average temporal smoothing
    smooth_kps, smooth_bbox = [], []
    for i in range(n):
        s_i = max(0, i - 2)
        e_i = min(n, i + 3)
        smooth_kps.append(np.mean([raw_kps[k] for k in range(s_i, e_i)], axis=0))
        smooth_bbox.append(np.mean([raw_bbox[k] for k in range(s_i, e_i)], axis=0))
        
    class FakeFace:
        def __init__(self, kps, bbox):
            self.kps = kps
            self.bbox = bbox
            
    # 3. Compute per-frame raw swaps
    raw_swaps = []
    for i in range(n):
        ff = FakeFace(smooth_kps[i], smooth_bbox[i])
        sw = swapper.get(frames[i].copy(), ff, source_face, paste_back=True)
        raw_swaps.append(sw)
        
    # 4. Dense Optical Flow Temporal Filter (DIS Flow)
    if enable_optical_flow and n > 1:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        coherent_swaps = [raw_swaps[0].copy()]
        for i in range(1, n):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            
            flow = dis.calc(curr_gray, prev_gray, None)
            fh, fw = curr_gray.shape
            map_x = np.tile(np.arange(fw), (fh, 1)).astype(np.float32) + flow[:, :, 0]
            map_y = np.tile(np.arange(fh), (fw, 1)).T.astype(np.float32) + flow[:, :, 1]
            warped_prev = cv2.remap(coherent_swaps[i-1], map_x, map_y, cv2.INTER_LINEAR)
            
            curr_sw = raw_swaps[i].astype(np.float32)
            warped_p = warped_prev.astype(np.float32)
            coherent_sw = curr_sw * 0.65 + warped_p * 0.35
            coherent_swaps.append(coherent_sw.clip(0, 255).astype(np.uint8))
    else:
        coherent_swaps = raw_swaps
        
    # 5. Smoothstep S-curve Gate for extreme pitch angles
    gate_weights = np.ones(n, dtype=np.float32)
    if enable_pitch_gate:
        # Detect steep tilt when eye-mouth distance collapses relative to bbox height
        for i in range(n):
            kps = smooth_kps[i]
            eye_mid_y = (kps[0][1] + kps[1][1]) / 2.0
            mouth_mid_y = (kps[3][1] + kps[4][1]) / 2.0
            vertical_dist = max(1.0, mouth_mid_y - eye_mid_y)
            bbox_h = max(1.0, smooth_bbox[i][3] - smooth_bbox[i][1])
            ratio = vertical_dist / bbox_h
            
            # Normal ratio is ~0.35-0.50. When tilted backward, drops below 0.22
            if ratio < 0.20:
                gate_weights[i] = 0.0
            elif ratio < 0.32:
                t = (ratio - 0.20) / 0.12
                gate_weights[i] = 3*t*t - 2*t*t*t
            else:
                gate_weights[i] = 1.0
                
        # Temporal smoothing on gate curve to prevent abrupt transitions
        smooth_gate = []
        for i in range(n):
            s_i = max(0, i - 2)
            e_i = min(n, i + 3)
            smooth_gate.append(float(np.mean(gate_weights[s_i:e_i])))
        gate_weights = np.array(smooth_gate, dtype=np.float32)
        
    # 6. Apply gate and subtle unsharp sharpening (NO GPEN)
    final_swapped_frames = []
    for i in range(n):
        gw = gate_weights[i]
        sw_f = coherent_swaps[i].astype(np.float32)
        base_f = frames[i].astype(np.float32)
        gated = sw_f * gw + base_f * (1.0 - gw)
        gated_u8 = gated.clip(0, 255).astype(np.uint8)
        
        # Unsharp mask
        blur = cv2.GaussianBlur(gated_u8, (0, 0), 2.0)
        sharp = cv2.addWeighted(gated_u8, 1.25, blur, -0.25, 0)
        final_swapped_frames.append(sharp)
        
    comfy_out = r"C:\Work\docker\ComfyUI\output"
    tmp_raw = os.path.join(comfy_out, "temp_swap_v2_raw.mp4")
    out = cv2.VideoWriter(tmp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for sf in final_swapped_frames:
        out.write(sf)
    out.release()
    
    tmp_h264 = os.path.join(comfy_out, "temp_swap_v2_h264.mp4")
    os.system("docker exec comfyui ffmpeg -y -i /opt/comfyui/output/temp_swap_v2_raw.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart /opt/comfyui/output/temp_swap_v2_h264.mp4")
    shutil.copy2(tmp_h264, output_swapped_video)
    return final_swapped_frames


def step4_pixel_perfect_composite_v2(source_video, mask_video, holdout_matte, swapped_frames, crop_coords, output_master_video, start_frame=0):
    """
    Sub-pixel alignment: pastes dynamically tracked crops back into exact native canvas coordinates.
    Feathered blending using user 3D mask with holdout protection.
    """
    print("\n[Step 4] Pixel-perfect native compositing v2.0...")
    cap_full = cv2.VideoCapture(source_video)
    cap_mask = cv2.VideoCapture(mask_video)
    cap_holdout = cv2.VideoCapture(holdout_matte) if (holdout_matte and os.path.exists(holdout_matte)) else None
    
    cap_full.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_mask.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if cap_holdout:
        cap_holdout.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
    fps = cap_full.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap_full.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_full.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    comfy_out = r"C:\Work\docker\ComfyUI\output"
    tmp_comp_raw = os.path.join(comfy_out, "temp_composite_v2_raw.mp4")
    out = cv2.VideoWriter(tmp_comp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for i, sw_frame in enumerate(swapped_frames):
        r1, frame_orig = cap_full.read()
        r2, mask_frame = cap_mask.read()
        if not r1: break
        
        x1, y1, sz = crop_coords[i]
        
        # Resize swapped crop back to patch size
        sw_patch = cv2.resize(sw_frame, (sz, sz), interpolation=cv2.INTER_LANCZOS4)
        
        # User mask patch with Gaussian feathering
        m_roi = mask_frame[y1:y1+sz, x1:x1+sz] if r2 else np.ones((sz, sz, 3), dtype=np.uint8) * 255
        m_gray = cv2.cvtColor(m_roi, cv2.COLOR_BGR2GRAY)
        
        # Holdout subtraction (if foreground foreground exists)
        if cap_holdout:
            r3, f_hold = cap_holdout.read()
            if r3:
                h_roi = f_hold[y1:y1+sz, x1:x1+sz]
                h_gray = cv2.cvtColor(h_roi, cv2.COLOR_BGR2GRAY)
                m_gray = np.clip(m_gray.astype(np.int16) - h_gray.astype(np.int16), 0, 255).astype(np.uint8)
                
        m_soft = cv2.GaussianBlur(m_gray, (19, 19), 6).astype(np.float32) / 255.0
        m_soft = m_soft[:, :, np.newaxis]
        
        orig_roi = frame_orig[y1:y1+sz, x1:x1+sz].astype(np.float32)
        blended_roi = sw_patch.astype(np.float32) * m_soft + orig_roi * (1.0 - m_soft)
        frame_orig[y1:y1+sz, x1:x1+sz] = blended_roi.clip(0, 255).astype(np.uint8)
        
        out.write(frame_orig)
        
    out.release()
    cap_full.release()
    cap_mask.release()
    if cap_holdout: cap_holdout.release()
    
    tmp_h264 = os.path.join(comfy_out, "temp_composite_v2_h264.mp4")
    os.system("docker exec comfyui ffmpeg -y -i /opt/comfyui/output/temp_composite_v2_raw.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart /opt/comfyui/output/temp_composite_v2_h264.mp4")
    shutil.copy2(tmp_h264, output_master_video)
    print(f"Master composite saved: {output_master_video}")
