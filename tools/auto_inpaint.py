import os
import sys
import cv2
import json
import time
import shutil
import argparse
import urllib.request
import numpy as np
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8188"
WS_DIR = Path(r"Z:\Work\Scripts and Addons\AI_Mask")
ART_DIR = Path(r"C:\Users\maxim\.gemini\antigravity\brain\09a95a4e-8b62-47fd-8d35-f049d3035823")
COMFY_INPUT = Path(r"C:\Work\docker\ComfyUI\input")
COMFY_OUTPUT = Path(r"C:\Work\docker\ComfyUI\output")
WORKFLOW_TEMPLATE = WS_DIR / "minimax_user_mask_crop_workflow.json"
SAMMIE_TEMP = Path(r"Z:\Work\VFX_Program\Sammie-Roto-2-main\temp")

def get_latest_file(folder: Path, pattern: str = "*.*", exclude: list = None):
    if not folder.exists(): return None
    files = list(folder.glob(pattern))
    if exclude:
        files = [f for f in files if f.name not in exclude and not any(ex in f.name for ex in exclude)]
    if not files: return None
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]

def auto_detect_inputs(video_arg=None, mask_arg=None, ref_arg=None):
    # 1. Mask
    mask_path = None
    if mask_arg and Path(mask_arg).exists():
        mask_path = Path(mask_arg)
    else:
        # Check latest_export.json from Sammie-Roto
        latest_json = SAMMIE_TEMP / "latest_export.json"
        if latest_json.exists():
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    p = data.get("mask_path")
                    if p and Path(p).exists(): mask_path = Path(p)
            except Exception: pass
        if not mask_path:
            # Find latest mask file in WS_DIR
            mask_path = get_latest_file(WS_DIR, "*маска*.mp4") or get_latest_file(WS_DIR, "*mask*.mp4")

    # 2. Video
    video_path = None
    if video_arg and Path(video_arg).exists():
        video_path = Path(video_arg)
    else:
        # Look for session video or latest master video
        session_conf = SAMMIE_TEMP / "session_settings.conf"
        if session_conf.exists():
            try:
                with open(session_conf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    p = data.get("video_file_path")
                    if p and Path(p).exists(): video_path = Path(p)
            except Exception: pass
        if not video_path:
            video_path = get_latest_file(WS_DIR, "*Master*.mp4", exclude=["SideBySide"]) or get_latest_file(WS_DIR, "*.mp4", exclude=["маска", "mask", "SideBySide", "crop"])

    # 3. Ref Image
    ref_path = None
    if ref_arg and Path(ref_arg).exists():
        ref_path = Path(ref_arg)
    else:
        user_up = ART_DIR / ".user_uploaded"
        ref_path = get_latest_file(user_up, "*.jpg") or get_latest_file(user_up, "*.png")

    return video_path, mask_path, ref_path

def run_pipeline(video_path: Path, mask_path: Path, ref_path: Path, mode: str = "auto", prompt: str = None, output_name: str = "Auto_Inpaint_Master.mp4"):
    print("=" * 65)
    print("⚡ UNIVERSAL AUTO-INPAINT ENGINE")
    print("=" * 65)
    print(f"🎬 Video:     {video_path.name if video_path else 'NOT FOUND'}")
    print(f"🎭 Mask:      {mask_path.name if mask_path else 'NOT FOUND'}")
    print(f"🖼 Reference: {ref_path.name if ref_path else 'NOT FOUND'}")
    print(f"⚙ Mode:      {mode}")

    if not video_path or not mask_path or not ref_path:
        print("❌ Error: Missing required inputs. Please specify --video, --mask, or --ref.")
        return False

    cap_v = cv2.VideoCapture(str(video_path))
    cap_m = cv2.VideoCapture(str(mask_path))
    fps = cap_v.get(cv2.CAP_PROP_FPS) or 24.0
    w_orig = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))

    v_frames, m_frames = [], []
    while True:
        rv, fv = cap_v.read()
        rm, fm = cap_m.read()
        if not rv or not rm: break
        v_frames.append(fv)
        m_frames.append(fm)
        if len(v_frames) >= 56: break # Default DiT window
    cap_v.release()
    cap_m.release()

    n = len(v_frames)
    print(f"[+] Loaded {n} frames ({w_orig}x{h_orig} @ {fps:.2f} fps)")

    # Analyze mask bounding box
    min_x, min_y, max_x, max_y = w_orig, h_orig, 0, 0
    for fm in m_frames:
        gray = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY) if len(fm.shape) == 3 else fm
        pts = np.where(gray > 30)
        if len(pts[0]) > 0:
            min_y = min(min_y, int(np.min(pts[0])))
            max_y = max(max_y, int(np.max(pts[0])))
            min_x = min(min_x, int(np.min(pts[1])))
            max_x = max(max_x, int(np.max(pts[1])))

    mask_w = max_x - min_x
    mask_h = max_y - min_y
    is_zoom = (mode == "zoom") or (mode == "auto" and (mask_w < w_orig * 0.7 or mask_h < h_orig * 0.7))
    print(f"[+] Mask Bounding Box: x=[{min_x}, {max_x}] (w={mask_w}), y=[{min_y}, {max_y}] (h={mask_h})")
    print(f"[+] Selected Strategy: {'768p DYNAMIC ZOOM CROP' if is_zoom else 'FULL FRAME'}")

    crop_v_frames, crop_m_frames = [], []
    crop_box = None

    if is_zoom:
        # Determine fixed or tracking crop box that fits the object with margin
        crop_sz = 512
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        
        # Clamp crop box inside frame
        x1 = max(0, min(w_orig - crop_sz, cx - crop_sz // 2))
        x2 = x1 + crop_sz
        y1 = max(0, min(h_orig - crop_sz, cy - crop_sz // 2))
        y2 = y1 + crop_sz
        crop_box = (x1, y1, x2, y2)
        print(f"[+] Zoom Crop Region: x=[{x1}, {x2}], y=[{y1}, {y2}] -> Upscaling to 768x768")

        for i in range(n):
            cv_f = cv2.resize(v_frames[i][y1:y2, x1:x2], (768, 768), interpolation=cv2.INTER_LANCZOS4)
            cm_f = cv2.resize(m_frames[i][y1:y2, x1:x2], (768, 768), interpolation=cv2.INTER_NEAREST)
            if cm_f.max() < 30: cm_f[0, 0] = [255, 255, 255]
            crop_v_frames.append(cv_f)
            crop_m_frames.append(cm_f)
    else:
        for i in range(n):
            crop_v_frames.append(cv2.resize(v_frames[i], (w_orig, h_orig)))
            crop_m_frames.append(cv2.resize(m_frames[i], (w_orig, h_orig)))

    # Stage to ComfyUI input
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    tmp_s = COMFY_INPUT / "source.mp4"
    tmp_m = COMFY_INPUT / "mask.mp4"
    tmp_r = COMFY_INPUT / "reference.png"

    inp_w, inp_h = (768, 768) if is_zoom else (w_orig, h_orig)
    wr_s = cv2.VideoWriter(str(tmp_s), cv2.VideoWriter_fourcc(*'mp4v'), fps, (inp_w, inp_h))
    wr_m = cv2.VideoWriter(str(tmp_m), cv2.VideoWriter_fourcc(*'mp4v'), fps, (inp_w, inp_h))
    for cv_f, cm_f in zip(crop_v_frames, crop_m_frames):
        wr_s.write(cv_f)
        wr_m.write(cm_f)
    wr_s.release()
    wr_m.release()
    shutil.copy2(str(ref_path), str(tmp_r))

    # Configure workflow
    with open(WORKFLOW_TEMPLATE, "r", encoding="utf-8") as f:
        wf = json.load(f)

    if prompt and "reference_conditioning" in wf:
        wf["reference_conditioning"]["inputs"]["prompt"] = prompt

    print("[+] Submitting job to ComfyUI (RTX 5080)...")
    p_data = json.dumps({"prompt": wf}).encode('utf-8')
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=p_data, headers={'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
    pid = resp.get("prompt_id")
    print(f"[+] Task queued! Prompt ID: {pid}")

    start_t = time.time()
    gen_mp4 = None
    while True:
        time.sleep(5)
        try:
            h_req = urllib.request.Request(f"{COMFY_URL}/history/{pid}")
            hist = json.loads(urllib.request.urlopen(h_req).read().decode('utf-8'))
        except Exception: continue

        if pid in hist:
            status = hist[pid].get("status", {})
            if status.get("completed", False):
                print(f"[+] Inpaint finished in {time.time() - start_t:.1f}s!")
                candidate_dirs = [COMFY_OUTPUT / "h3-relay", COMFY_OUTPUT]
                for cdir in candidate_dirs:
                    if cdir.exists():
                        vids = [p for p in cdir.glob("*.mp4") if p.stat().st_mtime >= start_t - 10]
                        if vids:
                            vids.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            gen_mp4 = vids[0]
                            break
                break

    if not gen_mp4 or not gen_mp4.exists():
        print("❌ Error: Output video not found.")
        return False

    # Paste-back if zoom
    final_output_path = WS_DIR / output_name
    if is_zoom:
        print("[*] Compositing 768p zoom back to native canvas with soft Gaussian edge...")
        cap_gen = cv2.VideoCapture(str(gen_mp4))
        gen_frames = []
        while True:
            rg, fg = cap_gen.read()
            if not rg: break
            gen_frames.append(fg)
        cap_gen.release()

        x1, y1, x2, y2 = crop_box
        cw = x2 - x1
        ch = y2 - y1

        final_frames = []
        for i in range(min(n, len(gen_frames))):
            f_base = v_frames[i].copy()
            patch = cv2.resize(gen_frames[i], (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            m_patch = m_frames[i][y1:y2, x1:x2]
            gray_m = cv2.cvtColor(m_patch, cv2.COLOR_BGR2GRAY) if len(m_patch.shape) == 3 else m_patch

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_dil = cv2.dilate(gray_m, kernel, iterations=2)
            mask_soft = cv2.GaussianBlur(mask_dil.astype(np.float32) / 255.0, (11, 11), 3.0)
            mask_3ch = np.stack([mask_soft] * 3, axis=-1)

            base_patch = f_base[y1:y2, x1:x2]
            blended = (patch.astype(np.float32) * mask_3ch + base_patch.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)
            f_base[y1:y2, x1:x2] = blended
            final_frames.append(f_base)

        wr_fin = cv2.VideoWriter(str(final_output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_orig, h_orig))
        for f in final_frames: wr_fin.write(f)
        wr_fin.release()
    else:
        shutil.copy2(str(gen_mp4), str(final_output_path))

    # Copy to artifacts
    shutil.copy2(str(final_output_path), str(ART_DIR / output_name))
    print(f"🎉 SUCCESS! Final Master Video saved: {final_output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Auto-Inpaint Engine")
    parser.add_argument("--video", "-v", help="Path to input video")
    parser.add_argument("--mask", "-m", help="Path to mask video")
    parser.add_argument("--ref", "-r", help="Path to reference image")
    parser.add_argument("--mode", choices=["auto", "zoom", "full"], default="auto")
    parser.add_argument("--prompt", "-p", help="Custom text prompt")
    parser.add_argument("--output", "-o", default="Auto_Inpaint_Master.mp4", help="Output filename")

    args = parser.parse_args()
    v, m, r = auto_detect_inputs(args.video, args.mask, args.ref)
    run_pipeline(v, m, r, mode=args.mode, prompt=args.prompt, output_name=args.output)
