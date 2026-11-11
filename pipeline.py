"""
MiniMax H3 Head-Crop + Rock-Solid InSwapper Pipeline
===================================================
Cinematic actor replacement and face swapping on complex 3D head turns & downward tilts.

Workflow:
1. Tight Head Crop (centered on user's rotoscoped mask + hair room, scaled to 768x768).
2. MiniMax H3 Generation (directly guided by user mask, SAM3 bypassed, expansion 8px).
3. Rock-Solid InSwapper (zero-drop detection, linear interpolation, 5-frame landmark smoothing).
4. Sub-pixel Native Composite (downscaled back to crop box, pasted at original coordinates with foreground protection).
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


def step1_prepare_crop(source_video, mask_video, start_frame=0, num_frames=56, crop_box=None, comfy_input_dir=r"C:\Work\docker\ComfyUI\input"):
    """
    Extracts a tight crop around the user's mask (+ hair room) and scales to 768x768.
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
    for _ in range(num_frames):
        r1, fs = cap_s.read()
        r2, fm = cap_m.read()
        if not r1: break
        frames_s.append(fs)
        frames_m.append(fm if r2 else np.zeros((h_orig, w_orig, 3), dtype=np.uint8))
    cap_s.release()
    cap_m.release()
    
    if crop_box is None:
        min_x, min_y, max_x, max_y = 10000, 10000, -1, -1
        for fm in frames_m:
            gray = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY)
            ys, xs = np.where(gray > 20)
            if len(xs) > 0:
                min_x = min(min_x, np.min(xs))
                max_x = max(max_x, np.max(xs))
                min_y = min(min_y, np.min(ys))
                max_y = max(max_y, np.max(ys))
        
        cx = int((min_x + max_x) / 2)
        cy = int((min_y + max_y) / 2)
        sz = 384
        x1 = max(0, cx - sz // 2)
        y1 = max(0, cy - sz // 2)
        crop_box = (x1, y1, sz)
    
    x1, y1, sz = crop_box
    x2, y2 = x1 + sz, y1 + sz
    print(f"Crop box: x=[{x1}:{x2}], y=[{y1}:{y2}], size={sz}x{sz} in {w_orig}x{h_orig} canvas")
    
    tmp_s_raw = os.path.join(comfy_input_dir, "temp_crop_s_raw.mp4")
    tmp_m_raw = os.path.join(comfy_input_dir, "temp_crop_m_raw.mp4")
    
    wr_s = cv2.VideoWriter(tmp_s_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (768, 768))
    wr_m = cv2.VideoWriter(tmp_m_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (768, 768))
    
    for fs, fm in zip(frames_s, frames_m):
        c_s = cv2.resize(fs[y1:y2, x1:x2], (768, 768), interpolation=cv2.INTER_LANCZOS4)
        c_m = cv2.resize(fm[y1:y2, x1:x2], (768, 768), interpolation=cv2.INTER_NEAREST)
        wr_s.write(c_s)
        wr_m.write(c_m)
    wr_s.release()
    wr_m.release()
    
    os.system("docker exec comfyui ffmpeg -y -i /opt/comfyui/input/temp_crop_s_raw.mp4 -c:v libx264 -pix_fmt yuv420p /opt/comfyui/input/source.mp4")
    os.system("docker exec comfyui ffmpeg -y -i /opt/comfyui/input/temp_crop_m.mp4 -c:v libx264 -pix_fmt yuv420p /opt/comfyui/input/mask.mp4")
    print("Prepared /opt/comfyui/input/source.mp4 and mask.mp4")
    return crop_box, (w_orig, h_orig), fps


def step2_generate_minimax_h3(workflow_json, comfy_url="http://127.0.0.1:8188", comfy_output_dir=r"C:\Work\docker\ComfyUI\output\h3-relay", prefix="h3-relay/crop_run"):
    """
    Submits prompt to ComfyUI and waits for output.
    """
    print("\n[Step 2] Queuing MiniMax H3 generation...")
    with open(workflow_json, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)
        
    prompt_dict["save"]["inputs"]["filename_prefix"] = prefix
    seed = random.randint(100000, 999999999)
    prompt_dict["noise"]["inputs"]["noise_seed"] = seed
    
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


def step3_rock_solid_inswapper(input_video, ref_image, output_swapped_video, insightface_root=r"C:\Work\docker\ComfyUI\models\insightface", inswapper_onnx=r"C:\Work\docker\ComfyUI\models\insightface\inswapper_128.onnx"):
    """
    Executes anti-drop InSwapper with temporal landmark smoothing.
    """
    print("\n[Step 3] Running Rock-Solid InSwapper...")
    app = insightface.app.FaceAnalysis(name='buffalo_l', root=insightface_root)
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.35)
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
            
    print(f"Tracking coverage: {det_mask.count(True)}/{len(frames)} frames detected.")
    
    # Zero-drop linear interpolation
    for i in range(len(frames)):
        if not det_mask[i]:
            p = next((k for k in range(i-1, -1, -1) if det_mask[k]), None)
            n = next((k for k in range(i+1, len(frames)) if det_mask[k]), None)
            if p is not None and n is not None:
                w_f = (i - p) / (n - p)
                raw_kps[i] = raw_kps[p] * (1.0 - w_f) + raw_kps[n] * w_f
                raw_bbox[i] = raw_bbox[p] * (1.0 - w_f) + raw_bbox[n] * w_f
            elif p is not None:
                raw_kps[i] = raw_kps[p].copy()
                raw_bbox[i] = raw_bbox[p].copy()
            elif n is not None:
                raw_kps[i] = raw_kps[n].copy()
                raw_bbox[i] = raw_bbox[n].copy()
                
    # 5-frame moving-average temporal smoothing
    smooth_kps, smooth_bbox = [], []
    for i in range(len(frames)):
        s_i = max(0, i - 2)
        e_i = min(len(frames), i + 3)
        smooth_kps.append(np.mean([raw_kps[k] for k in range(s_i, e_i)], axis=0))
        smooth_bbox.append(np.mean([raw_bbox[k] for k in range(s_i, e_i)], axis=0))
        
    class FakeFace:
        def __init__(self, kps, bbox):
            self.kps = kps
            self.bbox = bbox
            
    tmp_raw = os.path.join(os.path.dirname(output_swapped_video), "temp_swap_raw.mp4")
    out = cv2.VideoWriter(tmp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    swapped_frames = []
    for i in range(len(frames)):
        ff = FakeFace(smooth_kps[i], smooth_bbox[i])
        swapped = swapper.get(frames[i], ff, source_face, paste_back=True)
        out.write(swapped)
        swapped_frames.append(swapped)
    out.release()
    
    tmp_h264 = os.path.join(os.path.dirname(output_swapped_video), "temp_swap_h264.mp4")
    os.system(f"docker exec comfyui ffmpeg -y -i /opt/comfyui/output/temp_swap_raw.mp4 -c:v libx264 -pix_fmt yuv420p /opt/comfyui/output/temp_swap_h264.mp4")
    shutil.copy2(tmp_h264, output_swapped_video)
    return swapped_frames


def step4_pixel_perfect_composite(source_video, mask_video, tubes_matte, swapped_frames, crop_box, output_master_video, start_frame=0):
    """
    Sub-pixel alignment: pastes 384x384 back into exact native coordinates.
    """
    print("\n[Step 4] Pixel-perfect native compositing...")
    cap_full = cv2.VideoCapture(source_video)
    cap_mask = cv2.VideoCapture(mask_video)
    cap_tubes = cv2.VideoCapture(tubes_matte) if tubes_matte else None
    
    cap_full.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_mask.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fps = cap_full.get(cv2.CAP_PROP_FPS) or 24.0
    w_full = int(cap_full.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full = int(cap_full.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    x1, y1, sz = crop_box
    x2, y2 = x1 + sz, y1 + sz
    
    tmp_comp_raw = os.path.join(os.path.dirname(output_master_video), "temp_comp_raw.mp4")
    out = cv2.VideoWriter(tmp_comp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_full, h_full))
    
    kernel = np.ones((25, 25), np.uint8)
    
    for i in range(len(swapped_frames)):
        r1, ff = cap_full.read()
        r2, fm = cap_mask.read()
        if not r1: break
        clean_bg = ff.copy()
        
        fc_native = cv2.resize(swapped_frames[i], (sz, sz), interpolation=cv2.INTER_AREA)
        
        fm_box = fm[y1:y2, x1:x2, 0]
        fm_dilated = cv2.dilate(fm_box, kernel, iterations=1)
        fm_smooth = cv2.GaussianBlur(fm_dilated.astype(np.float32) / 255.0, (25, 25), 0)
        m_3d = np.repeat(fm_smooth[:, :, np.newaxis], 3, axis=2)
        
        roi = ff[y1:y2, x1:x2].astype(np.float32)
        ff[y1:y2, x1:x2] = (fc_native.astype(np.float32) * m_3d + roi * (1.0 - m_3d)).astype(np.uint8)
        
        if cap_tubes:
            r3, ft = cap_tubes.read()
            if r3:
                tm_gray = cv2.cvtColor(ft, cv2.COLOR_BGR2GRAY)
                if tm_gray.shape[:2] != (h_full, w_full):
                    tm_gray = cv2.resize(tm_gray, (w_full, h_full))
                m_tubes = (tm_gray.astype(np.float32) / 255.0)[:, :, np.newaxis]
                ff = (clean_bg.astype(np.float32) * m_tubes + ff.astype(np.float32) * (1.0 - m_tubes)).astype(np.uint8)
                
        out.write(ff)
        
    cap_full.release()
    cap_mask.release()
    if cap_tubes: cap_tubes.release()
    out.release()
    
    tmp_comp_h264 = os.path.join(os.path.dirname(output_master_video), "temp_comp_h264.mp4")
    os.system(f"docker exec comfyui ffmpeg -y -i /opt/comfyui/output/temp_comp_raw.mp4 -c:v libx264 -pix_fmt yuv420p /opt/comfyui/output/temp_comp_h264.mp4")
    shutil.copy2(tmp_comp_h264, output_master_video)
    print(f"Master output rendered: {output_master_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMax H3 Head-Crop + InSwapper Pipeline")
    parser.add_argument("--source", required=True, help="Path to source driving video")
    parser.add_argument("--mask", required=True, help="Path to user rotoscoped face mask video")
    parser.add_argument("--ref", required=True, help="Path to character reference image")
    parser.add_argument("--matte", default=None, help="Optional foreground occlusion matte (e.g. tubes)")
    parser.add_argument("--output", default="final_master.mp4", help="Path to output master video")
    args = parser.parse_args()
    
    box, dims, fps = step1_prepare_crop(args.source, args.mask)
    h3_gen = step2_generate_minimax_h3("workflow/minimax_h3_crop_workflow.json")
    swapped = step3_rock_solid_inswapper(h3_gen, args.ref, "temp_swapped_crop.mp4")
    step4_pixel_perfect_composite(args.source, args.mask, args.matte, swapped, box, args.output)
