import cv2
import av
import numpy as np
import shutil
import os
from pathlib import Path

orig_path = r'Z:\Work\Scripts and Addons\AI_Mask\18931062672079.mp4'
mask_path = r'Z:\Work\Scripts and Addons\AI_Mask\чел маска.mp4'
out_path = r'Z:\Work\Scripts and Addons\AI_Mask\18931062672079_SideBySide.mp4'
artifact_dir = r'C:\Users\maxim\.gemini\antigravity\brain\09a95a4e-8b62-47fd-8d35-f049d3035823'

cap1 = cv2.VideoCapture(orig_path)
cap2 = cv2.VideoCapture(mask_path)

fps = cap1.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_w = w1 * 2
out_h = h1

print(f"Combining {total_frames} frames: {w1}x{h1} + {w1}x{h1} -> {out_w}x{out_h} @ {fps:.2f} fps")

container = av.open(out_path, mode='w')
stream = container.add_stream('libx264', rate=int(round(fps)))
stream.width = out_w
stream.height = out_h
stream.pix_fmt = 'yuv420p'
stream.options = {'crf': '18', 'preset': 'fast'}

preview_frames = [0, total_frames // 4, total_frames // 2, (total_frames * 3) // 4, total_frames - 1]
preview_saved = False

idx = 0
while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        break

    # If mask is grayscale / 1 channel, convert to 3 channels
    if len(f2.shape) == 2 or f2.shape[2] == 1:
        f2 = cv2.cvtColor(f2, cv2.COLOR_GRAY2BGR)

    # Ensure same height
    if f2.shape[:2] != (h1, w1):
        f2 = cv2.resize(f2, (w1, h1))

    # Combine side by side
    combined = np.hstack([f1, f2])

    # Vertical divider line
    combined[:, w1-1:w1+2] = (80, 80, 80)

    # Draw neat header banner or labels
    # Background pill for labels
    cv2.rectangle(combined, (20, 20), (220, 70), (0, 0, 0), -1)
    cv2.rectangle(combined, (20, 20), (220, 70), (255, 255, 255), 2)
    cv2.putText(combined, "ORIGINAL", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(combined, (w1 + 20, 20), (w1 + 260, 70), (0, 0, 0), -1)
    cv2.rectangle(combined, (w1 + 20, 20), (w1 + 260, 70), (0, 200, 255), 2)
    cv2.putText(combined, "MASK (SAM 2.1)", (w1 + 30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

    # Frame counter at bottom right
    f_text = f"Frame {idx}/{total_frames}"
    cv2.putText(combined, f_text, (out_w - 220, out_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Save preview image
    if idx == total_frames // 3 and not preview_saved:
        preview_img_path = os.path.join(artifact_dir, "side_by_side_preview.jpg")
        cv2.imwrite(preview_img_path, combined)
        preview_saved = True

    # Encode with PyAV
    rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
    for packet in stream.encode(frame):
        container.mux(packet)

    idx += 1
    if idx % 100 == 0:
        print(f"Processed {idx}/{total_frames} frames ({idx*100//total_frames}%)")

for packet in stream.encode():
    container.mux(packet)

container.close()
cap1.release()
cap2.release()

print(f"Done! Combined video saved to {out_path}")

# Copy to artifacts
artifact_vid = os.path.join(artifact_dir, "18931062672079_SideBySide.mp4")
shutil.copy2(out_path, artifact_vid)
print(f"Copied to artifacts: {artifact_vid}")
