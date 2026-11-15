import os
import sys
import json
import time
import subprocess
from pathlib import Path

SAMMIE_DIR = Path(r"Z:\Work\VFX_Program\Sammie-Roto-2-main")
TEMP_DIR = SAMMIE_DIR / "temp"
IPC_CMD_FILE = TEMP_DIR / "sammie_ipc_command.json"
RUN_BAT = SAMMIE_DIR / "run_sammie.bat"

def is_sammie_running():
    try:
        from PySide6.QtCore import QLockFile, QDir
        lock_path = QDir.tempPath() + "/sammie-roto.lock"
        lock = QLockFile(lock_path)
        if lock.tryLock(50):
            lock.unlock()
            return False
        return True
    except Exception:
        return False

def open_video(video_path: str):
    vpath = Path(video_path).resolve()
    if not vpath.exists():
        print(f"Error: Video file not found: {vpath}")
        return False

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    running = is_sammie_running()

    if running:
        print(f"[open_in_sammie] Sammie-Roto is already running. Sending IPC command to load: {vpath.name}")
        cmd = {
            "action": "load_file",
            "file": str(vpath)
        }
        with open(IPC_CMD_FILE, "w", encoding="utf-8") as f:
            json.dump(cmd, f)
        print("[open_in_sammie] Video sent! Sammie-Roto should switch to this file in a moment.")
    else:
        print(f"[open_in_sammie] Starting Sammie-Roto with video: {vpath.name}")
        ps_cmd = f"$s = New-Object -ComObject Shell.Application; $s.ShellExecute('{str(RUN_BAT)}', '\"\"\"{str(vpath)}\"\"\"', '{str(SAMMIE_DIR)}', 'open', 1)"
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False)
        print("[open_in_sammie] Launched Sammie-Roto on desktop.")
    return True

def get_latest_mask(output_mask_path: str = None):
    """
    Extracts the current propagated mask from Sammie-Roto's temp/masks folder
    into a black-and-white video file using cv2.
    """
    masks_dir = TEMP_DIR / "masks"
    if not masks_dir.exists():
        print("Error: No temp/masks directory found.")
        return None

    session_conf = TEMP_DIR / "session_settings.conf"
    fps = 24.0
    if session_conf.exists():
        try:
            with open(session_conf, "r", encoding="utf-8") as f:
                sdata = json.load(f)
                fps = sdata.get("video_fps", 24.0)
        except Exception:
            pass

    frame_dirs = sorted([d for d in masks_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not frame_dirs:
        print("Error: No mask frames found in temp/masks.")
        return None

    if output_mask_path is None:
        output_mask_path = str(Path(r"Z:\Work\Scripts and Addons\AI_Mask") / "sammie_exported_mask.mp4")

    import cv2
    first_mask_file = frame_dirs[0] / "0.png"
    if not first_mask_file.exists():
        print(f"Error: Mask frame not found at {first_mask_file}")
        return None

    img0 = cv2.imread(str(first_mask_file), cv2.IMREAD_GRAYSCALE)
    h, w = img0.shape[:2]

    # Try mp4v or avc1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_mask_path, fourcc, fps, (w, h), isColor=False)

    for fd in frame_dirs:
        mf = fd / "0.png"
        if mf.exists():
            fimg = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        else:
            fimg = np.zeros((h, w), dtype=np.uint8)
        writer.write(fimg)

    writer.release()
    print(f"Mask successfully compiled: {output_mask_path} ({len(frame_dirs)} frames @ {fps} fps)")
    return output_mask_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--get-mask":
            out = sys.argv[2] if len(sys.argv) > 2 else None
            get_latest_mask(out)
        else:
            open_video(arg)
    else:
        # Default: open Pass 1 Table Master
        default_vid = r"Z:\Work\Scripts and Addons\AI_Mask\Courtyard_Pass1_Table_Master.mp4"
        open_video(default_vid)
