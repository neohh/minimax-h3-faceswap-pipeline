import sys
import os
import json

def choose_file_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Выберите видео для загрузки в Sammie-Roto",
            filetypes=[
                ("Видео и изображения", "*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg *.webp"),
                ("Все файлы", "*.*")
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"Error opening file dialog: {e}")
        return None

def send_file(target_file):
    if not target_file:
        return False
    target_file = os.path.abspath(target_file)
    if not os.path.exists(target_file):
        print(f"Файл не найден: {target_file}")
        return False

    ipc_file = r"Z:\Work\VFX_Program\Sammie-Roto-2-main\temp\sammie_ipc_command.json"
    os.makedirs(os.path.dirname(ipc_file), exist_ok=True)

    cmd = {
        "action": "load_file",
        "file": target_file
    }
    with open(ipc_file, "w", encoding="utf-8") as f:
        json.dump(cmd, f)
    print(f"[OK] Видео отправлено в Sammie-Roto: {os.path.basename(target_file)}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].strip():
        fpath = sys.argv[1].strip().strip('"')
    else:
        fpath = choose_file_gui()

    if fpath:
        send_file(fpath)
