"""Grab one wrist-camera frame in each video mode (to compare fields of view). Run from Terminal (camera permission)."""
import sys, time
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_check import CAM_NAME, camera_index
OUT = Path(__file__).resolve().parent / "ready_view_output" / "modes"; OUT.mkdir(parents=True, exist_ok=True)
idx = camera_index(CAM_NAME)
for w, h in [(1920, 1080), (1280, 720), (1280, 960), (800, 600), (640, 480)]:
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(25):
        ok, f = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(OUT / f"mode_{f.shape[1]}x{f.shape[0]}.jpg"), f); print("got", f.shape[1], "x", f.shape[0], "asked", w, h)
    time.sleep(0.5)
(OUT / "done").write_text("ok")
