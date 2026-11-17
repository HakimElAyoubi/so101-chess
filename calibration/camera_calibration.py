"""Wrist-camera calibration for the SO-101 (Innomaker U20CAM) using the empty chessboard as the target.

Run from Terminal (it needs camera permission):  python calibration/camera_calibration.py

Remove all pieces. The follower stays limp (torque off): move it by hand so the wrist camera sees the board from
many angles and distances, and press Space whenever the board is outlined in green. Aim for 15-25 photos:
board in the centre, near each image corner and edge, tilted, close and far. Press C to compute, Q to quit.

Outputs (calibration/output/):
  wrist_camera_intrinsics.json  pinhole (k1..k3, p1, p2) and fisheye (k1..k4) models, RMS reprojection
                                error, horizontal / vertical / diagonal field of view
  captures/NN.png + captures.json  every photo with the follower's joint angles at that moment
  undistorted_preview.png       a photo before/after correction
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root
import so101_config  # noqa: E402  (ports and Stockfish path from config.yaml)

CAM_NAME = "Innomaker-U20CAM-1080p-S1"
W, H = 1920, 1080
# Target: the printed flat checkerboard (calibration/checkerboard_25mm_A4.pdf, 9x6 squares of 25 mm = 8x5 inner corners).
# Run with  --board  to use the chessboard instead (8x8 squares of 28 mm); it must be perfectly flat.
PATTERN = (8, 5)
SQUARE_M = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--square-mm=")), 25)) / 1000
if "--board" in sys.argv:
    PATTERN, SQUARE_M = (7, 7), 0.028
FOLLOWER_PORT = so101_config.FOLLOWER_PORT
CALIB_FILE = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm_1.json"
OUT = Path(__file__).resolve().parent / "output"
WIN = "Wrist camera calibration"


def camera_index(name):
    ffmpeg = Path(sys.prefix) / "bin" / "ffmpeg"
    out = subprocess.run([str(ffmpeg), "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True).stderr
    names, inside = [], False
    for line in out.splitlines():
        if "video devices" in line:
            inside = True
        elif "audio devices" in line:
            break
        elif inside and "] [" in line:
            n = line.split("] ", 2)[-1].strip()
            if not n.startswith("Capture screen"):
                names.append(n)
    if name not in names:
        raise SystemExit(f"Camera '{name}' not found. Connected: {names}")
    return names.index(name)


class Follower:
    """Reads joint angles (deg, LeRobot convention) without enabling torque."""

    def __init__(self):
        self.ph = None
        try:
            import scservo_sdk as scs
            self.cal = json.loads(CALIB_FILE.read_text())
            self.ph = scs.PortHandler(FOLLOWER_PORT)
            if not (self.ph.openPort() and self.ph.setBaudRate(1_000_000)):
                self.ph = None
            self.pk = scs.PacketHandler(0)
        except Exception as e:
            print("follower not readable (photos will have no joint angles):", e)
            self.ph = None

    def joints(self):
        if not self.ph:
            return None
        out = {}
        for n, c in self.cal.items():
            pos, comm, _ = self.pk.read2ByteTxRx(self.ph, c["id"], 56)
            if comm != 0:
                return None
            if n == "gripper":
                out[n] = round((pos - c["range_min"]) / (c["range_max"] - c["range_min"]) * 100, 2)
            else:
                out[n] = round((pos - (c["range_min"] + c["range_max"]) / 2) * 360 / 4095, 2)
        return out


def find_corners(gray_small, scale):
    # Classic detector: findChessboardCornersSB segfaulted on live frames (OpenCV 4.13, macOS arm64).
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(np.ascontiguousarray(gray_small), PATTERN, flags)
    return (corners / scale) if ok else None


def fov_deg(f, size):
    return math.degrees(2 * math.atan(size / (2 * f)))


def calibrate(obj_list, img_list):
    res = {}
    # pinhole + radial/tangential
    rms, K, D, _, _ = cv2.calibrateCamera(obj_list, img_list, (W, H), None, None,
                                          flags=cv2.CALIB_RATIONAL_MODEL * 0)
    res["pinhole"] = {"rms_px": rms, "K": K.tolist(), "dist_k1k2p1p2k3": D.ravel().tolist()}
    # fisheye (equidistant), usually better for wide-angle lenses
    try:
        objf = [o.reshape(-1, 1, 3).astype(np.float64) for o in obj_list]
        imgf = [i.reshape(-1, 1, 2).astype(np.float64) for i in img_list]
        Kf, Df = np.zeros((3, 3)), np.zeros((4, 1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rmsf, Kf, Df, _, _ = cv2.fisheye.calibrate(objf, imgf, (W, H), Kf, Df, flags=flags,
                                                     criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-7))
        res["fisheye"] = {"rms_px": rmsf, "K": Kf.tolist(), "dist_k1k2k3k4": Df.ravel().tolist()}
    except cv2.error as e:
        res["fisheye"] = {"error": str(e)[:200]}
    best = min((k for k in res if "rms_px" in res[k]), key=lambda k: res[k]["rms_px"])
    K = np.array(res[best]["K"])
    fx, fy = K[0, 0], K[1, 1]
    if best == "pinhole":
        hfov, vfov = fov_deg(fx, W), fov_deg(fy, H)
        dfov = fov_deg(math.hypot(fx, fy) / math.sqrt(2), math.hypot(W, H))
    else:  # equidistant: r = f * theta
        hfov, vfov = math.degrees(W / 2 / fx) * 2, math.degrees(H / 2 / fy) * 2
        dfov = math.degrees(math.hypot(W, H) / 2 / ((fx + fy) / 2)) * 2
    res["best_model"] = best
    res["fov_deg"] = {"horizontal": round(hfov, 1), "vertical": round(vfov, 1), "diagonal": round(dfov, 1)}
    res["image_size"] = [W, H]
    res["n_images"] = len(obj_list)
    return res


def undistort(img, res):
    m = res[res["best_model"]]
    K = np.array(m["K"])
    if res["best_model"] == "fisheye":
        D = np.array(m["dist_k1k2k3k4"]).reshape(4, 1)
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), np.eye(3), balance=0.5)
        mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (W, H), cv2.CV_16SC2)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR)
    D = np.array(m["dist_k1k2p1p2k3"])
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0.5)
    return cv2.undistort(img, K, D, None, newK)


def text(img, s, y, col=(255, 255, 255), scale=0.9):
    cv2.putText(img, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


def main():
    import shutil
    if (OUT / "captures").exists() and any((OUT / "captures").iterdir()):  # keep earlier sessions
        shutil.move(str(OUT / "captures"), str(OUT / f"captures_{time.strftime('%Y%m%d_%H%M%S')}"))
        if (OUT / "captures.json").exists():
            shutil.move(str(OUT / "captures.json"), str(OUT / f"captures_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    (OUT / "captures").mkdir(parents=True, exist_ok=True)
    idx = camera_index(CAM_NAME)
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    for _ in range(30):  # first-run permission prompt / warm-up
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.5)
    if not ok:
        raise SystemExit("Camera did not deliver frames (check permission for Terminal).")
    if frame.shape[1] != W:
        raise SystemExit(f"Camera gave {frame.shape[1]}x{frame.shape[0]}, expected {W}x{H}.")
    follower = Follower()
    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_M
    obj_list, img_list, meta = [], [], []
    coverage = np.zeros((H // 8, W // 8), np.uint8)
    scale, result, flash, last_dump = 0.5, None, 0, 0.0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        corners = find_corners(small, scale)
        if time.time() - last_dump > 2:  # lets the assistant see the live view
            cv2.imwrite(str(OUT / "live.jpg"), cv2.resize(frame, (960, 540)))
            last_dump = time.time()
        view = frame.copy()
        cov = cv2.resize(coverage, (W, H), interpolation=cv2.INTER_NEAREST)
        view[cov > 0] = (0.7 * view[cov > 0] + 0.3 * np.array([0, 160, 0])).astype(np.uint8)
        if corners is not None:
            cv2.drawChessboardCorners(view, PATTERN, corners, True)
        if flash:
            cv2.rectangle(view, (0, 0), (W - 1, H - 1), (255, 255, 255), 30)
            flash -= 1
        text(view, f"photos: {len(obj_list)}  (aim for 15-25; green tint = image areas already covered)", 45)
        text(view, "board found - press SPACE" if corners is not None else "board not found: move the arm / add light",
             90, (80, 255, 80) if corners is not None else (60, 60, 255))
        text(view, "SPACE photo   C compute   Q quit", H - 30, (220, 220, 220), 0.8)
        if result:
            f = result["fov_deg"]
            text(view, f"{result['best_model']}: RMS {result[result['best_model']]['rms_px']:.2f} px  FOV h {f['horizontal']} "
                       f"v {f['vertical']} d {f['diagonal']} deg  (saved)", 135, (0, 220, 255))
        cv2.imshow(WIN, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" ") and corners is not None:
            refined = cv2.cornerSubPix(gray, corners.astype(np.float32), (7, 7), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
            n = len(obj_list)
            cv2.imwrite(str(OUT / "captures" / f"{n:02d}.png"), frame)
            obj_list.append(obj.copy())
            img_list.append(refined)
            meta.append({"file": f"{n:02d}.png", "joints_deg": follower.joints(), "time": time.strftime("%H:%M:%S")})
            hull = cv2.convexHull(refined.reshape(-1, 2) / 8).astype(np.int32)
            cv2.fillConvexPoly(coverage, hull, 1)
            (OUT / "captures.json").write_text(json.dumps(meta, indent=2))
            flash = 4
            print(f"photo {n}: joints {meta[-1]['joints_deg']}", flush=True)
        if key == ord("c"):
            if len(obj_list) < 8:
                print("need at least 8 photos", flush=True)
                continue
            print("computing...", flush=True)
            result = calibrate(obj_list, img_list)
            result["pattern_inner_corners"], result["square_m"] = list(PATTERN), SQUARE_M
            (OUT / "wrist_camera_intrinsics.json").write_text(json.dumps(result, indent=2))
            sample = cv2.imread(str(OUT / "captures" / "00.png"))
            both = np.hstack([cv2.resize(sample, (960, 540)), cv2.resize(undistort(sample, result), (960, 540))])
            cv2.imwrite(str(OUT / "undistorted_preview.png"), both)
            print(json.dumps({k: result[k] for k in ("best_model", "fov_deg", "n_images")}), flush=True)
            for m in ("pinhole", "fisheye"):
                print(m, "RMS px:", result[m].get("rms_px"), flush=True)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
