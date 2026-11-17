"""Move the follower slowly to a pose, hold it, grab a real wrist-camera frame and overlay the predicted board grid.

    python calibration/show_pose.py 12.1 -67.4 35.3 101.0 23.6 1.07 [--name candidate]

Run from Terminal (camera permission). Min-jerk <= 20 deg/s; torque comes on holding the current pose; the follower
keeps holding the target afterwards (torque on). Output: calibration/ready_view_output/real_<name>.jpg (+ _overlay).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, cv2
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root
import so101_config  # noqa: E402  (ports and Stockfish path from config.yaml)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "ready_view_output"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("q", type=float, nargs=6); ap.add_argument("--name", default="candidate")
    a = ap.parse_args()
    import subprocess
    busy = subprocess.run(["pgrep", "-f", "web_recorder|chess_recorder|lerobot-record|lerobot-teleoperate|shake_test"],
                          capture_output=True, text=True).stdout.split()
    if busy:
        raise SystemExit("The recorder (or another arm program) is running and owns the follower. Close it first.")
    from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
    from scan_check import CAM_NAME, camera_index
    cap = cv2.VideoCapture(camera_index(CAM_NAME))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    robot = SOFollower(SO101FollowerConfig(port=so101_config.FOLLOWER_PORT, id=so101_config.FOLLOWER_ID, position_p_coefficient=32))
    bus = robot.bus; bus.connect()
    here = bus.sync_read("Present_Position"); bus.sync_write("Goal_Position", here); robot.configure()
    q0, q1 = np.array([here[j] for j in J]), np.array(a.q)
    T = max(1.5, 1.875 * np.abs(q1 - q0)[:5].max() / 20.0); n = int(T * 40)
    print(f"moving {np.round(q0, 1)} -> {np.round(q1, 1)} in {T:.0f} s")
    for i in range(1, n + 1):
        s = i / n; k = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        bus.sync_write("Goal_Position", dict(zip(J, q0 + (q1 - q0) * k))); time.sleep(1 / 40)
        if i % 40 == 0:
            print(f"  {100 * i / n:3.0f}%", flush=True)
    time.sleep(1.5)
    p = bus.sync_read("Present_Position"); print("reached", np.round([p[j] for j in J], 1))
    for _ in range(30):
        ok, frame = cap.read()
    cap.release()
    bus.disconnect(disable_torque=False)                        # keep holding the pose
    if not ok:
        raise SystemExit("no camera frame")
    OUT.mkdir(exist_ok=True); cv2.imwrite(str(OUT / f"real_{a.name}.jpg"), frame)
    # overlay the predicted board grid (actual measured joints)
    import ready_view as rv
    v = rv.View(); v.set(np.array([p[j] for j in J])); img = frame.copy(); h, w = img.shape[:2]; sc = w / rv.W0
    for hull in v.jaw_hulls(sc):
        cv2.polylines(img, [hull.astype(np.int32)], True, (0, 0, 255), 3)
    s_ = v.s
    for f in range(9):
        for kk in range(2):
            pts = [[s_.cfg.board_gap + ((f / 8, t) if kk == 0 else (t, f / 8))[0] * rv.C.BOARD_SIZE,
                    -rv.C.BOARD_SIZE / 2 + ((f / 8, t) if kk == 0 else (t, f / 8))[1] * rv.C.BOARD_SIZE, s_.cfg.board_thickness]
                   for t in np.linspace(0, 1, 40)]
            uv, ang = v.project(np.array(pts), sc); uv = uv[ang < 89]
            cv2.polylines(img, [uv.astype(np.int32)], False, (0, 255, 0), 2)
    cv2.imwrite(str(OUT / f"real_{a.name}_overlay.jpg"), img)
    print("saved", OUT / f"real_{a.name}.jpg")
    (OUT / f"real_{a.name}.done").write_text("ok")


if __name__ == "__main__":
    main()
