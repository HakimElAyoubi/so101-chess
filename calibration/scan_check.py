"""Drive the real follower through the sim's 7 board-scan poses and compare the wrist camera with the sim.

Run from Terminal (camera permission):  python calibration/scan_check.py
Ctrl+C stops at once; the arm then HOLDS its position (torque stays on).

1. Plans joint-space moves (current pose -> view 0 -> ... -> view 6), at most 30 deg/s per joint, min-jerk timing.
2. Checks the whole path in the sim (start position, nominal board) for contact of the arm with the board, the
   pieces, the table or itself. Aborts before moving if anything comes within 3 mm.
3. At each view: holds 1.5 s, photographs (1920x1080), undistorts with the final calibration, renders the sim's
   wrist camera at the arm's MEASURED joint angles, and saves both side by side to calibration/scan_output/.
4. Ends holding the last view's... no: ends back at view 0 and keeps torque on (the arm does not drop).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "scan_output"
LOG = OUT / "scan_log.txt"
PORT, ROBOT_ID = so101_config.FOLLOWER_PORT, so101_config.FOLLOWER_ID
CAM_NAME = "Innomaker-U20CAM-1080p-S1"
INTR = Path(__file__).resolve().parent / "output" / "wrist_camera_intrinsics_FINAL.json"
SPEED, HZ, CLEARANCE = 30.0, 30, 0.003
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def segment(q0, q1):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    span = np.abs(q1[:5] - q0[:5]).max()
    T = max(1.0, 1.875 * span / SPEED)  # min-jerk peak velocity = 1.875 * span / T
    n = int(math.ceil(T * HZ))
    return [q0 + (q1 - q0) * minjerk(i / n) for i in range(1, n + 1)]


def camera_index(name):
    out = subprocess.run([str(Path(sys.prefix) / "bin" / "ffmpeg"), "-hide_banner", "-f", "avfoundation",
                          "-list_devices", "true", "-i", ""], capture_output=True, text=True).stderr
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
        raise SystemExit(f"camera '{name}' not found: {names}")
    return names.index(name)


def path_check(scene, path):
    """Smallest distance between arm geoms and anything else (board, pieces, table, non-adjacent arm links)."""
    import mujoco
    m, d = scene.model, scene.data
    arm_names = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
    arm_names += [m.body(b).name for b in range(m.nbody) if "camera" in m.body(b).name]  # wrist camera mount = part of the arm
    arm = {m.body(b).id for b in arm_names}
    chain = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
    adjacent = {frozenset((m.body(a).id, m.body(b).id)) for a, b in zip(chain, chain[1:])}
    adjacent.add(frozenset((m.body("gripper").id, m.body("moving_jaw_so101_v1").id)))
    worst = (9.0, "", -1)
    initial = {}  # contacts already present at the start pose (e.g. the folded rest pose touching itself)
    for k, q in enumerate(path):
        scene.set_joints(q)
        mujoco.mj_collision(m, d)
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if not ({b1, b2} & arm) or frozenset((b1, b2)) in adjacent or b1 == b2:
                continue
            key = frozenset((c.geom1, c.geom2))
            if k == 0:
                initial[key] = min(initial.get(key, 9.0), float(c.dist))
                continue
            if key in initial and c.dist >= initial[key] - 0.001:
                continue  # pre-existing contact that is not getting worse
            if c.dist < worst[0]:
                worst = (float(c.dist), f"{m.body(b1).name} / {m.body(b2).name}", k)
    # mj_collision only reports touching pairs, so also measure true distances arm <-> board / pieces / table
    coll = lambda g: m.geom_contype[g] or m.geom_conaffinity[g]
    arm_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm and coll(g)]
    env_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in arm and coll(g)
             and m.body(m.geom_bodyid[g]).name not in chain + arm_names]
    fromto = np.zeros(6)
    for k in range(0, len(path), 3):
        scene.set_joints(path[k])
        for ga in arm_g:
            for ge in env_g:
                dist = mujoco.mj_geomDistance(m, d, ga, ge, 0.02, fromto)
                if dist < worst[0]:
                    worst = (float(dist), f"{m.body(m.geom_bodyid[ga]).name} / {m.body(m.geom_bodyid[ge]).name or m.geom(ge).name}", k)
    return worst


def main():
    OUT.mkdir(exist_ok=True)
    LOG.write_text("")
    import chess
    from chess_sim.scene import ChessScene

    scene = ChessScene()
    scene.set_position(chess.STARTING_FEN)
    views = [np.array(v["lerobot"], float) for v in json.loads((ROOT / "chess_sim/assets/watch_pose_scan.json").read_text())["views"]]
    log(f"{len(views)} scan views loaded")

    # read the arm's current pose (torque still off), without moving it
    from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
    robot = SOFollower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    bus = robot.bus
    bus.connect()
    if not robot.calibration or not bus.is_calibrated:
        bus.disconnect(disable_torque=False)
        raise SystemExit("follower calibration mismatch")
    present = bus.sync_read("Present_Position", num_retry=3)
    q_now = np.array([present[j] for j in JOINTS], float)
    log(f"current pose: {np.round(q_now, 1).tolist()}")

    plan, marks = [], []
    q = q_now
    for i, v in enumerate(views + [views[0]]):
        seg = segment(q, v)
        plan += seg
        marks.append(len(plan) - 1)
        q = v
    # also keep the gripper where it is (closed) on the way to view 0
    worst = path_check(scene, plan)
    log(f"sim path check: {len(plan)} steps, closest approach {worst[0]*1000:.1f} mm ({worst[1]}) at step {worst[2]}")
    if worst[0] < CLEARANCE:
        bus.disconnect(disable_torque=False)
        raise SystemExit(f"ABORT before moving: the path comes within {worst[0]*1000:.1f} mm ({worst[1]}).")
    log(f"total motion time ~{len(plan)/HZ:.0f} s + holds")

    intr = json.loads(INTR.read_text())
    K = np.array(intr["K"])
    D = np.array(intr["dist_k1k2k3k4"]).reshape(4, 1)
    mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (1920, 1080), cv2.CV_16SC2)
    cap = cv2.VideoCapture(camera_index(CAM_NAME), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(30):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.5)
    if not ok:
        bus.disconnect(disable_torque=False)
        raise SystemExit("camera gave no frames")

    # torque on without a jump, then follow the plan
    raw = bus.sync_read("Present_Position", normalize=False, num_retry=3)
    bus.sync_write("Goal_Position", raw, normalize=False)
    robot.configure()
    log("torque on, moving")
    view_i = 0
    try:
        for k, qk in enumerate(plan):
            t0 = time.perf_counter()
            bus.sync_write("Goal_Position", {j: float(qk[n]) for n, j in enumerate(JOINTS)})
            if k in marks and view_i < len(views):
                time.sleep(1.5)
                for _ in range(8):
                    cap.read()  # flush buffered frames
                ok, frame = cap.read()
                meas = bus.sync_read("Present_Position", num_retry=3)
                qm = [meas[j] for j in JOINTS]
                err = np.abs(np.array(qm[:5]) - views[view_i][:5]).max()
                real = cv2.remap(frame, mx, my, cv2.INTER_LINEAR) if ok else np.zeros((1080, 1920, 3), np.uint8)
                scene.set_joints(qm)
                sim = cv2.cvtColor(scene.render("wrist_cam", 960, 540), cv2.COLOR_RGB2BGR)
                both = np.hstack([cv2.resize(real, (960, 540)), sim])
                cv2.putText(both, f"view {view_i}  real (undistorted)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(both, "sim at measured joints", (980, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imwrite(str(OUT / f"view{view_i}.jpg"), both)
                if ok:
                    cv2.imwrite(str(OUT / f"view{view_i}_raw.jpg"), frame)
                log(f"view {view_i}: reached (max joint error {err:.1f} deg), photo saved")
                view_i += 1
            time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t0)))
        log("done: back at view 0, holding (torque on)")
    except KeyboardInterrupt:
        log("stopped by user: holding position (torque on)")
    finally:
        cap.release()
        bus.disconnect(disable_torque=False)  # keep holding; power the arm off or run teleop to release


if __name__ == "__main__":
    main()
