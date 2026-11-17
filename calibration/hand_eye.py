"""Hand-eye calibration of the SO-101 wrist camera (where the camera sits on the gripper).

Run from Terminal (camera permission):  python calibration/hand_eye.py
Ctrl+C stops at once; the arm then HOLDS its position (torque stays on).

The iPad lies flat and still on the table showing calibration/checkerboard_ipad.png (9x6 squares of 24 mm).
1. Search: from safe high poses the arm looks to its right and left of the board for the pattern; if it is not
   found anywhere, the arm returns to the watch pose and the script stops ("not placed properly").
2. Plans ~14 poses around the first detection that keep the pattern in view with varied wrist angles. Every move is
   checked in the sim first: >= 3 mm from the board / pieces / itself and >= 20 mm above the table (the sim does
   not know the iPad is there). At most 30 deg/s per joint.
3. At each pose: hold, photograph, detect the pattern, record the joint angles.
4. Solves AX = XB (OpenCV calibrateHandEye, several methods) for the camera pose in the gripper frame, reports how
   well the methods agree, and saves calibration/hand_eye_output/hand_eye.json (OpenCV and MuJoCo camera frames).
5. Returns to the watch pose and holds.
"""

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_check import JOINTS, PORT, ROBOT_ID, CAM_NAME, INTR, HZ, camera_index, segment, path_check  # noqa: E402

OUT = Path(__file__).resolve().parent / "hand_eye_output"
LOG = OUT / "hand_eye_log.txt"
PATTERN, SQUARE = (6, 4), 0.015   # checkerboard_ipad_small.png: 7x5 squares of 15 mm (6x4 inner corners)
TABLE_CLEARANCE, CLEARANCE = 0.020, 0.003
N_POSES = 18


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def table_clearance(scene, path):
    """Smallest distance between any arm geom and the table along the path."""
    import mujoco
    m, d = scene.model, scene.data
    arm_names = ["upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
    arm_names += [m.body(b).name for b in range(m.nbody) if "camera" in m.body(b).name]
    arm = {m.body(n).id for n in arm_names}
    table = m.geom("table").id
    arm_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm and (m.geom_contype[g] or m.geom_conaffinity[g])]
    worst, fromto = 9.0, np.zeros(6)
    for k in range(0, len(path), 3):
        scene.set_joints(path[k])
        for g in arm_g:
            worst = min(worst, mujoco.mj_geomDistance(m, d, g, table, 0.05, fromto))
    return worst


def safe(scene, path):
    c = path_check(scene, path)[0]
    t = table_clearance(scene, path)
    return c >= CLEARANCE and t >= TABLE_CLEARANCE, c, t


class Detector:
    def __init__(self):
        intr = json.loads(INTR.read_text())
        self.K = np.array(intr["K"])
        self.D = np.array(intr["dist_k1k2k3k4"]).reshape(4, 1)
        self.obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float64)
        self.obj[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE

    def detect(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, None, fx=0.5, fy=0.5)
        ok, c = cv2.findChessboardCorners(small, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
        c = cv2.cornerSubPix(g, (c * 2).astype(np.float32), (11, 11), (-1, -1), (3, 100, 1e-4))
        und = cv2.fisheye.undistortPoints(c.reshape(-1, 1, 2).astype(np.float64), self.K, self.D)
        ok, rvec, tvec = cv2.solvePnP(self.obj, und, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(self.obj, rvec, tvec, np.eye(3), None)
        err_px = float(np.sqrt((((proj - und) * self.K[0, 0]) ** 2).sum(-1).mean()))
        return dict(rvec=rvec, tvec=tvec, err_px=err_px, centre_px=c.reshape(-1, 2).mean(0).tolist(),
                    und=und.reshape(-1, 2))


def gripper_pose(scene, q):
    scene.set_joints(q)
    b = scene.model.body("gripper").id
    return scene.data.xmat[b].reshape(3, 3).copy(), scene.data.xpos[b].copy()


PRIOR = None  # camera-in-gripper (+ joint zero offsets) from a previous fit, used only to PREDICT views
PRIOR_OFF = np.zeros(6)
_pp = Path(__file__).resolve().parent / "hand_eye_output" / "prior.json"
if _pp.exists():
    _p = json.loads(_pp.read_text())
    PRIOR = (np.array(_p["R_cam2gripper_opencv"]), np.array(_p["t_cam2gripper_m"]))
    PRIOR_OFF = np.concatenate([_p.get("joint_offsets_deg", [0] * 5), [0.0]])


def cam_pose_cv(scene, q):
    """Wrist camera pose (world) in the OpenCV convention (z forward, y down): from the previous hand-eye result
    when available, else from the sim's mount."""
    if PRIOR is not None:
        Rg, tg = gripper_pose(scene, np.asarray(q, float) + PRIOR_OFF)
        return Rg @ PRIOR[0], Rg @ PRIOR[1] + tg
    scene.set_joints(q)
    c = scene.model.camera("wrist_cam").id
    R = scene.data.cam_xmat[c].reshape(3, 3) @ np.diag([1, -1, -1])
    return R, scene.data.cam_xpos[c].copy()


class Arm:
    def __init__(self):
        from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
        self.robot = SOFollower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
        self.bus = self.robot.bus
        self.bus.connect()
        if not self.robot.calibration or not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            raise SystemExit("follower calibration mismatch")
        raw = self.bus.sync_read("Present_Position", normalize=False, num_retry=3)
        self.bus.sync_write("Goal_Position", raw, normalize=False)  # hold where it is, no jump
        self.robot.configure()

    def q(self):
        p = self.bus.sync_read("Present_Position", num_retry=3)
        return np.array([p[j] for j in JOINTS], float)

    def follow(self, path):
        for qk in path:
            t0 = time.perf_counter()
            self.bus.sync_write("Goal_Position", {j: float(qk[n]) for n, j in enumerate(JOINTS)})
            time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t0)))

    def close(self):
        self.bus.disconnect(disable_torque=False)


def grab(cap):
    for _ in range(8):
        cap.read()
    ok, frame = cap.read()
    return frame if ok else None


def main():
    OUT.mkdir(exist_ok=True)
    LOG.write_text("")
    import chess
    from chess_sim.scene import ChessScene
    scene = ChessScene()
    scene.set_position(chess.STARTING_FEN)
    views = [np.array(v["lerobot"], float) for v in json.loads((ROOT / "chess_sim/assets/watch_pose_scan.json").read_text())["views"]]
    watch = views[0]
    det = Detector()

    cap = cv2.VideoCapture(camera_index(CAM_NAME), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(30):
        if cap.read()[0]:
            break
        time.sleep(0.5)
    arm = Arm()
    log(f"connected; arm at {np.round(arm.q(), 1).tolist()}")

    def go(target, label):
        path = segment(arm.q(), target)
        ok, c, t = safe(scene, path)
        if not ok:
            log(f"  skip {label}: sim clearance {c*1000:.1f} mm to obstacles, {t*1000:.1f} mm to the table")
            return False
        arm.follow(path)
        time.sleep(1.2)
        return True

    samples = []
    try:
        # ---------------------------------------------------------------- 1. search
        drop = views[-1]
        search = [watch]
        for pan in (35, 45, 57, -35, -57):  # the iPad was seen at +35 / +57 (robot right) in the first search
            for lift, elbow, flex in ((-41, 41, 77), (-25, 20, 85)):
                roll = drop[4] if pan > 0 else -drop[4]
                search.append(np.array([pan, lift, elbow, flex, roll, drop[5]], float))
        found = None
        for i, pose in enumerate(search):
            if not go(pose, f"search pose {i}"):
                continue
            frame = grab(cap)
            if frame is None:
                continue
            cv2.imwrite(str(OUT / f"search{i}.jpg"), cv2.resize(frame, (960, 540)))
            d = det.detect(frame)
            log(f"search pose {i} (pan {pose[0]:+.0f}): {'PATTERN FOUND' if d else 'not found'}")
            if d:
                found = (arm.q(), d, frame)
                break
        if not found:
            log("iPad pattern NOT FOUND from any search pose: not placed where the arm can see it. Returning to watch pose.")
            go(watch, "watch")
            return

        # ---------------------------------------------------------------- 2. plan poses around the detection
        q0, d0, frame0 = found
        Rc, tc = cam_pose_cv(scene, q0)  # approximate (the mount is what we are calibrating)
        Rt, _ = cv2.Rodrigues(d0["rvec"])
        corners = np.array([[0, 0, 0], [(PATTERN[0] - 1) * SQUARE, 0, 0], [0, (PATTERN[1] - 1) * SQUARE, 0], [(PATTERN[0] - 1) * SQUARE, (PATTERN[1] - 1) * SQUARE, 0]])
        world_corners = (Rc @ (Rt @ corners.T + d0["tvec"])).T + tc
        centre = world_corners.mean(0)
        log(f"pattern centre (approx, world): {np.round(centre, 3).tolist()} m, height {centre[2]*100:.1f} cm")
        if centre[2] > 0.06:
            log("pattern estimated well above the table: the iPad may not be lying flat. Stopping for safety.")
            go(watch, "watch")
            return
        rng = np.random.default_rng(0)
        cands = []
        for _ in range(8000):
            dq = np.array([rng.uniform(-22, 22), rng.uniform(-18, 18), rng.uniform(-20, 20), rng.uniform(-22, 22),
                           rng.uniform(-110, 110), 0.0])
            q = q0 + dq
            R, t = cam_pose_cv(scene, q)
            pc = (R.T @ (world_corners - t).T).T
            if (pc[:, 2] < 0.12).any() or (pc[:, 2] > 0.45).any():
                continue
            u = det.K[0, 0] * pc[:, 0] / pc[:, 2] + det.K[0, 2]
            v = det.K[1, 1] * pc[:, 1] / pc[:, 2] + det.K[1, 2]
            if (u < 100).any() or (u > 1050).any() or (v < 100).any() or (v > 980).any():
                continue  # keep it in the left ~60 % (the jaw covers the right) with margin
            Rg, _ = gripper_pose(scene, q)
            cands.append((q, Rg))
        log(f"{len(cands)} candidate poses keep the pattern in the predicted view")
        chosen = [(q0, gripper_pose(scene, q0)[0])]
        while len(chosen) < N_POSES + 4 and cands:  # greedy: maximise rotation diversity
            def mind(Rg):
                return min(np.degrees(np.arccos(np.clip((np.trace(Rg.T @ Rc2) - 1) / 2, -1, 1))) for _, Rc2 in chosen)
            best = max(range(len(cands)), key=lambda i: mind(cands[i][1]))
            chosen.append(cands.pop(best))
        poses = [q for q, _ in chosen]

        # ---------------------------------------------------------------- 3. collect
        samples.append((q0, d0))
        cv2.imwrite(str(OUT / "pose00.jpg"), cv2.resize(frame0, (960, 540)))
        for i, q in enumerate(poses[1:], 1):
            if len(samples) >= N_POSES:
                break
            if not go(q, f"pose {i}"):
                continue
            frame = grab(cap)
            d = det.detect(frame) if frame is not None else None
            qm = arm.q()
            if frame is not None:
                cv2.imwrite(str(OUT / f"pose{i:02d}.jpg"), cv2.resize(frame, (960, 540)))
            if d and d["err_px"] < 1.5:
                samples.append((qm, d))
                log(f"pose {i}: pattern found (PnP error {d['err_px']:.2f} px) -> {len(samples)} samples")
            else:
                log(f"pose {i}: {'pattern not found' if not d else 'poor detection %.1f px' % d['err_px']}")
        go(watch, "watch")
        log("back at the watch pose, holding")
    except KeyboardInterrupt:
        log("stopped by user: holding position (torque on)")
    finally:
        cap.release()
        arm.close()

    # ---------------------------------------------------------------- 4. solve
    if len(samples) < 6:
        log(f"only {len(samples)} good samples: not enough for a hand-eye solution (need >= 6)")
        return
    Rg2b, tg2b, Rt2c, tt2c = [], [], [], []
    for q, d in samples:
        R, t = gripper_pose(scene, q)
        Rg2b.append(R); tg2b.append(t.reshape(3, 1))
        Rt2c.append(cv2.Rodrigues(d["rvec"])[0]); tt2c.append(d["tvec"])
    results = {}
    for name, meth in (("tsai", cv2.CALIB_HAND_EYE_TSAI), ("park", cv2.CALIB_HAND_EYE_PARK),
                       ("horaud", cv2.CALIB_HAND_EYE_HORAUD), ("daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS)):
        R, t = cv2.calibrateHandEye(Rg2b, tg2b, Rt2c, tt2c, method=meth)
        results[name] = (R, t.ravel())
    Rs = np.array([r for r, _ in results.values()]); ts = np.array([t for _, t in results.values()])
    R_ref, t_ref = results["park"]
    spread_t = float(np.linalg.norm(ts - ts.mean(0), axis=1).max() * 1000)
    spread_r = max(float(np.degrees(np.arccos(np.clip((np.trace(R_ref.T @ r) - 1) / 2, -1, 1)))) for r in Rs)
    # consistency: the target's world pose from every sample should be the same
    tw = []
    for (q, d), Rg, tg in zip(samples, Rg2b, tg2b):
        Rcw = Rg @ R_ref; tcw = Rg @ t_ref + tg.ravel()
        tw.append(Rcw @ d["tvec"].ravel() + tcw)
    tw = np.array(tw); target_spread = float(np.linalg.norm(tw - tw.mean(0), axis=1).max() * 1000)
    # the sim camera mount for comparison
    sim_R, sim_t = None, None
    import mujoco
    c = scene.model.camera("wrist_cam").id
    sim_t = scene.model.cam_pos[c].copy()
    q4 = scene.model.cam_quat[c]; sim_R = np.zeros(9); mujoco.mju_quat2Mat(sim_R, q4); sim_R = sim_R.reshape(3, 3) @ np.diag([1, -1, -1])
    dpos = float(np.linalg.norm(t_ref - sim_t) * 1000)
    dang = float(np.degrees(np.arccos(np.clip((np.trace(sim_R.T @ R_ref) - 1) / 2, -1, 1))))
    quat_mj = np.zeros(4); mujoco.mju_mat2Quat(quat_mj, (R_ref @ np.diag([1, -1, -1])).ravel())
    out = {"n_samples": len(samples), "frame": "camera pose in the 'gripper' body frame",
           "R_cam2gripper_opencv": R_ref.tolist(), "t_cam2gripper_m": t_ref.tolist(),
           "mujoco_cam_pos": t_ref.tolist(), "mujoco_cam_quat_wxyz": quat_mj.tolist(),
           "method_spread_mm": spread_t, "method_spread_deg": spread_r,
           "target_world_consistency_mm": target_spread,
           "vs_current_sim_mount": {"position_diff_mm": dpos, "angle_diff_deg": dang},
           "methods": {k: {"R": v[0].tolist(), "t": v[1].tolist()} for k, v in results.items()}}
    # save raw samples so the result can be re-analysed without moving the arm
    (OUT / "samples.json").write_text(json.dumps([{"joints": [float(x) for x in q], "und": d["und"].tolist()}
                                                  for q, d in samples], indent=1))
    # refinement: jointly fit camera-in-gripper (6) + target-in-world (6) + zero offsets of the 5 arm joints (5)
    # by minimising the corner reprojection error over all samples (Levenberg-Marquardt, numerical Jacobian).
    # The joint offsets absorb LeRobot's calibration zero (middle of the swept range) differing from the model zero.
    fit_offsets = len(samples) >= 12
    def rt(v):
        return cv2.Rodrigues(v[:3].reshape(3, 1))[0], v[3:6]
    Rw0 = Rg2b[0] @ R_ref @ Rt2c[0]; tw0 = Rg2b[0] @ (R_ref @ tt2c[0].ravel() + t_ref) + tg2b[0].ravel()
    x = np.concatenate([cv2.Rodrigues(R_ref)[0].ravel(), t_ref, cv2.Rodrigues(Rw0)[0].ravel(), tw0, np.zeros(5)])
    npar = 17 if fit_offsets else 12
    x = x[:npar]
    f = det.K[0, 0]
    def resid(x):
        Rc, tc = rt(x[:6]); Rw, tw = rt(x[6:12])
        off = np.concatenate([x[12:17], [0.0]]) if fit_offsets else np.zeros(6)
        r = []
        P = (Rw @ det.obj.T).T + tw
        for (q, d) in samples:
            Rg, tg = gripper_pose(scene, np.asarray(q) + off)
            Pg = (Rg.T @ (P - tg).T).T
            Pc = (Rc.T @ (Pg - tc).T).T
            r.append(((Pc[:, :2] / Pc[:, 2:3]) - d["und"]).ravel() * f)
        r = np.concatenate(r)
        if fit_offsets:  # weak prior: offsets stay small unless the data needs them
            r = np.concatenate([r, x[12:17] * 0.5])
        return r
    lam = 1e-2
    for it in range(80):
        r0 = resid(x); J = np.zeros((r0.size, npar))
        for k in range(npar):
            dx = np.zeros(npar); dx[k] = 1e-5 if k >= 12 else 1e-6
            J[:, k] = (resid(x + dx) - r0) / dx[k]
        H = J.T @ J; g = J.T @ r0
        step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
        if (resid(x + step) ** 2).sum() < (r0 ** 2).sum():
            x = x + step; lam = max(lam * 0.3, 1e-7)
        else:
            lam *= 10
        if np.abs(step).max() < 1e-8 or lam > 1e8:
            break
    if fit_offsets:
        out["joint_zero_offsets_deg"] = dict(zip(JOINTS[:5], np.round(x[12:17], 2).tolist()))
        log(f"joint zero offsets (deg): {out['joint_zero_offsets_deg']}")
    rf = (resid(x)[:-5] if fit_offsets else resid(x)).reshape(len(samples), -1, 2)
    per = np.sqrt((rf ** 2).sum(-1).mean(-1))
    Rc, tc = rt(x[:6])
    quat_r = np.zeros(4); mujoco.mju_mat2Quat(quat_r, (Rc @ np.diag([1, -1, -1])).ravel())
    out["refined"] = {"R_cam2gripper_opencv": Rc.tolist(), "t_cam2gripper_m": tc.tolist(),
                      "mujoco_cam_pos": tc.tolist(), "mujoco_cam_quat_wxyz": quat_r.tolist(),
                      "rms_px": float(np.sqrt((rf ** 2).sum(-1).mean())), "per_sample_rms_px": per.round(2).tolist(),
                      "vs_current_sim_mount": {"position_diff_mm": float(np.linalg.norm(tc - sim_t) * 1000),
                                               "angle_diff_deg": float(np.degrees(np.arccos(np.clip((np.trace(sim_R.T @ Rc) - 1) / 2, -1, 1))))}}
    (OUT / "hand_eye.json").write_text(json.dumps(out, indent=2))
    log(f"REFINED: reprojection RMS {out['refined']['rms_px']:.2f} px over {len(samples)} samples "
        f"(worst sample {per.max():.2f} px); mount differs from the sim by "
        f"{out['refined']['vs_current_sim_mount']['position_diff_mm']:.1f} mm / {out['refined']['vs_current_sim_mount']['angle_diff_deg']:.1f} deg")
    log(f"HAND-EYE: {len(samples)} samples; methods agree within {spread_t:.1f} mm / {spread_r:.2f} deg; "
        f"target consistency {target_spread:.1f} mm; real mount differs from the sim by {dpos:.1f} mm and {dang:.1f} deg")


if __name__ == "__main__":
    main()
