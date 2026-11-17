"""Find an arm pose whose wrist camera sees the WHOLE chess board (real fisheye model + calibrated hand-eye +
joint offsets in the MuJoCo twin), clear of the gripper jaw, with margin, for the board placement tolerance.

    python calibration/ready_view.py validate      # overlay the predicted board grid on a real frame (ready pose)
    python calibration/ready_view.py search        # optimise, write ready_view_output/best.json + previews
"""
import json, sys, glob
from pathlib import Path
import numpy as np, cv2, mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_sim import config as C                      # noqa: E402
from chess_sim.scene import ChessScene                 # noqa: E402

OUT = Path(__file__).resolve().parent / "ready_view_output"
INTR = json.loads((Path(__file__).resolve().parent / "output/wrist_camera_intrinsics_FINAL.json").read_text())
K0, D = np.array(INTR["K"], float), np.array(INTR["dist_k1k2k3k4"], float).reshape(4, 1)
W0, H0 = INTR["image_size"]                            # 1920 x 1080
READY = json.loads((ROOT / "recording/ready_pose.json").read_text())
JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
q_ready = np.array([READY[j] for j in JN])


class View:
    def __init__(self):
        self.s = ChessScene()
        m = self.s.model
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, C.WRIST_CAM)
        # jaw / gripper mesh geoms (what can block the view)
        bodies = {m.body(n).id for n in ("gripper", "moving_jaw_so101_v1")}
        self.jaw_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                          and m.geom_rgba[g][3] > 0 and "camera" not in (m.geom(g).name or "")]

    def set(self, q):
        self.s.set_joints(q)

    def project(self, P, scale=1.0):
        """world points (N,3) -> pixel coords in a (W0*scale x H0*scale) image; also returns angle from the axis."""
        d = self.s.data
        pos, R = d.cam_xpos[self.cam], d.cam_xmat[self.cam].reshape(3, 3)
        pc = (np.asarray(P) - pos) @ R                          # camera frame (MuJoCo: x right, y up, -z forward)
        cv = np.stack([pc[:, 0], -pc[:, 1], -pc[:, 2]], 1)      # OpenCV: x right, y down, z forward
        ang = np.degrees(np.arctan2(np.linalg.norm(cv[:, :2], axis=1), cv[:, 2]))
        uv, _ = cv2.fisheye.projectPoints(cv.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K0, D)
        return uv.reshape(-1, 2) * scale, ang

    def jaw_hulls(self, scale=1.0):
        m, d = self.s.model, self.s.data
        hulls = []
        for g in self.jaw_geoms:
            mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n][:: max(1, n // 400)]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            uv, ang = self.project(w, scale)
            uv = uv[ang < 85]
            if len(uv) >= 3:
                hulls.append(cv2.convexHull(uv.astype(np.float32)))
        return hulls

    def targets(self, dx=0.0, dy=0.0, yaw=0.0, n=9):
        """Board playing surface sampled n x n (+ border ring), and piece-top points on the edge squares."""
        s = self.s
        bc, h = s.cfg.board_centre, C.BOARD_SIZE / 2 + C.BOARD_BORDER * 0.5
        c, si = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        lin = np.linspace(-h, h, n)
        pts = []
        for lx in lin:
            for ly in lin:
                pts.append([bc[0] + dx + c * lx - si * ly, bc[1] + dy + si * lx + c * ly, s.cfg.board_thickness])
        top = s.cfg.board_thickness + C.PIECE_SPECS["king"][0]       # tallest piece on the edge squares
        hs = C.BOARD_SIZE / 2 - C.SQUARE / 2
        for lx in np.linspace(-hs, hs, 8):
            for ly in (-hs, hs):
                for a, b in ((lx, ly), (ly, lx)):
                    pts.append([bc[0] + dx + c * a - si * b, bc[1] + dy + si * a + c * b, top])
        return np.array(pts)


def validate():
    v = View(); v.set(q_ready)
    frames = sorted(glob.glob(str(ROOT / "article/session_logs/*/frames/*_start.jpg")))
    img = cv2.imread(frames[-1]); h, w = img.shape[:2]; sc = w / W0
    for hull in v.jaw_hulls(sc):
        cv2.polylines(img, [hull.astype(np.int32)], True, (0, 0, 255), 2)
    s = v.s
    for f in range(9):                                       # grid lines of the playing surface
        for k in range(2):
            pts = []
            for t in np.linspace(0, 1, 30):
                a, b = (f / 8, t) if k == 0 else (t, f / 8)
                x = s.cfg.board_gap + a * C.BOARD_SIZE; y = -C.BOARD_SIZE / 2 + b * C.BOARD_SIZE
                pts.append([x, y, s.cfg.board_thickness])
            uv, ang = v.project(np.array(pts), sc)
            uv = uv[ang < 89]
            cv2.polylines(img, [uv.astype(np.int32)], False, (0, 255, 0), 1)
    for name, sq in (("a8", "a8"), ("h8", "h8"), ("a1", "a1"), ("h1", "h1")):
        uv, _ = v.project(s.square_xyz(sq)[None], sc)
        cv2.putText(img, name, tuple(int(x) for x in uv[0]), 0, 0.6, (0, 255, 255), 2)
    out = OUT / "validate_ready_pose.jpg"; cv2.imwrite(str(out), img); print("frame:", frames[-1]); print("wrote", out)




# ------------------------------------------------------------------ search
BOUNDS = np.array([[-60, 60], [-60, 25], [-90, 90], [-95, 97], [-150, 150]], float)   # LeRobot deg; wrist <= 97 (limit 103.7)
BOARD_POSES = [(0, 0, 0)] + [(dx, dy, yw) for dx in (-0.010, 0.005) for dy in (-0.020, 0.020) for yw in (-5.0, 5.0)]
MIN_CLEAR = 0.09          # jaw tips >= 9 cm above the board (king 5 cm + 4 cm)


def make_cost(v, jaw):
    s = v.s
    tg = [v.targets(*bp) for bp in BOARD_POSES]
    tips_z = lambda: min(p[2] for p in s.jaw_tip_positions()) - s.cfg.board_thickness

    def margins(q5):
        v.set(np.r_[q5, q_ready[5]])
        worst, per = 9.0, []
        for P in tg:
            uv, ang = v.project(P)
            m = np.minimum.reduce([uv[:, 0], W0 - uv[:, 0], uv[:, 1], H0 - uv[:, 1]]) / H0
            m = np.where(ang > 85, -1.0, m)
            for hull in jaw:
                inside = np.array([cv2.pointPolygonTest(hull, (float(a), float(b)), True) for a, b in uv])
                m = np.minimum(m, -inside / H0)
            per.append(m.min()); worst = min(worst, m.min())
        return worst, per

    def cost(q5):
        q5 = np.asarray(q5, float)
        pen = np.sum(np.maximum(0, BOUNDS[:, 0] - q5) + np.maximum(0, q5 - BOUNDS[:, 1])) * 0.05
        if pen > 0:
            return 1.0 + pen
        w, _ = margins(q5)
        clear = tips_z()
        pen = max(0.0, MIN_CLEAR - clear) * 20
        return -w + pen + 0.0004 * np.abs(q5 - q_ready[:5]).mean()
    return cost, margins, tips_z


def nelder_mead(f, x0, step=4.0, iters=500, tol=1e-6):
    """Plain Nelder-Mead (no scipy in this env)."""
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        if vals[-1] - vals[0] < tol and np.abs(np.array(pts) - pts[0]).max() < 0.05:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]; vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals)); return pts[i], vals[i]


def search():
    v = View(); v.set(q_ready); jaw = v.jaw_hulls()      # the jaw is rigid w.r.t. the camera (gripper closed)
    cost, margins, tips_z = make_cost(v, jaw)
    rng = np.random.default_rng(0)
    watch = np.array(json.loads((ROOT / "chess_sim/assets/watch_pose.json").read_text())["lerobot"][:5])
    starts = [q_ready[:5], watch]
    cand = [q_ready[:5] + rng.normal(0, 15, 5) for _ in range(1500)] + \
           [BOUNDS[:, 0] + rng.random(5) * (BOUNDS[:, 1] - BOUNDS[:, 0]) for _ in range(1500)] + starts
    prog = ROOT / "data/progress.txt"
    scored = []
    for i, q in enumerate(cand):
        scored.append((cost(q), q))
        if i % 200 == 0:
            prog.write_text(f"[{'#' * int(30 * i / len(cand)):<30}] {100 * i / len(cand):5.1f}%  sampling {i}/{len(cand)}\n")
    scored.sort(key=lambda t: t[0])
    best = None
    for k, (c0, q) in enumerate(scored[:12]):
        x, fx = nelder_mead(cost, q)
        prog.write_text(f"[{'#' * int(30 * (k + 1) / 12):<30}] {100 * (k + 1) / 12:5.1f}%  refining {k + 1}/12  best cost {min(fx, best[0] if best else 9):.4f}\n")
        if best is None or fx < best[0]:
            best = (fx, x)
    q5 = best[1]; w, per = margins(q5)
    q = np.r_[q5, q_ready[5]]
    res = dict(lerobot=[round(float(x), 2) for x in q], worst_margin_frac_of_height=round(float(w), 4),
               nominal_margin=round(float(per[0]), 4), jaw_tip_clearance_m=round(float(tips_z()), 4),
               change_from_old_ready=[round(float(x), 1) for x in q - q_ready], board_poses=BOARD_POSES)
    (OUT / "best.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    prog.write_text("[##############################] 100.0%  done\n")
    preview(v, q, jaw, "best"); preview(v, q_ready, jaw, "old_ready")


def preview(v, q, jaw, tag):
    """Sim render of the wrist view (undistorted pinhole) + fisheye projection diagram of board/jaw."""
    v.set(q)
    img = np.full((H0 // 3, W0 // 3, 3), 255, np.uint8); sc = 1 / 3
    for hull in jaw:
        cv2.fillPoly(img, [(hull * sc).astype(np.int32)], (200, 200, 230))
    s = v.s
    for bp, col in zip(BOARD_POSES, [(0, 140, 0)] + [(160, 200, 160)] * 8):
        P = v.targets(*bp, n=2)[:4][[0, 1, 3, 2]]
        uv, _ = v.project(P, sc)
        cv2.polylines(img, [uv.astype(np.int32)], True, col, 2 if bp == (0, 0, 0) else 1)
    for sq in ("a1", "h1", "a8", "h8"):
        uv, _ = v.project(s.square_xyz(sq)[None], sc); cv2.putText(img, sq, tuple(int(x) for x in uv[0]), 0, .5, (0, 0, 0), 1)
    cv2.imwrite(str(OUT / f"diagram_{tag}.png"), img)
    try:
        r = s.render(C.WRIST_CAM, 640, 360); cv2.imwrite(str(OUT / f"sim_{tag}.png"), cv2.cvtColor(r, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print("render failed:", e)


if __name__ == "__main__":
    {"validate": validate, "search": search}[sys.argv[1] if len(sys.argv) > 1 else "validate"]()
