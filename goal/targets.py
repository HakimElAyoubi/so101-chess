"""Where are the move's squares? Shared by dataset building AND the live player, so training and deployment match.

- board pose per recording session: goal/board_poses.json (board_calib.py, fitted from the demos' own grasps)
- target_xyz(task, session): world xyz of the 'from' and 'to' targets (drop zone / spare zone: measured from demos)
- Projector: projects world points into the wrist image for a given joint state (calibrated twin + real fisheye lens)
- draw_markers(img, q, task, session): green dot = piece to take, magenta ring = where it goes
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
POSES = json.loads((Path(__file__).parent / "board_poses.json").read_text())
INTR = json.loads((ROOT / "calibration/output/wrist_camera_intrinsics_FINAL.json").read_text())
PAT = re.compile(r"(?:from|on) ([a-h][1-8]) to (?:the )?([a-h][1-8]|drop zone)")
SPARE = re.compile(r"place the spare \w+ \w+ on ([a-h][1-8])")
PIECE_MID_Z = 0.020          # aim the 'from' marker at mid-piece height, the 'to' marker at the board surface
LATEST = max((k for k in POSES["sessions"] if k.isdigit()), key=int)


def session_of(ep):
    return str(POSES["episodes"].get(str(ep), LATEST))


CORR_SIGMA = 0.045    # local correction map: Gaussian radius (m, ~1.6 squares) and smoothing (absorbs ~0.5 mm of centring noise)
CORR_SMOOTH = 0.05


def local_correction(corr, xy):
    """Offset (m) the session's correction map adds at world xy: Gaussian-RBF interpolation of what the board model
    still missed at the squares centred by hand (exact there up to the smoothing, fading to 0 ~2 squares away)."""
    c = np.asarray(corr["xy"], float); a = np.asarray(corr["alpha"], float)
    w = np.exp(-((c - np.asarray(xy, float)[:2]) ** 2).sum(1) / (2 * corr["sigma"] ** 2))
    return w @ a


HEIGHT_SIGMA = 0.07  # height map: Gaussian radius (m, 2.5 squares); a height error varies slowly over the board


def local_height(h, xy):
    """How much higher (m) the board top is at world xy than the model's flat board, in the arm's forward kinematics,
    from a session's 'height' fit of touch-offs: Gaussian-RBF interpolation, fading to 0 (the model) away from the
    touched squares, never extrapolated (a plane through three touch-offs on ranks 7-8 put h1 14 mm low). Only for
    targets that must be physically exact (the calibration pointing): the hover templates and the skill's dot come from
    demos on this same arm in its own forward kinematics, its height errors included, so they don't take it."""
    if not h.get("alpha"):
        return 0.0                                    # an old plane-only fit (2026-10-28): not used
    xy, c = np.asarray(xy, float)[:2], np.asarray(h["xy"], float)
    return float(np.exp(-((c - xy) ** 2).sum(1) / (2 * h["sigma"] ** 2)) @ np.asarray(h["alpha"], float))


class Projector:
    def __init__(self):
        import mujoco
        from chess_sim import config as C
        from chess_sim.scene import ChessScene
        self.C, self.mj = C, mujoco
        self.s = ChessScene()
        self.cam = mujoco.mj_name2id(self.s.model, mujoco.mjtObj.mjOBJ_CAMERA, C.WRIST_CAM)
        self.K0 = np.array(INTR["K"], float)
        self.D = np.array(INTR["dist_k1k2k3k4"], float).reshape(4, 1)
        self.W0 = INTR["image_size"][0]

    # ---- board geometry for a session
    def square_xyz(self, sq, session, z=0.0):
        return self.board_point("abcdefgh".index(sq[0]), int(sq[1]), session, z)   # nominal geometry: pose-free whatever
                                                                                  # pose the twin's own board body has

    def board_point(self, f, r, session, z=0.0):
        """Any point of the board in continuous board coordinates (file f: 0 = a .. 7 = h, rank r: 1 .. 8; square
        centres at whole numbers, the board's edges at -0.5 / 7.5 and 0.5 / 8.5), as square_xyz places it."""
        from chess_sim import config as CC
        nom = np.array([self.s.cfg.board_gap + (8.5 - r) * CC.SQUARE, (f - 3.5) * CC.SQUARE])
        return self.board_xyz(nom, session, z)

    def board_xyz(self, nom, session, z=0.0):
        """World xyz of a point given on the nominal (unmoved) board: the session's pose, board size and correction map."""
        p = POSES["sessions"][session]
        bc = self.s.cfg.board_centre[:2]
        a = np.radians(p["yaw_deg"])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        size = np.array([p.get("scale_rank") or 1.0, p.get("scale_file") or 1.0])   # board size in the arm's frame
        xy = R @ ((nom - bc) * size) + bc + np.array([p["dx"], p["dy"]])
        if p.get("correction"):                       # what the rigid + size model can't: the arm's own errors over the board
            xy = xy + local_correction(p["correction"], xy)
        return np.array([xy[0], xy[1], self.s.cfg.board_thickness + z])

    def board_dz(self, session, xy):
        """The measured board top under the arm at xy relative to the model's board (m; touch-offs, see local_height)."""
        h = POSES["sessions"][session].get("height")
        return local_height(h, xy) if h else 0.0

    def drop_xyz(self, session):
        p = POSES["sessions"][session].get("drop_xy")
        if p is None:                         # nominal drop-zone centre shifted with the board
            c = self.s.drop_cell_xyz(1, 1)[:2]
            p = c + np.array([POSES["sessions"][session]["dx"], POSES["sessions"][session]["dy"]])
        return np.array([p[0], p[1], self.s.cfg.board_thickness])

    def target_xyz(self, task, session):
        """(from_xyz, to_xyz) in the arm's base frame (metres)."""
        m = PAT.search(task)
        if m:
            frm, to = m.groups()
            f = self.square_xyz(frm, session, PIECE_MID_Z)
            t = self.drop_xyz(session) if to == "drop zone" else self.square_xyz(to, session)
            return f, t
        m = SPARE.search(task)
        if m:
            sp = POSES["sessions"][session].get("spare_xy") or (self.s.spare_xyz("black")[:2]).tolist()
            return np.array([sp[0], sp[1], self.s.cfg.board_thickness + PIECE_MID_Z]), self.square_xyz(m.group(1), session)
        raise ValueError(f"unknown task format: {task}")

    # ---- camera projection
    def project(self, q6, pts, width):
        """World points -> pixel coordinates in an image `width` px wide (16:9). Returns (uv, visible mask)."""
        self.s.set_joints(q6)
        d = self.s.data
        pos, R = d.cam_xpos[self.cam], d.cam_xmat[self.cam].reshape(3, 3)
        pc = (np.asarray(pts) - pos) @ R
        cv = np.stack([pc[:, 0], -pc[:, 1], -pc[:, 2]], 1)
        ang = np.degrees(np.arctan2(np.linalg.norm(cv[:, :2], axis=1), cv[:, 2]))
        uv, _ = cv2.fisheye.projectPoints(cv.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), self.K0, self.D)
        uv = uv.reshape(-1, 2) * (width / self.W0)
        h = width * 9 / 16
        vis = (ang < 80) & (uv[:, 0] > -50) & (uv[:, 0] < width + 50) & (uv[:, 1] > -50) & (uv[:, 1] < h + 50)
        return uv, vis

    def backproject(self, q6, uv, width, z=None):
        """Pixel(s) in a `width`-px 16:9 image -> world xy on the horizontal plane at height z (default: the board's
        top plane) by ray / plane intersection with the real fisheye model."""
        self.s.set_joints(q6)
        d = self.s.data
        pos, R = d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3).copy()
        px = np.asarray(uv, np.float64).reshape(-1, 1, 2) * (self.W0 / width)
        n = cv2.fisheye.undistortPoints(px, self.K0, self.D).reshape(-1, 2)          # normalised (x/z, y/z), OpenCV axes
        rays_cv = np.column_stack([n, np.ones(len(n))])
        rays = np.column_stack([rays_cv[:, 0], -rays_cv[:, 1], -rays_cv[:, 2]]) @ R.T  # MuJoCo camera axes -> world
        z0 = self.s.cfg.board_thickness if z is None else float(z)
        t = (z0 - pos[2]) / rays[:, 2]
        return (pos + rays * t[:, None])[:, :2]

    def corner_xy_nominal(self, corner):
        """Nominal (unmoved board) world xy of a playing-surface corner: 'a8', 'h8', 'h1' or 'a1'."""
        c = self.s.square_xyz(corner)[:2]
        bc = self.s.cfg.board_centre[:2]
        return bc + (c - bc) * (8 / 7)          # square centre -> outer corner (4 squares vs 3.5 squares from the centre)

    def fit_board(self, clicks):
        """clicks: {corner: world xy}. Rigid 2-D fit of the nominal board onto the clicked corners.
        Returns dict(dx, dy, yaw_deg, rms_mm, per_corner_mm) in the format of board_poses.json sessions."""
        names = list(clicks)
        A = np.array([self.corner_xy_nominal(k) for k in names]); B = np.array([clicks[k] for k in names])
        bc = self.s.cfg.board_centre[:2]
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        U, _, Vt = np.linalg.svd(Ac.T @ Bc); Rm = (U @ Vt).T
        if np.linalg.det(Rm) < 0:
            Rm = np.eye(2)
        d = B.mean(0) - (Rm @ (A.mean(0) - bc)) - bc
        res = np.linalg.norm((A - bc) @ Rm.T + bc + d - B, axis=1)
        return dict(dx=float(d[0]), dy=float(d[1]), yaw_deg=float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))),
                    rms_mm=float(np.sqrt((res ** 2).mean()) * 1000), per_corner_mm={k: round(float(r) * 1000, 1) for k, r in zip(names, res)})

    def fit_squares(self, points, prior):
        """points: {square: world xy of its real centre} (where the arm's closed jaws were centred by hand).
        Fits dx, dy, yaw and the board's size along ranks / files (scale_rank / scale_file) in the arm's frame. Weak
        priors keep yaw and the sizes at `prior` (a session dict) where the points don't determine them (1 point: only
        the position moves; a6 + h6: no rank size). Returns a session dict + per_square_mm residuals."""
        bc = self.s.cfg.board_centre[:2]
        names = list(points)
        A = np.array([self.s.square_xyz(k)[:2] for k in names]) - bc; B = np.array([points[k] for k in names]) - bc
        s0 = np.array([prior.get("scale_rank") or 1.0, prior.get("scale_file") or 1.0])
        W_SIZE, W_YAW = 0.03, 0.0003          # prior weights: 1 % of size or 1 deg cost as much as 0.3 mm at a square
        best = None
        for yaw in prior["yaw_deg"] + np.arange(-10, 10.001, 0.02):
            a = np.radians(yaw); Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            C = B @ Rm                        # the points in the yaw-rotated board frame: C = size * A + e
            size, e, cost = np.zeros(2), np.zeros(2), (W_YAW * (yaw - prior["yaw_deg"])) ** 2
            for k in range(2):                # per axis: [A 1] [size e]^T = C, plus the size prior row
                M = np.vstack([np.column_stack([A[:, k], np.ones(len(A))]), [W_SIZE, 0.0]])
                y = np.append(C[:, k], W_SIZE * s0[k])
                x = np.linalg.lstsq(M, y, rcond=None)[0]; size[k], e[k] = x
                cost += float(((M @ x - y) ** 2).sum())
            if best is None or cost < best[0]:
                best = (cost, yaw, size.copy(), Rm @ e)
        _, yaw, size, d = best
        a = np.radians(yaw); Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        res = np.linalg.norm((A * size) @ Rm.T + d - B, axis=1)
        return dict(dx=float(d[0]), dy=float(d[1]), yaw_deg=float(yaw), scale_rank=float(size[0]), scale_file=float(size[1]),
                    per_square_mm={k: round(float(r) * 1000, 1) for k, r in zip(names, res)})

    def fit_correction(self, points, fit):
        """Correction map for a fit_squares() result: the residual vectors at the centred squares, interpolated
        (local_correction). None below 3 squares (then the model alone is exact anyway)."""
        if len(points) < 3:
            return None
        bc = self.s.cfg.board_centre[:2]; a = np.radians(fit["yaw_deg"])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        names = list(points)
        A = np.array([self.s.square_xyz(k)[:2] for k in names]) - bc
        model = (A * [fit["scale_rank"], fit["scale_file"]]) @ R.T + bc + [fit["dx"], fit["dy"]]
        E = np.array([points[k] for k in names]) - model
        K = np.exp(-((model[:, None] - model[None]) ** 2).sum(2) / (2 * CORR_SIGMA ** 2))
        alpha = np.linalg.solve(K + CORR_SMOOTH * np.eye(len(names)), E)
        corr = dict(xy=model.tolist(), alpha=alpha.tolist(), sigma=CORR_SIGMA, squares=names)
        left = np.linalg.norm(model + np.array([local_correction(corr, m) for m in model]) - np.array([points[k] for k in names]), axis=1)
        corr["left_mm"] = {k: round(float(r) * 1000, 1) for k, r in zip(names, left)}
        return corr

    def fit_height(self, touches):
        """touches: {square: [x, y, z]} world positions where the fingertips touched the board (FK of the measured
        joints). Their heights above the model's board, interpolated (local_height). A 'height' entry for a session."""
        names = list(touches)
        X = np.array([touches[k][:2] for k in names], float); dz = np.array([touches[k][2] for k in names]) - self.s.cfg.board_thickness
        K = np.exp(-((X[:, None] - X[None]) ** 2).sum(2) / (2 * HEIGHT_SIGMA ** 2))
        return dict(xy=X.tolist(), alpha=np.linalg.solve(K + CORR_SMOOTH * np.eye(len(X)), dz).tolist(), sigma=HEIGHT_SIGMA,
                    squares=names, dz_mm={k: round(float(v) * 1000, 1) for k, v in zip(names, dz)})

    def draw_markers(self, img, q6, task, session, which=("from", "to"), to_xyz=None):
        """Draw in place on an RGB HxWx3 image: green filled dot on the piece to take, magenta ring on the target.
        which: the markers to draw (the skill policies see only their own: grasp -> "from", place -> "to").
        to_xyz: put the ring here instead (drop zone: where the piece was really released / the free spot chosen live)."""
        f, t = self.target_xyz(task, session)
        if to_xyz is not None:
            t = np.asarray(to_xyz, float)
        uv, vis = self.project(q6, np.stack([f, t]), img.shape[1])
        r = max(6, int(img.shape[1] / 90))
        if vis[1] and "to" in which:
            c = tuple(int(v) for v in uv[1])
            cv2.circle(img, c, r + 3, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(img, c, r + 3, (255, 0, 255), 2, cv2.LINE_AA)
        if vis[0] and "from" in which:
            c = tuple(int(v) for v in uv[0])
            cv2.circle(img, c, r, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, r - 2, (0, 255, 0), -1, cv2.LINE_AA)
        return img
