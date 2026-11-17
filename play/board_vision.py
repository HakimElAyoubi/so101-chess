"""Where a square really is, measured by the wrist camera: checkerboard corners around the square + PnP with the
real 28 mm pitch -> the square's centre and the board plane relative to the gripper, independent of the arm's joint
errors (the camera is rigid on the gripper). Used by calibration/auto_board_calib.py through the player.

measure(img_rgb, q6, square, session) at a pointing / hover pose:
  1. the model's prediction of the 4 x 4 corners around the square -> pixels (Projector = calibrated twin + lens)
  2. the real X-corners nearest to them (Harris, sub-pixel, the four quadrants must alternate)
  3. the quads' colours tell whether the model is off by a whole square (every square is a different colour from
     its neighbours; a shift of one square flips the parity), the smallest consistent error wins
  4. PnP (planar, IPPE + LM, RANSAC) on the corners with 28 mm spacing -> the square's centre and the board's plane
     in the camera, then in the FK world through the camera's FK pose: the same FK as the gripper site, so the
     offset between them is exact up to the arm's orientation error (~1 mm per degree, gone after one nudge)
Returns dict(offset=[dx, dy] m from the site to the true centre (FK world), centre=[x, y] m, shift=[ki, kj] squares the
model was off by, h_tips=m the fingertips are above the real board, h_tips_fk, tilt_deg, scale, n_corners, rms_px,
colour_conf, corners=[...], used=[...]) or None (not enough corners).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goal"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import targets  # noqa: E402
from targets import Projector  # noqa: E402
from chess_sim import config as C  # noqa: E402

PITCH = 0.028                     # the real board's square pitch (m)
WIN = (-1.5, -0.5, 0.5, 1.5)      # corner window around the square (squares from its centre), along files and ranks
MIN_CONTRAST = 35.0               # dark / light difference (0-255) an X-corner's quadrants must show
MIN_CORNERS = 4                   # for the plane fit (planar PnP needs 4)
# The camera's board distance comes out short by a constant: at a8 / b7 / g7 the physical touch-offs (the fingertips on
# the board, 2026-10-28) put the board 7.0-8.2 mm lower than the camera does (hand-eye calibration: 15 px rms). The
# PnP translation is lengthened by this much along its ray before anything is derived from it.
DEPTH_BIAS = 0.0075
MAX_RMS_PX = 2.0                  # of the plane fit, at 1280 px width (scaled by the focal length)
INLIER_PX = 2.5                   # RANSAC: a corner this close to the fitted plane counts


class BoardVision:
    def __init__(self, proj: Projector = None, pinhole_fovy=None, depth_bias=None):
        """pinhole_fovy: use a pinhole model with that vertical field of view (deg) instead of the calibrated fisheye
        (for frames rendered by the twin)."""
        self.P = proj or Projector()
        self.s = self.P.s
        self.site = self.s.model.site("gripperframe").id
        self.pinhole_fovy = pinhole_fovy
        self.depth_bias = (0.0 if pinhole_fovy else DEPTH_BIAS) if depth_bias is None else depth_bias

    # ---- camera
    def to_normalized(self, px, width, height):
        """Pixels of a width x height frame -> normalised image coordinates (x/z, y/z), OpenCV axes."""
        px = np.asarray(px, np.float64).reshape(-1, 1, 2)
        if self.pinhole_fovy:
            f = 0.5 * height / np.tan(np.radians(self.pinhole_fovy) / 2)
            K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
            return cv2.undistortPoints(px, K, None).reshape(-1, 2)
        return cv2.fisheye.undistortPoints(px * (self.P.W0 / width), self.P.K0, self.P.D).reshape(-1, 2)

    def project(self, q6, pts, width, height):
        if not self.pinhole_fovy:
            return self.P.project(q6, pts, width)
        self.s.set_joints(q6)
        pos, R = self.s.camera_pose(C.WRIST_CAM)
        pc = (np.asarray(pts) - pos) @ R
        f = 0.5 * height / np.tan(np.radians(self.pinhole_fovy) / 2)
        z = -pc[:, 2]
        uv = np.column_stack([width / 2 + f * pc[:, 0] / z, height / 2 - f * pc[:, 1] / z])
        return uv, z > 0.01

    # ---- corners
    @staticmethod
    def _xcorner(gray, c, u, v, on_board, min_contrast=MIN_CONTRAST):
        """Is c a checkerboard X-corner? u, v: pixel vectors to the next corners along the two grid directions.
        on_board: which of the 4 quadrants (+u+v, +u-v, -u+v, -u-v) lie on the playing surface. Returns a contrast
        (0 = rejected)."""
        h, w = gray.shape
        def patch(off):
            x, y = c + off
            if not (3 <= x < w - 3 and 3 <= y < h - 3):
                return None
            return float(gray[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].mean())
        q = [patch(0.25 * (su * u + sv * v)) for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
        same = [(0, 3), (1, 2)]                         # diagonal pairs share a colour
        ok = [i for i in range(4) if on_board[i] and q[i] is not None]
        if len(ok) < 2:
            return 0.0
        a = [q[i] for i in ok if i in (0, 3)]; b = [q[i] for i in ok if i in (1, 2)]
        if not a or not b:
            return 0.0
        contrast = abs(np.mean(a) - np.mean(b))
        for i, j in same:
            if i in ok and j in ok and abs(q[i] - q[j]) > 0.5 * contrast:
                return 0.0
        return contrast if contrast >= min_contrast else 0.0

    def find_corners(self, gray, expected, u, v, on_board, radius):
        """expected: [n, 2] predicted corner pixels; u, v: grid step vectors (px) at each; returns [n, 2] found (nan
        where none) and their contrasts. Two passes: the first (radius 0.5 square) gives the median offset of the real
        grid from the predicted one, the second searches 0.3 square around the shifted predictions, so a prediction
        that is nearly half a square off can't snap to a neighbouring corner."""
        harris = cv2.cornerHarris(gray, blockSize=7, ksize=3, k=0.04)
        # local maxima only (a corner's response is a blob: the top responses of a window would all be one corner)
        k = int(np.clip(0.15 * np.median(radius), 3, 25)) * 2 + 1
        harris = np.where(harris >= cv2.dilate(harris, np.ones((k, k), np.uint8)), harris, 0.0)
        # pass 1: 0.75 square around each prediction (no point of the board is farther than 0.71 square from a corner,
        # so every visible prediction finds one). Which corner it finds is ambiguous when the model is about half a
        # square off, so only the FRACTIONAL part of each offset (in units of the local grid steps) is trusted: it is the
        # same for every corner, its median moves the predicted grid onto the real lattice (up to a whole-square shift,
        # resolved later by the squares' colours). Pass 2: 0.3 square around the moved predictions.
        found, contrast = self._find_pass(gray, harris, expected, u, v, on_board, 1.5 * radius)
        ok = ~np.isnan(found[:, 0])
        if ok.sum() >= 3:
            frac = []
            for i in np.where(ok)[0]:
                ab_ = np.linalg.solve(np.column_stack([u[i], v[i]]), found[i] - expected[i])   # offset in grid units
                frac.append((ab_ + 0.5) % 1.0 - 0.5)
            frac = np.array(frac)
            # circular median: the fractional parts live on a circle of length 1 (-0.5 == +0.5)
            def cmed(x):
                ang = 2 * np.pi * x; return float(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang))) / (2 * np.pi))
            fa, fb = cmed(frac[:, 0]), cmed(frac[:, 1])
            moved = expected + fa * u + fb * v
            found, contrast = self._find_pass(gray, harris, moved, u, v, on_board, 0.6 * radius)
        return found, contrast

    def _find_pass(self, gray, harris, expected, u, v, on_board, radius):
        h, w = gray.shape
        found, contrast = np.full((len(expected), 2), np.nan), np.zeros(len(expected))
        for i, (e, ui, vi, ob, r) in enumerate(zip(expected, u, v, on_board, radius)):
            x0, y0 = int(e[0] - r), int(e[1] - r); x1, y1 = int(e[0] + r) + 1, int(e[1] + r) + 1
            if x1 <= 2 or y1 <= 2 or x0 >= w - 2 or y0 >= h - 2:
                continue
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            win = harris[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            mask = (xx - e[0]) ** 2 + (yy - e[1]) ** 2 <= r * r
            if not mask.any():
                continue
            cand = np.where(mask, win, -np.inf)
            # the strongest few responses: the best one that is a real X-corner (jaw edges / screws respond too)
            flat = cand.ravel(); order = np.argsort(flat)[::-1][:8]
            best = None
            for k in order:
                if not np.isfinite(flat[k]) or flat[k] <= 0:
                    break
                py, px = divmod(int(k), cand.shape[1]); p = np.array([[x0 + px, y0 + py]], np.float32)
                ws = int(np.clip(0.12 * min(np.linalg.norm(ui), np.linalg.norm(vi)), 4, 14))
                try:
                    cv2.cornerSubPix(gray.astype(np.uint8), p, (ws, ws), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
                except cv2.error:
                    continue
                c = p[0].astype(np.float64)
                if np.linalg.norm(c - e) > r + 3:
                    continue
                con = self._xcorner(gray, c, ui, vi, ob)
                if con > 0 and (best is None or con > best[1]):
                    best = (c, con)
                    if con > 2 * MIN_CONTRAST:
                        break
            if best is not None:
                found[i], contrast[i] = best
        return found, contrast

    # ---- the measurement
    def measure(self, img_rgb, q6, square, session, debug=None):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        H, W = gray.shape
        f0, r0 = "abcdefgh".index(square[0]), int(square[1])
        P = self.P
        # 1. predicted corners (board coordinates: files 0..7, ranks 1..8, corners at the half-integers): the 4 x 4
        #    corners around the square; at the board's edge the window slides inward so it keeps 4 x 4 real corners
        def win(x0, lo, hi):
            return sorted((a for a in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5) if lo <= x0 + a <= hi), key=abs)[:4]
        ab = [(a, b) for a in win(f0, -0.5, 7.5) for b in win(r0, 0.5, 8.5)]
        pts = np.array([P.board_point(f0 + a, r0 + b, session) for a, b in ab])
        uv, vis = self.project(q6, pts, W, H)
        du, _ = self.project(q6, np.array([P.board_point(f0 + a + 1, r0 + b, session) for a, b in ab]), W, H)
        dv, _ = self.project(q6, np.array([P.board_point(f0 + a, r0 + b + 1, session) for a, b in ab]), W, H)
        u, v = du - uv, dv - uv
        on_board = [[(0 <= f0 + a + 0.5 * su <= 7) and (1 <= r0 + b + 0.5 * sv <= 8) for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
                    for a, b in ab]
        sq_px = np.minimum(np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1))
        radius = 0.5 * sq_px
        inside = vis & (uv[:, 0] > -0.5 * sq_px) & (uv[:, 0] < W + 0.5 * sq_px) & (uv[:, 1] > -0.5 * sq_px) & (uv[:, 1] < H + 0.5 * sq_px)
        found = np.full((len(ab), 2), np.nan); contrast = np.zeros(len(ab))
        if inside.sum() >= MIN_CORNERS:
            fi, ci = self.find_corners(gray, uv[inside], u[inside], v[inside], [on_board[i] for i in np.where(inside)[0]], radius[inside])
            found[inside], contrast[inside] = fi, ci
        ok = ~np.isnan(found[:, 0])
        out = dict(square=square, n_corners=int(ok.sum()), corners=[(ab[i], found[i].tolist()) for i in range(len(ab)) if ok[i]])
        if ok.sum() < max(MIN_CORNERS, 5):
            out["why"] = f"only {int(ok.sum())} corners found"; return self._debug(out, img_rgb, uv, found, None, debug)
        # 2. the quads' colours: the model's error in whole squares. The snap offset (found - predicted, px) moves
        #    the predicted quad centres onto the real ones, found or not
        snap = np.median(found[ok] - uv[ok], axis=0)
        qidx = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if 0 <= f0 + i <= 7 and 1 <= r0 + j <= 8]
        qc, _ = self.project(q6, np.array([P.board_point(f0 + i, r0 + j, session) for i, j in qidx]), W, H)
        qc = qc + snap
        inten = []
        for c in qc:
            x, y = int(round(c[0])), int(round(c[1])); k = int(0.12 * np.median(sq_px))
            inten.append(float(np.median(gray[max(0, y - k):y + k + 1, max(0, x - k):x + k + 1])) if 2 * k < x < W - 2 * k and 2 * k < y < H - 2 * k else np.nan)
        inten = np.array(inten); have = ~np.isnan(inten)
        if have.sum() < 2 or np.nanmax(inten) - np.nanmin(inten) < MIN_CONTRAST:
            out["why"] = "quads' colours not readable"; return self._debug(out, img_rgb, uv, found, None, debug)
        thr = 0.5 * (np.nanmax(inten) + np.nanmin(inten))
        dark = inten < thr
        # dark(i, j) = (phase + i + j) odd: fit the phase by majority
        votes = [(int(dark[n]) + i + j) % 2 for n, (i, j) in enumerate(qidx) if have[n]]
        phase = int(round(np.mean(votes))); colour_conf = float(np.mean([vt == phase for vt in votes]))
        dark00 = (phase + 0 + 0) % 2 == 1
        expect_dark = (f0 + r0) % 2 == 1
        parity_odd = dark00 != expect_dark
        # 3. PnP in a frame at the labelled (0, 0) quad's centre: X toward the h-file, Y toward rank 8, 28 mm pitch
        obj = np.array([[PITCH * a, PITCH * b, 0.0] for (a, b), o in zip(ab, ok) if o])
        imgn = self.to_normalized(found[ok], W, H)
        K1 = np.eye(3)
        # start from the model's own pose of this frame (a planar target has two PnP solutions; the model is within a
        # centimetre and a few degrees of the right one)
        flip = np.diag([1.0, -1.0, -1.0])
        self.s.set_joints(q6)
        pos, Rc = self.s.camera_pose(C.WRIST_CAM)
        ya = np.radians(targets.POSES["sessions"][session]["yaw_deg"])
        M = np.column_stack([[-np.sin(ya), np.cos(ya), 0.0], [-np.cos(ya), -np.sin(ya), 0.0], [0.0, 0.0, 1.0]])   # X: toward h, Y: toward rank 8
        R_cv = flip @ Rc.T
        rvec0, _ = cv2.Rodrigues(R_cv @ M); tvec0 = (R_cv @ (P.square_xyz(square, session) - pos)).reshape(3, 1)
        fpx = 0.5 * H / np.tan(np.radians(self.pinhole_fovy) / 2) if self.pinhole_fovy else self.P.K0[0, 0] * W / self.P.W0
        R0, _ = cv2.Rodrigues(rvec0)
        # RANSAC over 4-corner subsets (planar PnP: of its two solutions, the one near the model's orientation), then
        # the inliers refit: a wrong "corner" on a jaw edge or a piece can't pull the plane
        rng = np.random.default_rng(0)
        n = len(obj); subsets = [rng.choice(n, 4, replace=False) for _ in range(min(150, 4 * n * n))] if n > 4 else [np.arange(4)]
        best = None
        for sub in subsets:
            try:
                k, rvs, tvs, _ = cv2.solvePnPGeneric(obj[sub], imgn[sub], K1, None, flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:
                continue
            for rv, tv in zip(rvs, tvs):
                Rk, _ = cv2.Rodrigues(rv)
                if np.degrees(np.arccos(np.clip((np.trace(Rk.T @ R0) - 1) / 2, -1, 1))) > 25:
                    continue
                proj, _ = cv2.projectPoints(obj, rv, tv, K1, None)
                res = np.linalg.norm(proj.reshape(-1, 2) - imgn, axis=1) * fpx
                inl_k = np.where(res <= INLIER_PX)[0]
                score = (len(inl_k), -res[inl_k].mean() if len(inl_k) else 0)
                if best is None or score > best[0]:
                    best = (score, rv, tv, inl_k)
        if best is None or len(best[3]) < 4:
            out["why"] = "PnP failed"; return self._debug(out, img_rgb, uv, found, None, debug)
        _, rvec, tvec, inl = best
        okp, rvec, tvec = cv2.solvePnP(obj[inl], imgn[inl], K1, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        for _ in range(3):                  # re-admit / drop corners against the refit
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K1, None)
            res = np.linalg.norm(proj.reshape(-1, 2) - imgn, axis=1) * fpx
            inl2 = np.where(res <= max(INLIER_PX, 2.5 * np.median(res[inl])))[0]
            if set(inl2) == set(inl) or len(inl2) < 4:
                break
            inl = inl2
            okp, rvec, tvec = cv2.solvePnP(obj[inl], imgn[inl], K1, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], imgn[inl], K1, None, rvec, tvec)
        proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K1, None)
        rms_px = float(np.sqrt(((proj.reshape(-1, 2) - imgn[inl]) ** 2).sum(1).mean()) * fpx)
        if len(inl) < MIN_CORNERS or rms_px > MAX_RMS_PX:
            out.update(why=f"poor plane fit ({len(inl)} corners, rms {rms_px:.1f} px)", used=int(len(inl)), rms_px=rms_px)
            return self._debug(out, img_rgb, uv, found, None, debug)
        Rb, _ = cv2.Rodrigues(rvec); t = tvec.ravel()
        t = t * (1 + self.depth_bias / np.linalg.norm(t))    # the board is really this much farther along the view ray
        # OpenCV camera axes -> MuJoCo camera axes -> world (the camera's FK pose)
        centre_w = pos + Rc @ (flip @ t)
        X_w, Y_w, n_w = (Rc @ (flip @ Rb[:, k]) for k in range(3))
        if n_w[2] < 0:
            n_w = -n_w
        # 4. the whole-square ambiguity: the real square (f0, r0) is the found quad minus k' squares; k' of the right
        #    parity with the smallest error wins. Everything in the camera frame: the fingertips' position in it is the
        #    fixed camera-gripper geometry (both from the same FK), so the arm's orientation error cancels out of the
        #    in-plane offset; only the world directions of the result use the FK orientation (gone after one nudge).
        site = self.s.data.site_xpos[self.site].copy()
        t1, t2 = self.s.jaw_tip_positions(); tips_w = (np.asarray(t1) + np.asarray(t2)) / 2
        tips_c = flip @ Rc.T @ (tips_w - pos)
        n_c = Rb[:, 2] if np.dot(Rb[:, 2], -t) > 0 else -Rb[:, 2]          # toward the camera: up from the board
        def cand(i, j):
            cc = t - PITCH * (i * Rb[:, 0] + j * Rb[:, 1])                  # the real (f0, r0) centre, camera frame
            d = cc - tips_c; d_plane = d - np.dot(d, n_c) * n_c              # in the board plane
            return np.linalg.norm(d_plane), (i, j), cc, d_plane
        # the colours decide only when the unshifted square is far off (> 10 mm): the model is never that wrong once a
        # few squares are taught, while a glare-lit or half-hidden quad can misread a colour (f8, 2026-10-30)
        best = cand(0, 0); colour_mismatch = parity_odd
        if best[0] > 0.010:
            best = None
            for i in (-2, -1, 0, 1, 2):
                for j in (-2, -1, 0, 1, 2):
                    if ((i + j) % 2 == 1) == parity_odd:
                        c = cand(i, j)
                        if best is None or c[0] < best[0]:
                            best = c
            colour_mismatch = False
        _, shift, centre_c, d_plane = best
        d_w = Rc @ (flip @ d_plane)                                          # FK world directions
        true_c = pos + Rc @ (flip @ centre_c)
        h_tips = float(np.dot(tips_c - centre_c, n_c))
        jaw_c = flip @ Rc.T @ self.s.data.site_xmat[self.site].reshape(3, 3)[:, 0]   # the jaw (pointing) axis, camera frame
        cam_dist_fk = float(np.linalg.norm(P.square_xyz(square, session) - pos))
        out.update(offset=d_w[:2].tolist(), centre=(site + d_w)[:2].tolist(), site=site.tolist(), shift=list(shift),
                   h_tips=h_tips, h_tips_fk=float(tips_w[2] - self.s.cfg.board_thickness), board_z=float(tips_w[2] - h_tips),
                   tilt_deg=float(np.degrees(np.arccos(np.clip(n_w[2], -1, 1)))), normal=n_w.tolist(),
                   jaw_vs_normal_deg=float(np.degrees(np.arccos(np.clip(np.dot(-jaw_c, n_c), -1, 1)))),
                   axes=dict(h=X_w.tolist(), rank8=Y_w.tolist()), scale=float(np.linalg.norm(t) / cam_dist_fk),
                   rms_px=rms_px, used=int(len(inl)), colour_conf=colour_conf, colour_mismatch=bool(colour_mismatch),
                   snap_px=snap.tolist(), contrast=float(np.mean(contrast[ok])))
        if abs(out["scale"] - 1) > 0.08 or abs(h_tips - out["h_tips_fk"]) > 0.02 or out["tilt_deg"] > 25:
            out["why"] = f"implausible plane (scale {out['scale']:.2f}, tips {1000 * h_tips:.0f} vs FK {1000 * out['h_tips_fk']:.0f} mm, tilt {out['tilt_deg']:.0f} deg)"
        return self._debug(out, img_rgb, uv, found, (obj, inl, rvec, tvec, true_c, X_w, Y_w, pos, Rc), debug)

    def _debug(self, out, img_rgb, uv, found, pnp, path):
        if path is None:
            return out
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
        for e in uv:
            cv2.circle(img, (int(e[0]), int(e[1])), 4, (0, 0, 255), 1, cv2.LINE_AA)          # red: predicted
        for f in found:
            if not np.isnan(f[0]):
                cv2.drawMarker(img, (int(round(f[0])), int(round(f[1]))), (0, 255, 0), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)   # green: found
        if pnp is not None:
            obj, inl, rvec, tvec, true_c, X_w, Y_w, pos, Rc = pnp
            W, H = img.shape[1], img.shape[0]
            sq = np.array([true_c + 0.5 * PITCH * (sx * X_w + sy * Y_w) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
            # project through the camera directly (no joints needed): world -> camera -> pixels
            pc = (sq - pos) @ Rc
            cv = np.stack([pc[:, 0], -pc[:, 1], -pc[:, 2]], 1)
            if self.pinhole_fovy:
                f = 0.5 * H / np.tan(np.radians(self.pinhole_fovy) / 2)
                uvs = np.column_stack([W / 2 + f * cv[:, 0] / cv[:, 2], H / 2 + f * cv[:, 1] / cv[:, 2]])
            else:
                uvs, _ = cv2.fisheye.projectPoints(cv.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.P.K0, self.P.D)
                uvs = uvs.reshape(-1, 2) * (W / self.P.W0)
            cv2.polylines(img, [uvs.astype(np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)   # magenta: the real square
            c = uvs[:4].mean(0); cv2.drawMarker(img, (int(c[0]), int(c[1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(img, f"{out['square']}: " + (f"off {1000 * out['offset'][0]:+.1f},{1000 * out['offset'][1]:+.1f} mm shift {out['shift']} tips {1000 * out['h_tips']:.0f} mm rms {out['rms_px']:.1f}px"
                                                if "offset" in out else "") + (" " + out["why"] if "why" in out else ""), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(path), img)
        return out
