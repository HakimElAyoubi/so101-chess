"""Where a pawn stands on a square, from one wrist-camera photo, independent of the camera's pose model:
the checkerboard corners around the square (board_vision's detector, labelled by the model's prediction) give a plane
homography board -> undistorted image; the pawn's base disc lies on that plane, so its centre maps back to board
coordinates. The base disc's centre is fitted as a circle through the blob's boundary on the side facing the
camera's nadir (everything of the pawn above the board projects away from the nadir, so that side of the silhouette is
the base's own edge). The labels are checked by the squares' colours (an odd shift flips them) and, when a previous
position is known, by continuity (a pawn does not jump a square between photos).
   python pawn_vision.py <photo.jpg> <sidecar.json>   (offline: the sidecar holds q6, square, session)
"""
import sys, json, numpy as np, cv2
sys.path[:0] = ["play", "goal", "calibration", "."]
import board_vision as BV
from chess_sim import config as C
PITCH = 28.0          # real square (mm)
DARK = 24             # gray below this: the gloss-black pawn (7-15); its own shadow on the board is 35-60, the dark squares 60+


class PawnVision:
    def __init__(self, bv=None):
        self.bv = bv or BV.BoardVision(); self.P = self.bv.P; self.s = self.P.s

    # ---- lens
    def to_pixels(self, nrm, W):
        pts = np.asarray(nrm, np.float64).reshape(-1, 1, 2)
        obj = np.concatenate([pts, np.ones((len(pts), 1, 1))], axis=2)
        px, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), self.P.K0, self.P.D)
        return px.reshape(-1, 2) * (W / self.P.W0)

    def board_to_px(self, Hm, ab, W):
        ab = np.asarray(ab, np.float64).reshape(-1, 2)
        return self.to_pixels(cv2.perspectiveTransform(ab[None], Hm)[0], W)

    def px_to_board(self, Hm, px, W, H):
        nrm = self.bv.to_normalized(np.asarray(px, np.float64).reshape(-1, 2), W, H)
        return cv2.perspectiveTransform(nrm[None], np.linalg.inv(Hm))[0]

    # ---- the pawn
    def base_centre(self, gray, Hm, centre_px, d_board, W, H, win=100, near_half=True, log=None, origin=(0, 0)):
        """Board coordinates (squares, labelled frame) of the pawn's base centre near centre_px, or None.
        Dark blobs near centre_px are the candidates (the pawn, but also dark squares in the gripper's shadow);
        each is tested by whether the Canny edges around it fit a circle of the base's size (R 6.5-11.5 mm) in the
        board plane; the best fit wins. The circle is refined on the half facing the camera's nadir (d_board:
        unit vector, squares frame, from the nadir toward the pawn), where the silhouette is the disc's bottom edge,
        on the plane; the far half is the top edge, pushed outward by parallax."""
        u, v = int(centre_px[0]), int(centre_px[1])
        x0, y0, x1, y1 = max(u - win, 0), max(v - win, 0), min(u + win, W), min(v + win, H)
        sub = gray[y0:y1, x0:x1]
        # only the board itself: the dark frame around it (and its letters) would pass for a pawn next to an edge square
        gy, gx = np.mgrid[y0:y1, x0:x1]
        fr = self.px_to_board(Hm, np.column_stack([gx.ravel(), gy.ravel()]), W, H) + np.array(origin, float)
        on = ((fr[:, 0] > -0.5 - 0.15) & (fr[:, 0] < 7.5 + 0.15) & (fr[:, 1] > 0.5 - 0.15) & (fr[:, 1] < 8.5 + 0.15)).reshape(sub.shape)
        mask = cv2.morphologyEx(((sub < 30) & on).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
        blobs = sorted((np.hypot(cent[i][0] + x0 - u, cent[i][1] + y0 - v), i) for i in range(1, n) if 600 < stats[i, cv2.CC_STAT_AREA] < 30000)
        d = np.asarray(d_board, float); d /= np.linalg.norm(d)
        rng = np.random.default_rng(1); results = []
        for _, i in blobs[:4]:
            rough = np.array([cent[i][0] + x0, cent[i][1] + y0]); area = int(stats[i, cv2.CC_STAT_AREA])
            r = 75; ex0, ey0 = max(int(rough[0]) - r, 0), max(int(rough[1]) - r, 0)
            ed = cv2.Canny(cv2.GaussianBlur(gray[ey0:int(rough[1]) + r, ex0:int(rough[0]) + r], (3, 3), 0), 30, 90)
            ys, xs = np.where(ed > 0)
            if len(xs) < 30:
                continue
            pts = np.column_stack([xs + ex0, ys + ey0]).astype(np.float64)
            bsq = self.px_to_board(Hm, pts, W, H); ab = bsq + np.array(origin, float)
            keep = (ab[:, 0] > -0.5) & (ab[:, 0] < 7.5) & (ab[:, 1] > 0.5) & (ab[:, 1] < 8.5)          # edges on the board only
            bd = bsq[keep] * PITCH
            rc = self.px_to_board(Hm, rough, W, H)[0] * PITCH
            dist = np.linalg.norm(bd - rc, axis=1)
            frac = np.abs(((bd / PITCH) + 0.5) % 1.0 - 0.5) * PITCH          # mm from the nearest lattice line, per axis
            on_line = (frac < 1.5).any(axis=1)                                   # the squares' own edges: not the pawn
            ring = bd[(dist > 4) & (dist < 15) & ~on_line]
            if len(ring) < 30:
                continue
            best = None
            for _k in range(600):                                # RANSAC circle through 3 edge points, R 6.5..11.5 mm
                p = ring[rng.choice(len(ring), 3, replace=False)]
                A = np.column_stack([p[:, 0], p[:, 1], np.ones(3)]); bb = -(p ** 2).sum(1)
                try:
                    sol = np.linalg.solve(A, bb)
                except np.linalg.LinAlgError:
                    continue
                cx, cy = -sol[0] / 2, -sol[1] / 2; Rk = np.sqrt(max(cx ** 2 + cy ** 2 - sol[2], 1e-6))
                if not 6.5 <= Rk <= 11.5:
                    continue
                inl = np.abs(np.linalg.norm(ring - [cx, cy], axis=1) - Rk) < 0.35
                if best is None or inl.sum() > best[0]:
                    best = (int(inl.sum()), inl, np.array([cx, cy]), float(Rk))
            if best is None or best[0] < 30:
                continue
            n_in, inl, c0, R0 = best
            q = ring[inl]
            if near_half:
                qn = q[(q - c0) @ d <= 0.3 * R0]
                q = qn if len(qn) >= 12 else q
            A = np.column_stack([q[:, 0], q[:, 1], np.ones(len(q))]); bb = -(q ** 2).sum(1)
            sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
            c_fit = np.array([-sol[0] / 2, -sol[1] / 2]); R = float(np.sqrt(max(c_fit @ c_fit - sol[2], 1e-6)))
            if not 6.5 <= R <= 11.5:
                continue
            cab = c_fit / PITCH + np.array(origin, float)                                        # a pawn stands on the board
            if not (-0.5 < cab[0] < 7.5 and 0.5 < cab[1] < 8.5):
                continue
            res = np.abs(np.linalg.norm(q - c_fit, axis=1) - R)
            # coverage: how much of the circle the inliers span (a dark square's edges never fit a 9 mm circle well)
            ang = np.degrees(np.arctan2(*(ring[inl] - c_fit).T[::-1])); cover = len(set((ang // 15).astype(int))) * 15
            results.append((n_in * cover, dict(c=c_fit, R_mm=R, fit_rms_mm=float(res.mean()), n_edge=int(len(q)), n_ransac=n_in, n_ring=int(len(ring)),
                                                cover_deg=int(cover), blob_px=area)))
        if not results:
            return None
        _, b = max(results, key=lambda t: t[0])
        if log is not None:
            log(f"pawn base: circle R {b['R_mm']:.1f} mm through {b['n_edge']} {'near-side ' if near_half else ''}edge points (rms {b['fit_rms_mm']:.2f} mm; "
                f"{b['n_ransac']} of {b['n_ring']} ring edges on it, {b['cover_deg']} deg of arc); dark blob {b['blob_px']} px")
        c_fit = b.pop("c")
        return c_fit / PITCH, b

    # ---- the whole measurement
    def measure(self, img_rgb, q6, square, session, last=None, guess_px=None, debug=None, log=print):
        """-> dict(toward_h, toward_r1 (real mm from the true centre), shift, n_corners, hom_rms_px, base_px, ...) or None.
        last: the pawn's previous (toward_h, toward_r1): the labels must keep it within half a square (continuity).
        guess_px: where to look for the pawn (default: the labelled square's centre, i.e. the model's prediction)."""
        W, H = img_rgb.shape[1], img_rgb.shape[0]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        m = self.bv.measure(img_rgb, q6, square, session, debug=debug)
        if not m or len(m.get("corners", [])) < 6:
            log(f"   corners: {(m or {}).get('why', 'none')}"); return None
        ab = np.array([c[0] for c in m["corners"]], float); px = np.array([c[1] for c in m["corners"]], float)
        nrm = self.bv.to_normalized(px, W, H)
        Hm, inl = cv2.findHomography(ab, nrm, cv2.RANSAC, 0.003)
        if Hm is None or inl.sum() < 5:                                   # looser: the lens model is weaker off-centre
            Hm, inl = cv2.findHomography(ab, nrm, cv2.RANSAC, 0.008)
        if Hm is None or inl.sum() < 5:
            log(f"   homography failed ({len(ab)} corners)"); return None
        inl = inl.ravel() > 0
        res = np.linalg.norm(cv2.perspectiveTransform(ab[inl][None], Hm)[0] - nrm[inl], axis=1) * self.P.K0[0, 0] * W / self.P.W0
        # the model's prediction of the square's centre, the camera's nadir direction and tilt (twin)
        f0, r0 = "abcdefgh".index(square[0]), int(square[1])
        self.s.set_joints(q6); pos, Rc = self.s.camera_pose(C.WRIST_CAM)
        c_model = self.P.square_xyz(square, session)
        pp, _ = self.bv.project(q6, np.array([c_model]), W, H); c_px = pp[0]
        import auto_board_calib as A
        far, hh = A.to_board_axes(c_model[:2] - pos[:2], session); d_board = np.array([hh, -far])      # (file, rank) axes
        axis_deg = float(np.degrees(np.arccos(np.clip(abs(Rc[2, 2]), -1, 1))))
        # 1. the pawn's base (labelled frame)
        if guess_px is None:
            guess_px = self.board_to_px(Hm, [(0.0, 0.0)], W)[0] if last is None else self.board_to_px(Hm, [(last[0] / PITCH, -last[1] / PITCH)], W)[0]
        base = self.base_centre(gray, Hm, guess_px, d_board, W, H, win=170 if last is None and axis_deg > 15 else 220,
                                near_half=axis_deg > 15, log=lambda t: log("   " + t), origin=(f0, r0))
        if base is None:
            log("   pawn base not found"); return None
        bc, binfo = base
        # 2. the labels' parity from the quads' colours around the pawn (the quad under the pawn skipped)
        qidx = [(i, j) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2) if 0 <= f0 + i <= 7 and 1 <= r0 + j <= 8 and np.linalg.norm(bc - [i, j]) > 0.6]
        qpx = self.board_to_px(Hm, qidx, W); inten = []
        for (x, y) in qpx:
            x, y = int(round(x)), int(round(y)); k = 8
            inten.append(float(np.median(gray[y - k:y + k + 1, x - k:x + k + 1])) if k < x < W - k and k < y < H - k else np.nan)
        inten = np.array(inten); have = ~np.isnan(inten)
        odd = None; parity_note = "colours unreadable"
        if have.sum() >= 3 and np.nanmax(inten) - np.nanmin(inten) > 40:
            thr = 0.5 * (np.nanmax(inten) + np.nanmin(inten)); dark = inten < thr
            votes = [(int(dark[n]) + i + j) % 2 for n, (i, j) in enumerate(qidx) if have[n]]
            phase = int(round(np.mean(votes))); conf = float(np.mean([v == phase for v in votes]))
            dark00 = phase % 2 == 1; expect_dark = (f0 + r0) % 2 == 1
            odd = dark00 != expect_dark
            parity_note = f"colour conf {conf:.2f} ({int(have.sum())} quads), labels {'off by an odd number of squares' if odd else 'parity ok'}"
        # 3. the shift: among the parity-consistent candidates, the one keeping the pawn nearest its last position
        #    (or, without one, nearest the model's prediction of the centre)
        cands = [(i, j) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2) if odd is None or ((i + j) % 2 == 1) == odd]
        if last is not None:
            lb = np.array([last[0], -last[1]]) / PITCH
            shift = min(cands, key=lambda c: np.linalg.norm(bc - np.array(c) - lb))
        else:
            cpx = self.board_to_px(Hm, cands, W)
            shift = cands[int(np.argmin(np.linalg.norm(cpx - c_px, axis=1)))]
        off = bc - np.array(shift, float)
        out = dict(toward_h=float(PITCH * off[0]), toward_r1=float(-PITCH * off[1]), shift=list(shift), n_corners=int(inl.sum()),
                   hom_rms_px=float(res.mean()), base_px=self.board_to_px(Hm, bc, W)[0].tolist(), parity=parity_note, axis_deg=axis_deg,
                   model_centre_px=c_px.tolist(), model_off_sq=(self.px_to_board(Hm, c_px, W, H)[0] - np.array(shift)).tolist(), H=Hm.tolist(), **binfo)
        if last is not None:
            out["jump_mm"] = float(np.hypot(out["toward_h"] - last[0], out["toward_r1"] - last[1]))
        return out

    def blob_near(self, img_rgb, q6, square, session, last):
        """Rough check for the hold test: is a pawn-sized dark blob still within ~12 mm of the pawn's last position
        (toward_h, toward_r1 mm)? -> distance in mm, or None if nothing is there (or no lattice)."""
        W, H = img_rgb.shape[1], img_rgb.shape[0]; gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        m = self.bv.measure(img_rgb, q6, square, session)
        if not m or len(m.get("corners", [])) < 6:
            return None
        ab = np.array([c[0] for c in m["corners"]], float); px = np.array([c[1] for c in m["corners"]], float)
        Hm, inl = cv2.findHomography(ab, self.bv.to_normalized(px, W, H), cv2.RANSAC, 0.004)
        if Hm is None or inl.sum() < 5:
            return None
        lb = np.array([last[0], -last[1]]) / PITCH
        c = self.board_to_px(Hm, lb, W)[0]; u, v = int(c[0]), int(c[1]); win = 70
        x0, y0 = max(u - win, 0), max(v - win, 0); sub = gray[y0:v + win, x0:u + win]
        if sub.size == 0:
            return None
        mask = cv2.morphologyEx((sub < 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
        best = None
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 800:
                continue
            bc = self.px_to_board(Hm, np.array([cent[i][0] + x0, cent[i][1] + y0]), W, H)[0]
            d = PITCH * np.linalg.norm(bc - lb)
            if d < 14 and (best is None or d < best):
                best = float(d)
        return best


if __name__ == "__main__":
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB); side = json.load(open(sys.argv[2]))
    import targets
    pv = PawnVision()
    r = pv.measure(img, np.array(side["q6"]), side["square"], side["session"], last=side.get("last"), debug=sys.argv[1].replace(".jpg", "_corners.jpg"))
    print(json.dumps(r, indent=1))
