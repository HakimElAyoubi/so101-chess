"""Fit the board pose from ONE wrist image at a known joint state: choose (dx, dy, yaw) so the projected grid lines
(7 inner + 2 outer per direction) land on the strongest image edges. Pieces and the jaw add noise but most lines are
long and mostly unoccluded. Usage: python goal/fit_board_image.py frame.jpg joints.json [out.jpg]"""
import json, sys
import numpy as np, cv2
sys.path.insert(0, __import__("os").path.dirname(__file__))
from targets import Projector, POSES, LATEST


def grid_world(P, pose, n=40, sizes=(1.0, 1.0)):
    bc = P.s.cfg.board_centre[:2]; h = 4 * 0.028; z = P.s.cfg.board_thickness
    a = np.radians(pose[2]); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    lines = []
    for i in range(9):
        t = -h + i * 0.028; s = np.linspace(-h, h, n)
        for pts in (np.column_stack([np.full(n, t), s]), np.column_stack([s, np.full(n, t)])):
            xy = (pts * sizes) @ R.T + bc + pose[:2]
            lines.append(np.column_stack([xy, np.full(n, z)]))
    return lines


def main():
    img = cv2.imread(sys.argv[1]); q = json.load(open(sys.argv[2]))
    res, out = fit(img, q)
    cv2.imwrite(sys.argv[3] if len(sys.argv) > 3 else "fit.jpg", out)
    json.dump(res, open("goal/check/image_fit.json", "w"), indent=1)


def fit(img, q, verbose=True, base=None, sizes=(1.0, 1.0)):
    """base: (dx, dy, yaw) to search around (default: session 7); sizes: the board's (ranks, files) size factors in the
    arm's frame, held fixed."""
    J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    q6 = np.array([q[j] for j in J]); W = img.shape[1]
    P = Projector()
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    mag = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (5, 5), 0)
    mag /= np.percentile(mag, 99)
    gn = (g - np.median(g)) / (np.percentile(g, 90) - np.percentile(g, 10) + 1e-6)
    bc = P.s.cfg.board_centre[:2]; z = P.s.cfg.board_thickness
    # 9 sample points per square (inner 60 %), squares indexed (file 0=a .. 7=h, rank 1..8); a1 is dark
    offs = np.array([(i, j) for i in (-0.3, 0, 0.3) for j in (-0.3, 0, 0.3)]) * 0.028
    sq = [(f, r) for f in range(8) for r in range(1, 9)]
    nominal = np.array([P.s.square_xyz("abcdefgh"[f] + str(r))[:2] for f, r in sq])
    expect = np.array([-1.0 if (f + r) % 2 == 1 else 1.0 for f, r in sq])     # a1: f=0, r=1 -> dark (-1)
    def color_score(pose):
        a = np.radians(pose[2]); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        cen = ((nominal - bc) * sizes) @ R.T + bc + pose[:2]
        pts = (cen[:, None, :] + (offs * sizes)[None] @ R.T).reshape(-1, 2)
        uv, vis = P.project(q6, np.column_stack([pts, np.full(len(pts), z)]), W)
        ok = vis & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0] - 1)
        val = np.full(len(pts), np.nan); u, v = uv[ok].astype(int).T; val[ok] = gn[v, u]
        med = np.nanmedian(val.reshape(len(sq), -1), axis=1)                 # median: robust to a piece on the square
        good = ~np.isnan(med)
        return float(np.mean(np.clip(med[good], -1, 1) * expect[good])) if good.sum() > 16 else -1.0
    def score(pose):
        pts = np.concatenate(grid_world(P, pose, sizes=sizes))
        uv, vis = P.project(q6, pts, W)
        inside = vis & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0] - 1)
        if inside.sum() < 200: return 0.0
        u, v = uv[inside].astype(int).T
        return float(np.mean(np.minimum(mag[v, u], 1.5))) + 0.5 * color_score(pose)
    s7 = POSES["sessions"]["7"]
    base = np.array([s7["dx"], s7["dy"], s7["yaw_deg"]] if base is None else base, float)
    best = (score(base), base)
    if verbose: print(f"last session pose score {best[0]:.3f}")
    for dx in np.arange(-0.032, 0.0321, 0.004):          # about +-1 square around the last session's pose
        for dy in np.arange(-0.032, 0.0321, 0.004):
            for dyaw in np.arange(-8, 8.1, 1.0):
                p = base + [dx, dy, dyaw]; sc = score(p)
                if sc > best[0]: best = (sc, p)
    # local refinement
    p, sc = best[1].copy(), best[0]
    for step in ([0.002, 0.002, 0.5], [0.001, 0.001, 0.2], [0.0005, 0.0005, 0.1]):
        improved = True
        while improved:
            improved = False
            for k in range(3):
                for sgn in (-1, 1):
                    t = p.copy(); t[k] += sgn * step[k]; s_ = score(t)
                    if s_ > sc: p, sc, improved = t, s_, True
    if verbose:
        print(f"colour agreement: last session {color_score(base):+.2f}, fitted {color_score(p):+.2f} (1 = every visible square right)")
        print(f"fitted: dx {p[0]*1000:+.1f} mm  dy {p[1]*1000:+.1f} mm  yaw {p[2]:+.2f} deg  score {sc:.3f} | "
              f"change vs last session: {np.hypot(*(p[:2]-base[:2]))*1000:.0f} mm, {p[2]-base[2]:+.1f} deg")
    out = img.copy()
    for pose, col in ((base, (0, 0, 255)), (p, (0, 255, 0))):
        for L in grid_world(P, pose, sizes=sizes):
            uv, vis = P.project(q6, L, W); uv = uv[vis]
            if len(uv) > 1: cv2.polylines(out, [uv.astype(np.int32)], False, col, 1, cv2.LINE_AA)
    return dict(dx=float(p[0]), dy=float(p[1]), yaw_deg=float(p[2]), score=sc), out


if __name__ == "__main__":
    main()
