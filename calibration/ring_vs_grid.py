"""How far the grasp recorder's white ring sits from where it should be on the printed board, per hover frame.

The recorder draws the ring at the model's spot (square centre + the drill's offset, board level) through the twin's
camera with the arm's real joints, so the ring is where the model believes the spot is. The printed checkerboard
corners around the square (board_vision's detector, labelled by the model's prediction) give a plane homography
board <-> undistorted image, and the ring's pixel mapped through it is the model's spot in printed-board coordinates.
Ring minus intended spot = the model's error at that hover, as the camera sees it (FK + camera model included).

    python calibration/ring_vs_grid.py <frames dir or *.jpg ...>      (frames named <item>_<square>_<piece>_...jpg)
Writes <dir>/ring_check/<frame>_ringcheck.jpg overlays and prints one line per frame + a summary per square.
"""
import sys, json, glob, os, re
sys.path[:0] = ["play", "goal", "calibration", "recording", "."]
import numpy as np, cv2
import staged as S, targets, board_vision as BV
from chess_recorder import READY, JOINT_NAMES

PITCH = 28.0
P = targets.Projector(); Pl = S.Planner(P); sess = targets.LATEST; bv = BV.BoardVision()
rd = json.loads(READY.read_text()); ready6 = np.array([rd[j] for j in JOINT_NAMES], float)
plan = json.load(open("recording/grasp_drill.json"))["episodes"]


def find_marks(bgr):
    """The recorder's overlay in a 640-wide view: green dot (0,220,0) r5 filled, white ring r11 thickness 2."""
    b, g, r = cv2.split(bgr.astype(np.int16))
    green = ((g > 140) & (r < 110) & (b < 110)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(green)
    dots = [(st[i, cv2.CC_STAT_AREA], tuple(cen[i])) for i in range(1, n) if 30 < st[i, cv2.CC_STAT_AREA] < 250]
    if not dots:
        return None, None, 0.0
    dot = np.array(max(dots)[1])
    white = (np.minimum(np.minimum(b, g), r) > 225).astype(np.float32)
    k = 14; yy, xx = np.mgrid[-k:k + 1, -k:k + 1]; rr = np.hypot(xx, yy)
    tpl = ((rr > 9.5) & (rr < 12.5)).astype(np.float32)
    Hh, Ww = white.shape
    x0, y0 = int(max(dot[0] - 70, k)), int(max(dot[1] - 70, k)); x1, y1 = int(min(dot[0] + 70, Ww - k - 1)), int(min(dot[1] + 70, Hh - k - 1))
    sub = white[y0 - k:y1 + k + 1, x0 - k:x1 + k + 1]
    res = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    ring = np.array([x0 + loc[0], y0 + loc[1]], float)
    return dot, (ring if mx > 0.3 else None), float(mx)


def check(path, out_dir):
    name = os.path.basename(path); mo = re.match(r"(\d+)_([a-h][1-8])_", name)
    if not mo:
        return None
    e = plan[int(mo.group(1))]; sq, piece, off = e["square"], e["piece"], e["offset_mm"]
    if sq != mo.group(2):
        return None
    bgr = cv2.imread(path); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); Hh, Ww = bgr.shape[:2]
    dot, ring, score = find_marks(bgr)
    if ring is None:
        return dict(frame=name, square=sq, why=f"ring not found (score {score:.2f})")
    hover, _ = Pl.grasp_hover(P.square_xyz(sq, sess), piece, ready6, S.HOVER_GRIP)
    if hover is None:
        return dict(frame=name, square=sq, why="no hover")
    m = bv.measure(rgb, hover, sq, sess)
    if not m or len(m.get("corners", [])) < 6:
        return dict(frame=name, square=sq, why=f"corners: {(m or {}).get('why', 'none')}")
    ab = np.array([c[0] for c in m["corners"]], float); px = np.array([c[1] for c in m["corners"]], float)
    nrm = bv.to_normalized(px, Ww, Hh)
    Hm, inl = cv2.findHomography(ab, nrm, cv2.RANSAC, 0.003)
    if Hm is None or inl.sum() < 5:
        Hm, inl = cv2.findHomography(ab, nrm, cv2.RANSAC, 0.008)
    if Hm is None or inl.sum() < 5:
        return dict(frame=name, square=sq, why=f"homography failed ({len(ab)} corners)")
    inl = inl.ravel() > 0
    res = np.linalg.norm(cv2.perspectiveTransform(ab[inl][None], Hm)[0] - nrm[inl], axis=1) * P.K0[0, 0] * Ww / P.W0
    def b2px(pts):
        return cv2.fisheye.projectPoints(np.concatenate([cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(1, -1, 2), Hm)[0].reshape(-1, 1, 2), np.ones((len(pts), 1, 1))], axis=2),
                                         np.zeros(3), np.zeros(3), P.K0, P.D)[0].reshape(-1, 2) * (Ww / P.W0)
    rb = cv2.perspectiveTransform(bv.to_normalized(ring, Ww, Hh)[None], np.linalg.inv(Hm))[0][0]      # (file, rank) squares from the labelled centre
    ring_mm = np.array([PITCH * rb[0], PITCH * rb[1]])                                                 # toward h, toward rank 8
    want = np.array([off[1], off[0]]); err = ring_mm - want
    # the corner labels come from the model's prediction with approximate joints and can slip a whole square; the
    # squares' colours say whether the slip is odd (one square along a file or a rank) or even (none, or diagonal):
    # among the slips with that parity, the one that leaves the ring nearest its intended spot wins
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); f0, r0 = "abcdefgh".index(sq[0]), int(sq[1])
    qidx = [(i, j) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2) if 0 <= f0 + i <= 7 and 1 <= r0 + j <= 8 and np.hypot(rb[0] - i, rb[1] - j) > 0.6]
    inten = []
    for (x, y) in b2px(qidx):
        x, y = int(round(x)), int(round(y)); kk = 6
        inten.append(float(np.median(gray[y - kk:y + kk + 1, x - kk:x + kk + 1])) if kk < x < Ww - kk and kk < y < Hh - kk else np.nan)
    inten = np.array(inten); have = ~np.isnan(inten); odd = None
    if have.sum() >= 3 and np.nanmax(inten) - np.nanmin(inten) > 40:
        thr = 0.5 * (np.nanmax(inten) + np.nanmin(inten))
        votes = [(int(inten[n] < thr) + i + j) % 2 for n, (i, j) in enumerate(qidx) if have[n]]
        phase = int(round(np.mean(votes))); dark00 = phase % 2 == 1; odd = dark00 != ((f0 + r0) % 2 == 1)
    cands = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if odd is None or ((i + j) % 2 == 1) == odd]
    relabel = np.array(min(cands, key=lambda c: np.linalg.norm(err - PITCH * np.array(c))), float)
    err = err - PITCH * relabel; ring_mm = ring_mm - PITCH * relabel
    # overlay: detected ring (red), printed centre (yellow cross), intended spot (yellow ring)
    v = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    c0, cw = b2px([tuple(relabel)])[0], b2px([(relabel[0] + want[0] / PITCH, relabel[1] + want[1] / PITCH)])[0]
    for (a, bb), ok in zip(px, inl):
        cv2.circle(v, (int(2 * a), int(2 * bb)), 5, (0, 255, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(v, (int(2 * c0[0]), int(2 * c0[1])), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.circle(v, (int(2 * cw[0]), int(2 * cw[1])), 22, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(v, (int(2 * ring[0]), int(2 * ring[1])), 22, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(v, f"{name}: ring is {err[0]:+.1f} mm toward h, {err[1]:+.1f} toward rank 8 off the printed spot ({int(inl.sum())} corners, rms {res.mean():.1f} px)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(v, f"{name}: ring is {err[0]:+.1f} mm toward h, {err[1]:+.1f} toward rank 8 off the printed spot ({int(inl.sum())} corners, rms {res.mean():.1f} px)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(out_dir, exist_ok=True); cv2.imwrite(os.path.join(out_dir, name.replace(".jpg", "_ringcheck.jpg")), v, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dict(frame=name, square=sq, piece=piece, want_h_r8=want.tolist(), ring_h_r8=ring_mm.round(1).tolist(), err_h_r8=err.round(1).tolist(),
                n_corners=int(inl.sum()), rms_px=float(res.mean()), ring_px=ring.tolist(), ring_score=score,
                flag=(f"labels slipped {relabel.astype(int).tolist()} squares (colours: {'odd' if odd else 'even' if odd is not None else '?'}), corrected" if np.abs(relabel).max() > 0 else ""))


if __name__ == "__main__":
    files = []
    for a in sys.argv[1:] or ["data/grasp_drill/20261101/frames"]:
        files += sorted(glob.glob(os.path.join(a, "*hover*.jpg"))) if os.path.isdir(a) else [a]
    files = [f for f in files if "ringcheck" not in f and "annotated" not in f]
    out_dir = os.path.join(os.path.dirname(files[0]) if files else ".", "ring_check")
    rows = []
    for f in files:
        r = check(f, out_dir)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['frame']:48s} " + (f"ring off by {r['err_h_r8'][0]:+6.1f} mm toward h, {r['err_h_r8'][1]:+6.1f} toward rank 8   ({r['n_corners']} corners, rms {r['rms_px']:.1f} px) {r['flag']}"
                                      if "err_h_r8" in r else r["why"]))
    json.dump(rows, open(os.path.join(out_dir, "ring_check.json"), "w"), indent=1)
    by = {}
    for r in rows:
        if "err_h_r8" in r:
            by.setdefault(r["square"], []).append(r["err_h_r8"])
    if by:
        print("per square (mean of the frames): " + "; ".join(f"{sq} {np.mean(v, 0)[0]:+.1f} h / {np.mean(v, 0)[1]:+.1f} r8 mm (n={len(v)})" for sq, v in by.items()))
