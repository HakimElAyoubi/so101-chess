"""Separate the board model's error from the gripper's own geometry, from the grasp drill.

Two independent observations per grasp, both of "the model's spot vs the printed one":
  1. physical: the human's motion of the gripper site from the hover (above the model's spot) to the grasp (pads on the
     piece at the printed spot):      d = R g - e        (R g: the fixed pad-vs-site offset, in the gripper's frame,
     turned by the hover's bearing; e: the model's error at that square, board frame)
  2. camera: the recorder's ring vs the printed spot (calibration/ring_vs_grid.py):   r = e + R c
     (c: the camera overlay's own offset, fixed in the gripper's frame)
Each is fitted by least squares for e (a shift over the board, or shift + rotation + the two scales) and the fixed
gripper-frame term. If both give the same e, that is the board model's error at the drill's hover poses.
    python calibration/fit_drill_hover_errors.py [dataset root]
"""
import sys, json, glob, os, time
sys.path[:0] = ["play", "goal", "calibration", "recording", "."]
import numpy as np, pandas as pd
import staged as S, targets, auto_board_calib as A
from chess_recorder import READY, JOINT_NAMES

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/huggingface/lerobot/LeHak0/so101_chess_grasp")
P = targets.Projector(); Pl = S.Planner(P); s = P.s; sess = targets.LATEST
rd = json.loads(READY.read_text()); ready6 = np.array([rd[j] for j in JOINT_NAMES], float)
site = s.model.site("gripperframe").id
plan = json.load(open("recording/grasp_drill.json"))["episodes"]


def site_pose(q6):
    s.set_joints(np.asarray(q6, float)); return s.data.site_xpos[site].copy(), s.data.site_xmat[site].reshape(3, 3).copy()


def axes_board(R):
    """The gripper's opening axis and finger axis at this pose, as unit vectors in the board's (far, h) frame."""
    out = []
    for v in (R[:2, 2], R[:2, 0]):
        f, h = A.to_board_axes(v / (np.linalg.norm(v) + 1e-9), sess); u = np.array([f, h]); out.append(u / np.linalg.norm(u))
    return out


def board_mm(sq):
    f, r = "abcdefgh".index(sq[0]), int(sq[1]); return np.array([(4.5 - r) * 28.0, (f - 3.5) * 28.0])      # (far, h) from the centre


def fit(obs, sign, label):
    """obs: list of (square, value (far, h) mm, a_open, a_x). Model: value = sign * e(sq) + g1 a_open + g2 a_x."""
    for n_e, name in ((2, "shift"), (5, "shift + rotation + scales")):
        rows, rhs = [], []
        for sq, v, ao, ax in obs:
            b = board_mm(sq)
            for comp in (0, 1):
                e_row = [sign if comp == 0 else 0, sign if comp == 1 else 0]
                if n_e == 5:   # e = e0 + th * (-h, far) + (kf * far, kh * h)
                    e_row += [sign * (-b[1] if comp == 0 else b[0]), sign * (b[0] if comp == 0 else 0), sign * (b[1] if comp == 1 else 0)]
                rows.append(e_row + [ao[comp], ax[comp]]); rhs.append(v[comp])
        Am, y = np.array(rows), np.array(rhs)
        sol, *_ = np.linalg.lstsq(Am, y, rcond=None); res = y - Am @ sol
        e0 = sol[:2]; g = sol[n_e:]
        txt = f"   [{label}, {name}] model error e: {e0[0]:+.1f} mm toward rank 1, {e0[1]:+.1f} toward h"
        if n_e == 5:
            txt += f", rotation {np.degrees(sol[2]):+.2f} deg, scale {1 + sol[3]:.3f} ranks / {1 + sol[4]:.3f} files"
        txt += f" | fixed gripper-frame term: {g[0]:+.1f} along the opening axis, {g[1]:+.1f} along the fingers | residual rms {np.sqrt(np.mean(res ** 2)):.1f} mm (n={len(obs)})"
        print(txt)
        if n_e == 5:
            def e_at(sq):
                b = board_mm(sq); return e0 + sol[2] * np.array([-b[1], b[0]]) + np.array([sol[3] * b[0], sol[4] * b[1]])
            sqs = sorted(set(o[0] for o in obs), key=lambda q: (q[1], q[0]))
            print("      e per square: " + ", ".join(f"{q} ({e_at(q)[0]:+.0f},{e_at(q)[1]:+.0f})" for q in sqs))
    return sol


# 1. physical
meta = {int(json.loads(l)["episode_index"]): json.loads(l) for l in open("recording/grasp_meta.jsonl") if l.strip()}
parts = []
for f in sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet"))):
    try:
        parts.append(pd.read_parquet(f))
    except Exception as ex:
        print(f"skipping {os.path.basename(f)}: {str(ex)[:50]}")
df = pd.concat(parts, ignore_index=True)
phys = []
for ep, g in df.groupby("episode_index"):
    m = meta.get(int(ep))
    if not m or str(m["board_session"]) != sess:
        continue
    Q = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())
    poses = [site_pose(q) for q in Q]; k = int(np.argmin([p[2] for p, _ in poses]))
    p0, R0 = poses[0]; pk, _ = poses[k]
    d = pk - p0; far, hh = A.to_board_axes(d[:2], sess); ao, ax = axes_board(R0)
    phys.append((m["square"], np.array([far, hh]), ao, ax))
print(f"physical: {len(phys)} grasps on session {sess}  (d = R g - e)")
fit(phys, -1.0, "physical")

# 2. camera ring
rc = os.path.join("data/grasp_drill/20261101/frames/ring_check/ring_check.json")
ring = []
if os.path.exists(rc):
    for r in json.load(open(rc)):
        if "err_h_r8" not in r:
            continue
        e = plan[int(r["frame"].split("_")[0])]
        hover, _ = Pl.grasp_hover(P.square_xyz(e["square"], sess), e["piece"], ready6, S.HOVER_GRIP)
        if hover is None:
            continue
        _, R0 = site_pose(hover); ao, ax = axes_board(R0)
        ring.append((r["square"], np.array([-r["err_h_r8"][1], r["err_h_r8"][0]]), ao, ax))       # (far, h) = (-toward r8, toward h)
    print(f"\ncamera ring: {len(ring)} frames  (r = e + R c)")
    fit(ring, +1.0, "ring")
