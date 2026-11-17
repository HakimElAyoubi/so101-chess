"""Fold the grasp-free hover measurements (calibration/pawn_place.py --hover-only) into the board calibration.

Each hover_<square>.json says how far the model's centre must move (model_shift_far_h, real mm toward rank 1 / toward
the h-file) so that, at that square, the jaws come down on a pawn the way they did at the squares where the grasps
worked (e4, a8). The measured squares' taught points move by exactly that; every other taught point moves by a
similarity transform (shift + rotation + the two board scales) fitted to the measured shifts, so the whole board
follows the trend; the board model + correction map are refitted as the player does; the height map is kept; a new
session is saved.    python calibration/apply_hover_errors.py data/calibration/pawn_place [--dry]
"""
import json, sys, glob
from datetime import datetime
import numpy as np
sys.path[:0] = ["play", "goal", "calibration", "."]
import targets

root = sys.argv[1]; DRY = "--dry" in sys.argv
P = targets.Projector(); s = P.s
data = json.load(open("goal/board_poses.json")); ses = data["sessions"]
key = max((k for k in ses if k.isdigit()), key=int); cur = ses[key]
targets.POSES["sessions"].update(ses)
a = np.radians(cur["yaw_deg"]); ax_rank = np.array([np.cos(a), np.sin(a)]); ax_file = np.array([-np.sin(a), np.cos(a)])
s_rank, s_file = cur.get("scale_rank") or 1.0, cur.get("scale_file") or 1.0

meas = {}
for f in sorted(glob.glob(f"{root}/**/hover_*.json", recursive=True)):         # the newest per square wins
    r = json.load(open(f)); meas[r["square"]] = r
if len(meas) < 2:
    sys.exit("fewer than two measured squares")
# the zero of this measurement is set at the squares where the real grasps worked with the current model (e4, a8):
# their mean reading is the method's own bias (a few mm across the jaws, from the camera model) and is subtracted
REF = [sq for sq in ("e4", "a8") if sq in meas]
if REF:
    bias = np.mean([meas[sq]["model_shift_far_h"] for sq in REF], axis=0)
    print(f"reference squares {', '.join(REF)}: the method's own reading there {np.round(bias, 1).tolist()} (far, h) mm is taken as zero"
          + (f" (they differ by {np.round(np.array(meas[REF[0]]['model_shift_far_h']) - np.array(meas[REF[1]]['model_shift_far_h']), 1).tolist()})" if len(REF) == 2 else ""))
    for r in meas.values():
        r["model_shift_far_h"] = (np.array(r["model_shift_far_h"]) - bias).tolist()
print(f"session {key}; {len(meas)} squares measured at the hover:")
for sq, r in meas.items():
    print(f"   {sq}: model centre to move {r['model_shift_far_h'][0]:+.1f} mm toward rank 1, {r['model_shift_far_h'][1]:+.1f} mm toward h "
          f"(pawn {r['D_along_across'][0]:+.1f} along / {r['D_along_across'][1]:+.1f} across the jaws, {r['n']} photos, spread {np.round(r['spread'], 1).tolist()})")

def world_shift(r):
    """The square's shift in the arm's frame (m)."""
    far, hh = r["model_shift_far_h"]; return (far * s_rank * ax_rank + hh * s_file * ax_file) / 1000.0

# similarity fit (shift, rotation, scale along ranks and files) to the measured shifts, in board axes
pts = np.array([P.square_xyz(sq, key)[:2] for sq in meas]); tgt = np.array([p + world_shift(meas[sq]) for p, sq in zip(pts, meas)])
bc = s.cfg.board_centre[:2]
B = np.array([[(p - bc) @ ax_rank, (p - bc) @ ax_file] for p in pts]); T = np.array([[(t - bc) @ ax_rank, (t - bc) @ ax_file] for t in tgt])
n_par = 5 if len(meas) >= 4 else 2
if n_par == 5:
    # x' = x + dx + kx * x - th * y ; y' = y + dy + th * x + ky * y
    A = []; y = []
    for (x, yv), (xt, yt) in zip(B, T):
        A.append([1, 0, x, 0, -yv]); y.append(xt - x)
        A.append([0, 1, 0, yv, x]); y.append(yt - yv)
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    dx, dy, kx, ky, th = sol
    print(f"fitted: shift {1000 * dx:+.1f} / {1000 * dy:+.1f} mm (far / h), scale {1 + kx:.4f} along ranks, {1 + ky:.4f} along files, rotation {np.degrees(th):+.2f} deg")
else:
    dx, dy = (T - B).mean(0); kx = ky = th = 0.0
    print(f"fitted: shift {1000 * dx:+.1f} / {1000 * dy:+.1f} mm (far / h)")
def model(p):
    x, yv = (p - bc) @ ax_rank, (p - bc) @ ax_file
    xn, yn = x + dx + kx * x - th * yv, yv + dy + th * x + ky * yv
    return bc + xn * ax_rank + yn * ax_file
res = np.array([1000 * np.linalg.norm(model(p) - t) for p, t in zip(pts, tgt)])
print("residuals of the fit at the measured squares (mm): " + ", ".join(f"{sq} {r_:.1f}" for sq, r_ in zip(meas, res)))
if DRY:
    sys.exit(0)

taught = {}
for sq, xy in cur["taught"].items():
    xy = np.asarray(xy, float)
    taught[sq] = (xy + world_shift(meas[sq])).tolist() if sq in meas else model(xy).tolist()
tz = {sq: [*(np.asarray(v[:2]) + world_shift(meas[sq]) if sq in meas else model(np.asarray(v[:2]))), v[2]] for sq, v in (cur.get("taught_z") or {}).items() if v}
fit = P.fit_squares(taught, cur); corr = P.fit_correction(taught, fit)
height = P.fit_height(tz) if tz else cur.get("height")
new = str(int(key) + 1)
ses[new] = {**{k: fit[k] for k in ("dx", "dy", "yaw_deg", "scale_rank", "scale_file")},
            "drop_xy": model(np.asarray(cur["drop_xy"])).tolist() if cur.get("drop_xy") else None, "spare_xy": cur.get("spare_xy"),
            "rms_mm": None, "taught": taught, "per_square_mm": fit["per_square_mm"], "correction": corr, "taught_z": tz, "height": height,
            "hover_test": {sq: r["model_shift_far_h"] for sq, r in meas.items()},
            "source": f"session {key} corrected by the hover (jaw aim) measurements at {', '.join(meas)}",
            "time": datetime.now().isoformat(timespec="seconds")}
json.dump(data, open("goal/board_poses.json", "w"), indent=1)
targets.POSES["sessions"][new] = ses[new]
chk = [(sq, 1000 * np.linalg.norm(P.square_xyz(sq, new)[:2] - tgt[i])) for i, sq in enumerate(meas)]
print(f"session {new} saved: yaw {fit['yaw_deg']:.2f}, size {fit['scale_rank']:.4f} / {fit['scale_file']:.4f}; the measured squares now sit "
      + ", ".join(f"{sq} {d:.1f} mm" for sq, d in chk) + " from their corrected positions")
