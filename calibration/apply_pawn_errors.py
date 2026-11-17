"""Fold the pawn placement test into the board calibration.

Each pawn-test run folder holds results_<square>.json: where a pawn put down with the gripper site on the model's
centre landed, relative to the true centre (real board mm, toward rank 1 / toward the h-file), per cycle, with the
jaws' opening axis in the same frame. The landing offset is two things added:
    landing(sq, cycle) = e(sq) + g1 * axis(sq) + g2 * perp(sq)
 - e: the calibration error of the model at that square (what to correct; a shift, or shift + rotation over the board)
 - g: the gripper's own offset at the player's jaw roll (fixed pad -> pawn axis, plus the twin's site error), fixed
   in the gripper frame, so it turns with the square's bearing: along the opening axis (g1) and across it (g2).
Along-axis offsets repeat to ~0.5 mm, across-axis ones to ~2 mm (the pawn can sit anywhere across the pads), so the
across equations get less weight. The fitted e moves every taught point of the latest session, the board model +
correction map are refitted as the player does, the height map is kept, and a new session is saved.

    python calibration/apply_pawn_errors.py data/calibration/pawn_place [--dry] [--rotation]
"""
import json, sys, glob
from datetime import datetime
import numpy as np
sys.path[:0] = ["play", "goal", "calibration", "."]
import targets

root = sys.argv[1]; DRY = "--dry" in sys.argv; ROT = "--rotation" in sys.argv
P = targets.Projector(); s = P.s
data = json.load(open("goal/board_poses.json")); ses = data["sessions"]
key = max((k for k in ses if k.isdigit()), key=int); cur = ses[key]
targets.POSES["sessions"].update(ses)
a = np.radians(cur["yaw_deg"]); ax_rank = np.array([np.cos(a), np.sin(a)]); ax_file = np.array([-np.sin(a), np.cos(a)])
s_rank, s_file = cur.get("scale_rank") or 1.0, cur.get("scale_file") or 1.0

# the cycles: (square, landing (far, h) mm, opening axis (far, h)); the newest results file per square wins
files = {}
for f in sorted(glob.glob(f"{root}/**/results_*.json", recursive=True)):
    r = json.load(open(f))
    if isinstance(r, dict) and r.get("results"):
        files[r["square"]] = r
cycles = [(sq, np.array([c["after"]["toward_r1"], c["after"]["toward_h"]]), np.array(c["open_axis_far_h"]))
          for sq, r in files.items() for c in r["results"] if c.get("after")]
if len(cycles) < 2:
    sys.exit("fewer than two measured cycles")
squares = sorted(set(c[0] for c in cycles)); print(f"session {key}; {len(cycles)} cycles at {', '.join(squares)}")
for sq in squares:
    L = np.array([c[1] for c in cycles if c[0] == sq]); print(f"   {sq}: landed {np.round(L.mean(0), 1).tolist()} (far, h) mm over {len(L)} cycle(s)")

# board position of each square (mm from the board's centre, board axes) for the rotation term
def board_mm(sq):
    f, r = "abcdefgh".index(sq[0]), int(sq[1]); return np.array([(4.5 - r) * 28.0, (f - 3.5) * 28.0])      # (far, h)
# unknowns: e_far, e_h, [theta], g1, g2
n_e = 3 if ROT and len(squares) >= 3 else 2
rows, rhs, w = [], [], []
for sq, land, ax in cycles:
    perp = np.array([-ax[1], ax[0]]); b = board_mm(sq)
    for comp, unit, weight in ((land @ ax, ax, 1.0), (land @ perp, perp, 0.5)):          # along / across the jaws
        row = [unit[0], unit[1]] + ([(-b[1]) * unit[0] + b[0] * unit[1]] if n_e == 3 else []) + [1.0 if unit is ax else 0.0, 1.0 if unit is perp else 0.0]
        rows.append(row); rhs.append(comp); w.append(weight)
A = np.array(rows) * np.sqrt(w)[:, None]; y = np.array(rhs) * np.sqrt(w)
sol, *_ = np.linalg.lstsq(A, y, rcond=None)
e = sol[:2]; theta = sol[2] if n_e == 3 else 0.0; g = sol[n_e:]
pred = np.array(rows) @ sol; res = (np.array(rhs) - pred)
print(f"gripper offset at the player's roll: {g[0]:+.1f} mm along the jaws' opening axis, {g[1]:+.1f} mm across")
print(f"calibration error (the model's centre vs the true one): {e[0]:+.1f} mm toward rank 1, {e[1]:+.1f} mm toward h"
      + (f", rotation {np.degrees(theta) * 1000:+.1f} mrad" if n_e == 3 else "") + f"; fit residuals rms {np.sqrt(np.mean(res ** 2)):.1f} mm (along {np.sqrt(np.mean(res[0::2] ** 2)):.1f}, across {np.sqrt(np.mean(res[1::2] ** 2)):.1f})")
for sq in squares:
    ee = e + (theta * np.array([-board_mm(sq)[1], board_mm(sq)[0]]) if n_e == 3 else 0)
    print(f"   {sq}: error {np.round(ee, 1).tolist()} (far, h) mm -> the model moves by {np.round(-ee, 1).tolist()}")
if DRY:
    sys.exit(0)

def correct(xy):
    """World xy (m) of a model point -> where it should be: moved by -e (and the rotation), real mm -> the arm's mm."""
    far, hh = (np.asarray(xy[:2]) - s.cfg.board_centre[:2]) @ ax_rank, (np.asarray(xy[:2]) - s.cfg.board_centre[:2]) @ ax_file
    ee = e + (theta * np.array([-1000 * hh / s_file, 1000 * far / s_rank]) if n_e == 3 else 0)
    return np.asarray(xy[:2]) - (ee[0] * s_rank * ax_rank + ee[1] * s_file * ax_file) / 1000.0
taught = {sq: correct(xy).tolist() for sq, xy in cur["taught"].items()}
tz = {sq: [*correct(v[:2]), v[2]] for sq, v in (cur.get("taught_z") or {}).items() if v}
fit = P.fit_squares(taught, cur); corr = P.fit_correction(taught, fit)
height = P.fit_height(tz) if tz else cur.get("height")
new = str(int(key) + 1)
ses[new] = {**{k: fit[k] for k in ("dx", "dy", "yaw_deg", "scale_rank", "scale_file")},
            "drop_xy": correct(cur["drop_xy"]).tolist() if cur.get("drop_xy") else None, "spare_xy": cur.get("spare_xy"),
            "rms_mm": None, "taught": taught, "per_square_mm": fit["per_square_mm"], "correction": corr, "taught_z": tz, "height": height,
            "pawn_test": dict(squares=squares, error_far_h_mm=e.tolist(), rotation_rad=float(theta), gripper_along_across_mm=g.tolist(), cycles=len(cycles)),
            "source": f"session {key} corrected by the pawn placement test at {', '.join(squares)}",
            "time": datetime.now().isoformat(timespec="seconds")}
json.dump(data, open("goal/board_poses.json", "w"), indent=1)
chk = [(sq, 1000 * np.linalg.norm(P.square_xyz(sq, new)[:2] - correct(P.square_xyz(sq, key)))) for sq in squares]
print(f"session {new} saved: yaw {fit['yaw_deg']:.2f}, size {fit['scale_rank']:.4f} / {fit['scale_file']:.4f}; refit residual at the measured squares "
      + ", ".join(f"{sq} {d:.1f} mm" for sq, d in chk))
