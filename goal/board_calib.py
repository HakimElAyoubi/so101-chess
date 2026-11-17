"""Fit the real board pose (dx, dy, yaw) per recording session from the demonstrations themselves.

For each episode: grasp = the leader's gripper command closes (>12 % -> <5 %), release = it reopens; the jaw-centre xy at
those instants comes from the calibrated MuJoCo twin (joint offsets + hand-eye), and the task text says which squares
they were. Least squares over all grasp/release points of a session gives the board pose; residuals say how good it is.
Output: goal/board_poses.json  {session: {dx, dy, yaw_deg, rms_mm, n}, episodes: {ep: session}}"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_sim.scene import ChessScene

R = Path.home() / ".cache/huggingface/lerobot/LeHak0/so101_chess_moves"
PAT = re.compile(r"(?:from|on) ([a-h][1-8]) to (?:the )?([a-h][1-8]|drop zone)")


def events(act, release_frac=0.5, release_min=0.0, close_frac=0.4, close_min=0.0):
    """(grasp index, release index) pairs from the leader's gripper command: after an opening (peak > 6 %), a grasp is
    when it falls below close_frac of that peak (and by at least close_min %), a release when it rises again by
    release_frac of the travel (at least release_min %). The defaults miss ~20 demos: releases that open only
    part-way (2.8 -> 9.5 % after a 19.7 % opening) and wide pieces closed only to ~55 % of the opening (rooks, kings);
    the skill clips use release_frac=0.15, release_min=3, close_frac=0.7, close_min=3."""
    g = act[:, 5]; out, peak, k_close, low = [], 0.0, None, None
    for k in range(1, len(g)):
        if k_close is None:
            peak = max(peak, g[k])
            if peak > 6 and g[k] < close_frac * peak and peak - g[k] >= close_min:
                k_close, low = k, g[k]
        else:
            low = min(low, g[k])
            if g[k] > low + max(release_min, release_frac * (peak - low)):
                out.append((k_close, k)); k_close, peak = None, g[k]
    return out


def main():
    s = ChessScene()
    ep = pd.concat([pd.read_parquet(f) for f in sorted(R.glob("meta/episodes/*/*.parquet"))]).sort_values("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in sorted(R.glob("data/*/*.parquet"))])
    sess_of = dict(zip(ep.episode_index, ep["data/file_index"]))
    pts = []                                  # (session, ep, kind, measured xy, square)
    drops, spares = {}, {}                    # session -> release xy of drop-zone episodes / grasp xy of spare pieces
    for e, g in data.groupby("episode_index"):
        task = ep.set_index("episode_index").loc[e, "tasks"][0]
        m = PAT.search(task)
        if not m:
            if task.startswith("place the spare"):
                a = np.stack(g["action"].values); q = np.stack(g["observation.state"].values); ev = events(a)
                if ev:
                    s.set_joints(q[ev[0][0]]); t1, t2 = s.jaw_tip_positions()
                    spares.setdefault(int(sess_of[e]), []).append(((np.asarray(t1) + np.asarray(t2)) / 2)[:2])
            continue
        frm, to = m.groups()
        a = np.stack(g["action"].values); q = np.stack(g["observation.state"].values)
        ev = events(a)
        if not ev: continue
        kc, ko = ev[0]                        # first grasp / release of the episode
        for k, sq in ((kc, frm), (ko, to)):
            if sq == "drop zone":
                s.set_joints(q[k]); t1, t2 = s.jaw_tip_positions()
                drops.setdefault(int(sess_of[e]), []).append(((np.asarray(t1) + np.asarray(t2)) / 2)[:2])
                continue
            s.set_joints(q[k]); t1, t2 = s.jaw_tip_positions()
            pts.append((int(sess_of[e]), int(e), sq, ((np.asarray(t1) + np.asarray(t2)) / 2)[:2]))
    print(len(pts), "grasp/release points from", len({p[1] for p in pts}), "episodes")
    bc = s.cfg.board_centre[:2]
    def nominal(sq):                          # square centre for the nominal board (dx=dy=yaw=0)
        return s.square_xyz(sq)[:2]
    out = {"sessions": {}, "episodes": {}}
    for sess in sorted({p[0] for p in pts}):
        P = [p for p in pts if p[0] == sess]
        A = np.array([nominal(p[2]) for p in P]); B = np.array([p[3] for p in P])
        # rigid 2-D fit B ~ Rot(yaw) (A - bc) + bc + d   (Procrustes)
        keep = np.ones(len(P), bool)
        for _ in range(3):                    # robust: refit without points > 2.5 x the median residual (regrasps etc.)
            Ak, Bk = A[keep], B[keep]
            Ac, Bc = Ak - Ak.mean(0), Bk - Bk.mean(0)
            U, _, Vt = np.linalg.svd(Ac.T @ Bc); Rm = (U @ Vt).T
            if np.linalg.det(Rm) < 0: Rm = np.eye(2)
            d = Bk.mean(0) - (Rm @ (Ak.mean(0) - bc)) - bc
            allres = np.linalg.norm((A - bc) @ Rm.T + bc + d - B, axis=1)
            keep = allres < max(0.012, 2.5 * np.median(allres))
        yaw = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))
        res = allres[keep]
        raw = np.linalg.norm(A - B, axis=1)
        out["sessions"][str(sess)] = dict(dx=float(d[0]), dy=float(d[1]), yaw_deg=float(yaw), rms_mm=float(np.sqrt((res ** 2).mean()) * 1000),
                                          rms_nominal_mm=float(np.sqrt((raw ** 2).mean()) * 1000), n=len(P), n_used=int(keep.sum()),
                                          p90_mm=float(np.percentile(res, 90) * 1000))
        r = out["sessions"][str(sess)]
        print(f"session {sess}: n={r['n_used']:3d}/{r['n']:3d}  board dx {r['dx']*1000:+6.1f} mm dy {r['dy']*1000:+6.1f} mm yaw {r['yaw_deg']:+5.1f} deg | "
              f"error with nominal board {r['rms_nominal_mm']:5.1f} mm -> fitted {r['rms_mm']:5.1f} mm rms (90% < {r['p90_mm']:.0f} mm)")
    alld = [x for v in drops.values() for x in v]; alls = [x for v in spares.values() for x in v]
    for k, r in out["sessions"].items():
        dz = drops.get(int(k)) or alld; sp = spares.get(int(k)) or alls
        r["drop_xy"] = np.median(np.array(dz), 0).tolist() if dz else None
        r["spare_xy"] = np.median(np.array(sp), 0).tolist() if sp else None
        r["n_drop"], r["n_spare"] = len(drops.get(int(k), [])), len(spares.get(int(k), []))
    print("drop-zone releases per session:", {k: len(v) for k, v in drops.items()}, "| spare grasps:", {k: len(v) for k, v in spares.items()})
    out["episodes"] = {str(k): int(v) for k, v in sess_of.items()}
    json.dump(out, open(Path(__file__).parent / "board_poses.json", "w"), indent=1)


if __name__ == "__main__":
    main()
