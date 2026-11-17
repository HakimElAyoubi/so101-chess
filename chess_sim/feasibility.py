"""Feasibility of the board-pose randomisation: arm reach and wrist-camera view at extreme board poses.

    python -m chess_sim.feasibility                       # reach at the 9 poses of the 'wide' range + watch view
    python -m chess_sim.feasibility --x -20,-10,0,10,20   # scan the x offset (mm) at the y / yaw extremes
    python -m chess_sim.feasibility --workers 6 --out chess_sim/data/feasibility.json

Reach: for each board pose (dx, dy, yaw), every square x every piece type at its grasp height must admit a
self-collision-free top-down grasp pose (Expert.grasp_pose, the logic of test_motion a: lean <= 8 deg, all pieces
excluded from the clearance check).  View: the watch pose must keep the four board corners inside the wrist image
with config.WATCH_MARGIN, unoccluded, for every pose (ChessScene.watch_metrics).  Progress bar in
chess_sim/data/progress.txt (progress_view.command).
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_sim import config as C            # noqa: E402
from chess_sim.progress import ProgressBar   # noqa: E402

ALL_SQUARES = [f + r for r in "12345678" for f in "abcdefgh"]
_W: dict = {}                                # per-process scene / expert (workers)


def extreme_poses(board_range: Sequence[float]) -> List[Tuple[float, float, float]]:
    """Nominal pose + the 8 extreme combinations (x_min/x_max, y_min/y_max, -yaw/+yaw) of a board range + the two
    x extremes at y = yaw = 0 (11 poses)."""
    x0, x1, y0, y1, yaw = board_range
    out = [(0.0, 0.0, 0.0)]
    for p in itertools.product((x0, x1), (y0, y1), (-yaw, yaw)):
        p = tuple(float(v) for v in p)
        if p not in out:
            out.append(p)
    # the x extremes with the board centred and square: the rank-8 squares are CLOSEST to the pan axis there (a
    # sideways shift moves d8/e8 away from it), so the near-reach limit is not at a corner of the box
    for x in (x0, x1):
        if (float(x), 0.0, 0.0) not in out:
            out.append((float(x), 0.0, 0.0))
    return out


def _scene():
    if "scene" not in _W:
        from chess_sim.scene import ChessScene
        from chess_sim.expert import get_expert
        s = ChessScene()
        s.reset(seed=0, randomise=False, arm_pose=None)
        _W["scene"], _W["expert"] = s, get_expert(s)
    return _W["scene"], _W["expert"]


def reach_at_pose(pose: Tuple[float, float, float], squares: Optional[Sequence[str]] = None,
                  types: Sequence[str] = tuple(C.PIECE_TYPES)) -> dict:
    """IK reach of every (square, type) at a board pose (dx m, dy m, yaw deg): dict(pose, ok, total, fails,
    max_lean_deg)."""
    import mujoco
    from chess_sim.expert import PlanError
    s, ex = _scene()
    s.set_board_pose(*pose)
    s.set_position()
    mujoco.mj_forward(s.model, s.data)
    fails, ok, lean = [], 0, 0.0
    sq_list = list(squares or ALL_SQUARES)
    for sq in sq_list:
        for t in types:
            try:
                found, _, _, _ = ex.grasp_pose(s.square_xyz(sq), t, exclude=list(s.piece_names), open_q=C.gripper_open_ctrl(),
                                               label=f"{sq}/{t}", roll_from=0.0)
                ok += 1
                lean = max(lean, abs(found.tilt_deg))
            except PlanError:
                fails.append(f"{sq}/{t}")
    s.set_board_pose(0.0, 0.0, 0.0)
    s.set_position()
    return dict(pose=list(pose), ok=ok, total=len(sq_list) * len(types), fails=fails, max_lean_deg=round(lean, 2))


def reach_zones(jitter: float) -> dict:
    """IK reach of every drop cell x type and both spare cells (queen) with the pads at their nominal place and
    shifted by the extreme jitter (+-jitter in x and y, the four corners): dict(ok, total, fails)."""
    import numpy as np
    from chess_sim.expert import PlanError
    s, ex = _scene()
    fails, ok, tot = [], 0, 0
    offs = [(0.0, 0.0)] + [(sx * jitter, sy * jitter) for sx in (-1, 1) for sy in (-1, 1)] if jitter > 0 else [(0.0, 0.0)]
    for off in offs:
        d = np.array([off[0], off[1], 0.0])
        targets = [(f"drop{i}{j}/{t}", s.drop_cell_xyz(i, j) + d, t) for i, j in s.drop_zone_cells() for t in C.PIECE_TYPES]
        targets += [(f"spare_{c}/queen", s.spare_xyz(c) + d, "queen") for c in "wb"]
        for name, p, t in targets:
            tot += 1
            try:
                ex.grasp_pose(p, t, exclude=list(s.piece_names), open_q=C.gripper_open_ctrl(), label=name, roll_from=0.0)
                ok += 1
            except PlanError:
                fails.append(f"{name}@{off[0]*1000:+.0f},{off[1]*1000:+.0f}")
    return dict(ok=ok, total=tot, fails=fails)


def reach_table(poses: Sequence[Tuple[float, float, float]], workers: int = 1, squares: Optional[Sequence[str]] = None,
                progress: Optional[ProgressBar] = None, label: str = "reach") -> List[dict]:
    """reach_at_pose for every pose (in a process pool when workers > 1), in the order of `poses`."""
    out: Dict[int, dict] = {}
    args = [(p, squares) for p in poses]
    if workers <= 1:
        for i, a in enumerate(args):
            out[i] = reach_at_pose(*a)
            if progress:
                progress.update(i + 1, f"{label}: pose {i + 1}/{len(poses)} {out[i]['ok']}/{out[i]['total']}", force=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(poses))) as pool:
            res = {i: pool.apply_async(reach_at_pose, a) for i, a in enumerate(args)}
            done = 0
            while len(out) < len(res):
                for i, r in res.items():
                    if i not in out and r.ready():
                        out[i] = r.get()
                        done += 1
                        if progress:
                            progress.update(done, f"{label}: {done}/{len(poses)} poses, last {out[i]['pose']} "
                                                  f"{out[i]['ok']}/{out[i]['total']}", force=True)
                time.sleep(0.5)
    return [out[i] for i in range(len(poses))]


def watch_view(scene, vec6=None) -> dict:
    """Worst-case margin / smallest board area of the watch pose over scene.watch_board_poses(), plus per pose."""
    import numpy as np
    vec6 = scene.watch_pose() if vec6 is None else vec6
    r = scene.watch_metrics(vec6, ext_limit=np.inf, rays_all=True)
    per = [dict(pose=p, margin=round((1.0 - e) / 2.0, 4), area=round(a, 4)) for p, e, a in r["per_pose"]]
    return dict(feasible=r["feasible"], margin=(1.0 - r["extent"]) / 2.0, area_min=r["area_frac"],
                area_nominal=r["area_nominal"], occluded=r["occluded"], collision=r["collision"], per_pose=per)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", default=None, help="comma-separated x offsets (mm) to scan at the y/yaw extremes (default: the "
                                               "'wide' range's x_min, x_max)")
    ap.add_argument("--y", type=float, default=None, help="y half-range (mm), default the 'wide' range")
    ap.add_argument("--yaw", type=float, default=None, help="yaw half-range (deg), default the 'wide' range")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(C.DATA_DIR, "feasibility.json"))
    ap.add_argument("--progress-path", default=C.PROGRESS_TXT)
    ap.add_argument("--no-view", action="store_true")
    a = ap.parse_args(argv)
    x0, x1, y0, y1, yaw = C.Randomisation().board_range()
    yh = a.y / 1000.0 if a.y is not None else y1
    yawh = a.yaw if a.yaw is not None else yaw
    poses: List[Tuple[float, float, float]] = []
    xs = [float(v) / 1000.0 for v in a.x.split(",")] if a.x else [x0, x1]
    poses.append((0.0, 0.0, 0.0))
    for x in xs:
        if (x, 0.0, 0.0) not in poses:
            poses.append((x, 0.0, 0.0))
        for p in itertools.product((x,), (-yh, yh), (-yawh, yawh)):
            if p not in poses:
                poses.append(tuple(float(v) for v in p))
    pb = ProgressBar(len(poses), path=a.progress_path)
    pb.update(0, f"feasibility: reach at {len(poses)} board poses", force=True)
    t0 = time.time()
    table = reach_table(poses, workers=a.workers, progress=pb, label="feasibility reach")
    print(f"reach ({time.time() - t0:.0f}s): pose (dx mm, dy mm, yaw deg) -> reachable / total, max lean, failures")
    for r in table:
        p = r["pose"]
        print(f"  ({p[0]*1000:+5.1f}, {p[1]*1000:+5.1f}, {p[2]:+4.1f})  {r['ok']:3d}/{r['total']}  lean {r['max_lean_deg']:4.1f}  "
              f"{sorted(set(f.split('/')[0] for f in r['fails']))}")
    out = dict(reach=table)
    z = reach_zones(C.Randomisation().zone_jitter_m)
    out["zones"] = z
    print(f"drop zone + spare cells (nominal and the 4 jitter corners): {z['ok']}/{z['total']} {z['fails']}")
    if not a.no_view:
        from chess_sim.scene import ChessScene
        s = ChessScene()
        v = watch_view(s)
        out["view"] = v
        print(f"watch view: feasible {v['feasible']} worst margin {v['margin']*100:.2f}% area min {v['area_min']:.3f} "
              f"nominal {v['area_nominal']:.3f}")
        for q in v["per_pose"]:
            print(f"  {q}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    pb.done(f"feasibility DONE ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
