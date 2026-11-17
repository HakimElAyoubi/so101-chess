"""Expert success statistics under a randomisation preset (parallel, progress bar).

    python -m chess_sim.expert_stats --preset wide  [--workers 6] [--out chess_sim/data/expert_stats_wide.json]
    python -m chess_sim.expert_stats --preset narrow

Two samples (the logic of test_motion g and d, with randomise=True on every move):
  replies : Black's 20 replies to 1.e4 x seeds 1-3 (reset seed = seed * 100 + k), 60 moves
  random  : seeds 7/11/42 x 10 positions (the Italian opening + random play of 4-30 plies) x 2 moves (the second
            move from the position after the first, re-randomised), 60 moves
Each move is planned and executed by the scripted expert in the physics; success = no PlanError and the
execution's success checks (as run_move in test_motion).  Progress bar: chess_sim/data/progress.txt.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess  # noqa: E402

from chess_sim import config as C            # noqa: E402
from chess_sim.progress import ProgressBar   # noqa: E402

_W: dict = {}


def _random_position(rng: random.Random, plies: int) -> chess.Board:
    b = chess.Board()
    for _ in range(plies):
        if b.is_game_over():
            break
        moves = list(b.legal_moves)
        caps = [m for m in moves if b.is_capture(m)]
        b.push(rng.choice(caps) if caps and rng.random() < 0.5 else rng.choice(moves))
    return b


def _pick_move(rng: random.Random, b: chess.Board, want_capture: bool) -> chess.Move:
    moves = list(b.legal_moves)
    caps = [m for m in moves if b.is_capture(m)]
    castles = [m for m in moves if b.is_castling(m)]
    if castles:
        return castles[0]
    if want_capture and caps:
        return rng.choice(caps)
    return rng.choice(moves)


def jobs() -> list:
    out = []
    b = chess.Board()
    b.push_uci("e2e4")
    k = 0
    for seed in (1, 2, 3):
        for m in b.legal_moves:
            k += 1
            out.append(dict(sample="replies", fen=b.fen(), move=m.uci(), reset_seed=seed * 100 + k))
    n_capture = 0
    for seed in (7, 11, 42):
        rng = random.Random(seed)
        positions = []
        pb = chess.Board()
        for u in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"):
            pb.push_uci(u)
        positions.append(pb)
        while len(positions) < 10:
            p = _random_position(rng, rng.randint(4, 30))
            if p.is_game_over() or len(list(p.legal_moves)) < 3:
                continue
            positions.append(p)
        for pi, pos in enumerate(positions):
            board = pos.copy()
            for mi in range(2):
                if board.is_game_over():
                    break
                mv = _pick_move(rng, board, n_capture < 4 * (1 + (seed != 7)))
                n_capture += board.is_capture(mv)
                out.append(dict(sample="random", fen=board.fen(), move=mv.uci(), reset_seed=1000 * seed + 10 * pi + mi))
                board.push(mv)
    return out


def _randomisation(preset: str, overrides=()) -> "C.Randomisation":
    R = C.Randomisation.preset(preset)
    for kv in overrides:
        k, v = kv.split("=")
        cur = getattr(R, k)
        setattr(R, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))
    return R


def run_job(job: dict, preset: str, overrides=()) -> dict:
    from chess_sim.scene import ChessScene
    from chess_sim.expert import plan_move, execute, PlanError
    if "scene" not in _W:
        cfg = C.ChessConfig(randomisation=_randomisation(preset, overrides))
        _W["scene"] = ChessScene(cfg)
    s = _W["scene"]
    b = chess.Board(job["fen"])
    t0 = time.time()
    s.reset(seed=job["reset_seed"], randomise=True, fen=b.fen())
    pose = s.board_pose()
    try:
        plan = plan_move(s, job["move"], b)
    except PlanError as e:
        return dict(job, ok=False, reasons=[f"PlanError: {str(e)[:200]}"], board_pose=pose, wall=time.time() - t0)
    res = execute(s, plan)
    reasons = list(res.reasons)
    if plan.watch_error:
        reasons.append(f"watch return not planned: {plan.watch_error[:120]}")
    perr = max(res.metrics["piece_err_mm"].values()) if res.metrics["piece_err_mm"] else float("nan")
    return dict(job, ok=not reasons, reasons=[r[:200] for r in reasons], board_pose=pose, piece_err_mm=perr,
                max_lean=max([abs(pp.grasp.tilt_deg) for pp in plan.steps] + [abs(pp.place.tilt_deg) for pp in plan.steps] + [0.0]),
                wall=time.time() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=C.RANDOMISATION_PRESETS, default="wide")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--set", action="append", default=[], help="override a Randomisation field, e.g. --set piece_jitter_m=0.003")
    ap.add_argument("--only", choices=("replies", "random"), default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress-path", default=C.PROGRESS_TXT)
    a = ap.parse_args(argv)
    js = [j for j in jobs() if a.only is None or j["sample"] == a.only]
    pb = ProgressBar(len(js), path=a.progress_path)
    pb.update(0, f"expert stats ({a.preset}): starting {len(js)} moves", force=True)
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for r in pool.imap_unordered(_star, [(j, a.preset, tuple(a.set)) for j in js]):
            results.append(r)
            n_ok = sum(x["ok"] for x in results)
            pb.update(len(results), f"expert stats ({a.preset}): {n_ok}/{len(results)} OK, last {r['sample']} {r['move']} "
                                    f"{'OK' if r['ok'] else 'FAIL'}", force=True)
    for sample in ("replies", "random"):
        rs = [r for r in results if r["sample"] == sample]
        if not rs:
            continue
        print(f"{sample}: {sum(r['ok'] for r in rs)}/{len(rs)} successes ({a.preset} {' '.join(a.set)})")
        for r in rs:
            if not r["ok"]:
                p = r["board_pose"]
                print(f"   FAIL {r['move']} board ({p[0]*1000:+.1f}, {p[1]*1000:+.1f} mm, {p[2]:+.1f} deg) {r['reasons']}  fen {r['fen']}")
    out = a.out or os.path.join(C.DATA_DIR, f"expert_stats_{a.preset}{'_' + '_'.join(a.set) if a.set else ''}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    pb.done(f"expert stats ({a.preset}) DONE: {sum(r['ok'] for r in results)}/{len(results)} in {time.time() - t0:.0f}s")
    return 0


def _star(args):
    return run_job(*args)


if __name__ == "__main__":
    sys.exit(main())
