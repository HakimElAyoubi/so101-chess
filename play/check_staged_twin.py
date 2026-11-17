"""Twin check of the staged executor (plan step 2) before it touches the arm.

A. Every square: grasp hover IK (with the piece of the eval setup, or pawn and king on empty squares), jaw clearance to
   the neighbouring pieces at the hover, whether the target is in the wrist image and clear of the jaw, wrist flex
   (the leader must reach the same pose for the step-4 recordings), and the place hover IK.
B. The 20 eval moves run end to end on the simulated follower (play/fake_arm.py) with the scripted stand-in skill:
   result, retries, duration, and the smallest clearance of the jaws / the carried piece to the other pieces and of the
   arm to the board, per stage. Three moves are rerun with the first grasp shifted half a square to test the miss -> retry path.
The twin's clearances are known to be pessimistic by a few mm on the real arm (journal phase 4): information, not a gate.

    python play/check_staged_twin.py        -> play/results/staged_twin_check.json (+ progress in data/progress.txt)
"""
import json, sys, time
from pathlib import Path

import chess
import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "goal"), str(ROOT / "calibration")]
import staged as S                                      # noqa: E402
from fake_arm import FakeArm, ScriptedSkill             # noqa: E402
from targets import LATEST, POSES, Projector            # noqa: E402
from chess_sim import config as C                       # noqa: E402
from chess_sim.expert import Expert                     # noqa: E402
from chess_sim.scene import ChessScene                  # noqa: E402
from ready_view import View                             # noqa: E402

EVAL = json.loads((HERE / "eval_20.json").read_text())
READY = json.loads((ROOT / "recording/ready_pose.json").read_text())
JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
READY6 = np.array([READY[j] for j in JN], float)
PROG = ROOT / "data/progress.txt"
TYPES = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP, "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING}
FPS = 30
MAX_STEP_DEG = 4.0                                      # the player's per-tick limit


def progress(i, n, t0, what):
    el = time.time() - t0; eta = el / max(i, 1) * (n - i)
    PROG.write_text(f"[{'#' * int(30 * i / n):<30}] {100 * i / n:5.1f}%  {i}/{n} {what}  elapsed {el / 60:4.1f} min  "
                    f"ETA {eta / 60:4.1f} min | staged twin check\n")


def eval_board():
    b = chess.Board(None)
    for colour, key in ((chess.BLACK, "setup_black"), (chess.WHITE, "setup_white")):
        for sq, p in EVAL[key].items():
            b.set_piece_at(chess.parse_square(sq), chess.Piece(TYPES[p], colour))
    return b


def place(scene, board, P, session):
    """set_position, then every piece onto its square's centre as the session's board pose has it: the twin's board is
    rigid 28 mm squares, the real one (in the arm's frame) has its own size along ranks / files (scale_rank / _file)."""
    mapping = scene.set_position(board)
    for sq, b in mapping.items():
        p = scene.piece_pos(b); p[:2] = P.square_xyz(sq, session)[:2]
        scene.set_piece_pose(b, p, quat=scene.piece_quat(b))
    mujoco.mj_forward(scene.model, scene.data)
    return mapping


def main():
    session = LATEST
    pose = POSES["sessions"][session]
    P = Projector(); Pl = S.Planner(P)
    scene = ChessScene(); scene.set_board_pose(pose["dx"], pose["dy"], pose["yaw_deg"])
    board = eval_board(); mapping = place(scene, board, P, session)
    if not (pose.get("scale_rank") or pose.get("scale_file")):
        assert np.allclose(scene.square_xyz("a1")[:2], P.square_xyz("a1", session)[:2], atol=1e-6), "board pose mismatch"
    ex = Expert(scene); V = View()
    t0 = time.time(); n_total = 64 + len(EVAL["trials"]) + 3
    out = {"session": session, "board_pose": {k: pose.get(k) for k in ("dx", "dy", "yaw_deg", "scale_rank", "scale_file")}, "squares": [], "moves": []}

    # ---------------- A. hover per square
    for i, sq in enumerate(chess.SQUARE_NAMES):
        pc = board.piece_at(chess.parse_square(sq))
        for ptype in ([chess.piece_name(pc.piece_type)] if pc else ["pawn", "king"]):
            xyz = P.square_xyz(sq, session)
            q, info = Pl.grasp_hover(xyz, ptype, READY6, S.HOVER_GRIP)
            row = {"square": sq, "piece": ptype, "occupied": bool(pc), "ok": q is not None, **info}
            if q is not None:
                qr = C.lerobot_to_qpos(q)
                d, who = ex.jaw_piece_clearance(qr[:5], qr[5], exclude=[mapping[sq]] if sq in mapping else [])
                V.set(q)
                tgt = xyz + np.array([0, 0, 0.02])
                uv, ang = V.project(tgt[None], 1280 / 1920); uv = uv[0]
                inimg = bool(ang[0] < 80 and 0 <= uv[0] < 1280 and 0 <= uv[1] < 720)
                blocked = any(cv2.pointPolygonTest(h, (float(uv[0]), float(uv[1])), False) >= 0 for h in V.jaw_hulls(1280 / 1920))
                contacts = ex.arm_contacts(qr[:5], qr[5])
                row.update(wrist_flex=round(float(q[3]), 1), clearance_mm=round(1000 * d, 1), nearest=who, in_image=inimg,
                           clear_of_jaw=inimg and not blocked, u=round(float(uv[0]) / 1280, 2), v=round(float(uv[1]) / 720, 2),
                           arm_contacts=[c[:2] for c in contacts])
            ph, pi = Pl.place_hover(xyz, 0.018, q if q is not None else READY6, 0.8)
            row.update(place_ok=ph is not None, place_tilt=pi["tilt"])
            out["squares"].append(row)
        progress(i + 1, n_total, t0, "squares/moves")

    # ---------------- B. eval moves end to end on the simulated follower
    def run(task, offset=0.0):
        mapping = place(scene, board, P, session)
        bodies = {b: (scene.piece_type[b], scene.piece_pos(b)) for b in mapping.values()}
        arm = FakeArm(Pl, bodies, READY6)
        sk = ScriptedSkill(Pl, offset_m=offset)
        mv = S.StagedMove(Pl, sk, task, session, READY6, 0.0, drop_xyz=P.drop_xyz(session)); sk.move = mv
        frm = S.MOVE.search(task).group(3); target = mapping[frm]
        worst = {}
        t, k = 0.0, 0
        while mv.result is None and t < 120:
            a = mv.step(arm.q, None, t)
            if a is not None:
                arm.target = np.clip(a, arm.q - MAX_STEP_DEG, arm.q + MAX_STEP_DEG)
            arm.tick(1 / FPS); t += 1 / FPS; k += 1
            if k % 3:
                continue
            for b, (typ, p) in arm.pieces.items():          # moved pieces where the fake arm put them
                scene.set_piece_pose(b, p)
            qr = C.lerobot_to_qpos(arm.q)
            st = mv.stage
            # the first cm of a carry / retreat is still where the skill closed the jaws: that clearance is the skill's,
            # not the path's (the stand-in closes between two pieces on purpose in the miss test)
            if st in ("carry", "retreat"):
                if mv.t0 != getattr(mv, "_chk_t0", None):
                    mv._chk_t0, mv._chk_z0 = mv.t0, Pl.fk(arm.q)[0][2]
                if Pl.fk(arm.q)[0][2] - mv._chk_z0 < 0.010:
                    st = f"{st}, leaving the grasp point"
            d, who = ex.jaw_piece_clearance(qr[:5], qr[5], exclude=[target])
            if arm.held is not None:
                dc, whoc = ex.carried_piece_clearance(qr[:5], qr[5], target, arm.hang, np.zeros(3), exclude=[target])
                if dc < d:
                    d, who = dc, f"carried piece vs {whoc}"
            board_hit = [c for c in ex.arm_contacts(qr[:5], qr[5]) if "board" in (c[0] + c[1]) or "table" in (c[0] + c[1])]
            w = worst.setdefault(st, {"clearance_mm": 99.0, "nearest": "", "board_contact_mm": 0.0})
            if 1000 * d < w["clearance_mm"]:
                w["clearance_mm"], w["nearest"] = round(1000 * d, 1), who
            if board_hit:
                w["board_contact_mm"] = max(w["board_contact_mm"], round(1000 * max(c[2] for c in board_hit), 1))
        placed_xy = arm.pieces[target][1][:2]
        err = float(np.linalg.norm(placed_xy - mv.to_xyz[:2]) * 1000)
        return {"task": task, "offset_m": offset, "result": list(mv.result) if mv.result else ["timeout", ""],
                "retries": mv.retries, "duration_s": round(t, 1), "placement_error_mm": round(err, 1),
                "events": mv.events, "info": {k: v for k, v in mv.info.items()}, "per_stage": worst}

    for i, tr in enumerate(EVAL["trials"]):
        out["moves"].append(run(tr["task"]))
        progress(64 + i + 1, n_total, t0, "squares/moves")
    for j, i in enumerate((0, 6, 12)):            # half a square off: closes between two pieces -> miss -> retry
        out["moves"].append(run(EVAL["trials"][i]["task"], offset=0.014))
        progress(64 + len(EVAL["trials"]) + j + 1, n_total, t0, "squares/moves")

    path = HERE / "results/staged_twin_check.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    PROG.write_text(f"[{'#' * 30}] 100.0%  {n_total}/{n_total} done in {(time.time() - t0) / 60:.1f} min | staged twin check\n")
    summarize(out)
    print("->", path)


def summarize(out):
    sq = out["squares"]
    ok = [r for r in sq if r["ok"]]
    print(f"\nA. grasp hover: {len(ok)}/{len(sq)} square/piece cases solved; place hover {sum(r['place_ok'] for r in sq)}/{len(sq)}")
    bad = [f"{r['square']}({r['piece']})" for r in sq if not r["ok"]]
    if bad:
        print("   no grasp hover:", ", ".join(bad))
    if ok:
        dt = [r["tilt"] - r["tilt_template"] for r in ok]
        print(f"   tilt = template on {sum(abs(x) < 1e-6 for x in dt)}/{len(ok)}; largest change {max(dt, key=abs):+.0f} deg")
        wf = [r["wrist_flex"] for r in ok]
        print(f"   wrist flex at the hover: {min(wf):.1f} .. {max(wf):.1f} deg (human clip starts: ~88-93, leader limit 102.4)")
        print(f"   target in image {sum(r['in_image'] for r in ok)}/{len(ok)}, clear of the jaw {sum(r['clear_of_jaw'] for r in ok)}/{len(ok)}")
        occ = [r for r in ok if r["occupied"]]
        cl = sorted(occ, key=lambda r: r["clearance_mm"])[:5]
        print("   tightest jaw clearance at the hover (occupied squares):", ", ".join(f"{r['square']} {r['clearance_mm']} mm ({r['nearest']})" for r in cl))
        hits = [r["square"] for r in ok if r.get("arm_contacts")]
        print("   arm contacts at the hover:", hits or "none")
    print("\nB. eval moves on the simulated follower (scripted stand-in skill):")
    for m in out["moves"]:
        ps = m["per_stage"]
        worst = min(((s, v) for s, v in ps.items() if s in ("approach", "retreat", "carry", "lift", "return")), key=lambda x: x[1]["clearance_mm"], default=(None, None))
        bc = {s: v["board_contact_mm"] for s, v in ps.items() if v["board_contact_mm"] > 0}
        lowered = m["info"].get("carry", {}).get("max_lowered_m", 0) * 100
        print(f"   {m['result'][0]:7s} {m['task'][9:]:42s} {'offset ' if m['offset_m'] else ''}retries {m['retries']} {m['duration_s']:5.1f} s "
              f"place err {m['placement_error_mm']:5.1f} mm | scripted-stage min clearance "
              f"{worst[1]['clearance_mm'] if worst[0] else '-'} mm ({worst[0]}: {worst[1]['nearest'] if worst[0] else ''}) "
              f"| carry lowered {lowered:.1f} cm{' | BOARD ' + str(bc) if bc else ''}{' | ' + m['result'][1] if m['result'][0] != 'placed' else ''}")


if __name__ == "__main__":
    main()
