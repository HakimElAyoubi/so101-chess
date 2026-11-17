"""Grasp drill: the episodes recording/grasp_recorder.py walks through -> recording/grasp_drill.json.

Every episode is ONE grasp: both arms auto-move to the staged player's grasp hover above the square (the exact state the
skill starts from when playing), you take over with the leader, grasp the piece, lift it ~3 cm, Stop. The piece is put
off-centre on purpose, at one of 9 spots in its square, while the skill's green dot stays at the square's centre (as when
playing): the model learns to grasp the piece, not the dot.

Board: the eval setup (black: the start position; white: rook g1, bishops f1 + c1, pawns c2 / a3 / g3, king e1), the
same as play/eval_20.json. Three parts:
  A  the 16 black pieces at home (crowded back ranks), 3 spots each                       48
  C  the 7 white eval pieces at home, 3 spots each                                         21
  B  tours over the open board: each of the 12 kinds (6 types x 2 colours) visits 14 empty   168
     squares of ranks 1-6, a different spot each time
Run:  python recording/grasp_plan.py      (validates every hover on the latest calibrated board pose)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "grasp_drill.json"
BLACK = {**{f"{f}8": p for f, p in zip("abcdefgh", ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"])},
         **{f"{f}7": "pawn" for f in "abcdefgh"}}
WHITE = {"g1": "rook", "f1": "bishop", "c1": "bishop", "c2": "pawn", "a3": "pawn", "g3": "pawn", "e1": "king"}
# 9 spots in the square: (mm toward rank 8, mm toward the h-file) and how the page says it
SPOT_MM = 6.0
SPOTS = {"centre": (0, 0), "toward rank 8": (1, 0), "toward rank 1": (-1, 0), "toward the h-file": (0, 1),
         "toward the a-file": (0, -1), "toward the h8 corner": (.7, .7), "toward the a8 corner": (.7, -.7),
         "toward the h1 corner": (-.7, .7), "toward the a1 corner": (-.7, -.7)}
TOUR_SQUARES = 14
# where each tour piece comes from (white: the spare from the box unless the set has only the one on the board)
TOUR_FROM = {("black", "pawn"): "e7", ("black", "knight"): "b8", ("black", "bishop"): "c8", ("black", "rook"): "h8",
             ("black", "queen"): "d8", ("black", "king"): "e8", ("white", "pawn"): "box", ("white", "knight"): "box",
             ("white", "bishop"): "c1", ("white", "rook"): "box", ("white", "queen"): "box", ("white", "king"): "e1"}


def plan(seed=0):
    rng = np.random.default_rng(seed)
    names = list(SPOTS)
    eps = []

    def add(part, colour, piece, sq, spot, note=""):
        eps.append(dict(i=len(eps), part=part, colour=colour, piece=piece, square=sq, spot=spot,
                        offset_mm=[round(SPOT_MM * SPOTS[spot][0], 1), round(SPOT_MM * SPOTS[spot][1], 1)],
                        task=f"pick up the {colour} {piece}", note=note))

    for part, home, colour in (("A", BLACK, "black"), ("C", WHITE, "white")):
        for sq, piece in home.items():
            for spot in rng.choice(names, 3, replace=False):     # 3 different spots per piece, centre included at random
                add(part, colour, piece, sq, str(spot))
    occupied = set(BLACK) | set(WHITE)
    empty = [f"{f}{r}" for r in range(1, 7) for f in "abcdefgh" if f"{f}{r}" not in occupied]
    by_rank = {r: [s for s in empty if s[1] == str(r)] for r in range(1, 7)}
    for (colour, piece), src in TOUR_FROM.items():
        squares = []
        for r in range(1, 7):                                    # 2-3 squares per rank, files mixed
            squares += list(rng.choice(by_rank[r], 3 if r in (1, 3) else 2, replace=False))
        squares = squares[:TOUR_SQUARES]
        rng.shuffle(squares)
        spots = list(rng.permutation(names)) + list(rng.permutation(names))
        where = "a spare one from the box" if src == "box" else f"the one on {src}"
        for k, sq in enumerate(squares):
            note = (f"Tour of the {colour} {piece}: take {where}." if k == 0 else "") + \
                   (f" Last of this tour: afterwards put it back {'in the box' if src == 'box' else 'on ' + src}." if k == len(squares) - 1 else "")
            add("B", colour, piece, str(sq), str(spots[k]), note.strip())
    return eps


def validate(eps):
    """Every hover on the latest calibrated board pose: solvable, and a wrist the leader can follow (limit 102.4 deg)."""
    sys.path[:0] = [str(ROOT / "play"), str(ROOT / "goal")]
    import staged as S
    import targets
    P = targets.Projector(); Pl = S.Planner(P); sess = targets.LATEST
    ready = json.loads((HERE / "ready_pose.json").read_text())
    r6 = np.array([ready[j] for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]])
    bad, wrist = [], []
    for e in eps:
        q, info = Pl.grasp_hover(P.square_xyz(e["square"], sess), e["piece"], r6, S.HOVER_GRIP)
        if q is None:
            bad.append((e["i"], e["square"], e["piece"]))
        else:
            wrist.append(q[3])
    print(f"board pose {sess}: {len(eps) - len(bad)}/{len(eps)} hovers solvable; wrist flex {min(wrist):.0f}..{max(wrist):.0f} deg"
          + (f"; UNSOLVABLE: {bad}" if bad else ""))
    return not bad


if __name__ == "__main__":
    eps = plan()
    parts = {p: sum(e["part"] == p for e in eps) for p in "ACB"}
    kinds = {}
    for e in eps:
        kinds[f"{e['colour']} {e['piece']}"] = kinds.get(f"{e['colour']} {e['piece']}", 0) + 1
    print(len(eps), "grasps:", parts, "| per kind:", kinds)
    print("spots:", {s: sum(e["spot"] == s for e in eps) for s in SPOTS})
    if validate(eps):
        OUT.write_text(json.dumps(dict(setup_black=BLACK, setup_white=WHITE, spot_mm=SPOT_MM, episodes=eps), indent=1))
        print("->", OUT)
