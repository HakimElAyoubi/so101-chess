"""20-move evaluation set. Every trial starts from the same board (the drill setup); put the moved piece back by hand
afterwards. Validated: each source square holds the stated piece, each target is empty."""
import json
from pathlib import Path
D = json.loads((Path(__file__).parent.parent / "recording/drill_40.json").read_text())
board = {sq: ("black", p) for sq, p in D["setup_black"].items()} | {sq: ("white", p) for sq, p in D["setup_white"].items()}
TRIALS = [  # (category, piece, from, to)   to=None -> drop zone (white piece)
    ("common", "pawn", "e7", "e5"), ("common", "knight", "g8", "f6"), ("common", "knight", "b8", "c6"),
    ("common", "pawn", "d7", "d6"), ("common", "pawn", "c7", "c5"), ("common", "pawn", "h7", "h6"),
    ("common", "bishop", "f8", "d6"),
    ("drill_squares", "queen", "d8", "d2"), ("drill_squares", "bishop", "c8", "g4"), ("drill_squares", "rook", "h8", "h3"),
    ("drill_squares", "pawn", "e7", "e3"), ("drill_squares", "queen", "d8", "g4"),
    ("new_combination", "queen", "d8", "a5"), ("new_combination", "bishop", "c8", "f5"), ("new_combination", "rook", "a8", "a5"),
    ("new_combination", "bishop", "f8", "b4"), ("new_combination", "queen", "d8", "f6"),
    ("capture", "w:pawn", "c2", None), ("capture", "w:rook", "g1", None), ("capture", "w:bishop", "f1", None),
]
out = []
for i, (cat, piece, frm, to) in enumerate(TRIALS):
    col, p = ("white", piece[2:]) if piece.startswith("w:") else ("black", piece)
    assert board.get(frm) == (col, p), (i, frm, board.get(frm))
    if to:
        assert to not in board, (i, to)
        task = f"move the {col} {p} from {frm} to {to}"
    else:
        task = f"move the {col} {p} on {frm} to the drop zone"
    out.append(dict(i=i + 1, category=cat, task=task, frm=frm, to=to))
# how often does the training set contain each exact task?
tasks = [json.loads(l)["task"] for l in open(Path(__file__).parent.parent / "recording/episode_meta.jsonl")]
for t in out:
    t["train_count"] = tasks.count(t["task"])
json.dump(dict(setup_black=D["setup_black"], setup_white=D["setup_white"], trials=out),
          open(Path(__file__).parent / "eval_20.json", "w"), indent=1)
for t in out:
    print(f"{t['i']:2d} {t['category']:16s} {t['task']:45s} in training x{tasks.count(t['task'])}")
