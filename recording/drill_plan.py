"""Targeted 'drill' episodes to fill the gaps in LeHak0/so101_chess_moves before the first SmolVLA fine-tune.
Writes drill_40.json. Mostly round trips (the piece comes back), so the board needs setting up only once."""
import json
SETUP = {  # square -> piece; black start position + a few white pieces to capture from the empty half
    **{f"{f}8": p for f, p in zip("abcdefgh", ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"])},
    **{f"{f}7": "pawn" for f in "abcdefgh"},
}
WHITE = {"g1": "rook", "f1": "bishop", "c1": "bishop", "c2": "pawn", "a3": "pawn", "g3": "pawn", "e1": "king"}
# (piece, from, to)   to = "drop" for a white piece going to the drop zone, from = "spare" for a spare queen
DRILLS = [
    # deep half, long moves: queen tour (d2, g2, g4 never touched before)
    ("queen", "d8", "d2"), ("queen", "d2", "g2"), ("queen", "g2", "g4"), ("queen", "g4", "d8"),
    # rook tour on the h/f files (h2, f2, f3 never touched; h3 rare)
    ("rook", "h8", "h2"), ("rook", "h2", "f2"), ("rook", "f2", "f3"), ("rook", "f3", "h3"), ("rook", "h3", "h8"),
    # bishop tour (g4, e2, c4 never touched)
    ("bishop", "c8", "g4"), ("bishop", "g4", "e2"), ("bishop", "e2", "c4"), ("bishop", "c4", "a6"), ("bishop", "a6", "c8"),
    # knights in the crowded back rank (only 11 knight moves so far)
    ("knight", "b8", "c6"), ("knight", "c6", "e5"), ("knight", "e5", "c4"), ("knight", "c4", "a5"), ("knight", "a5", "c6"),
    ("knight", "c6", "b8"), ("knight", "g8", "h6"), ("knight", "h6", "g8"),
    # captures from the far half / far files (drop zone: only 4 from ranks 1-3 so far; c1, f1, g1 never touched)
    ("w:rook", "g1", "drop"), ("w:bishop", "f1", "drop"), ("w:bishop", "c1", "drop"), ("w:pawn", "c2", "drop"),
    ("w:pawn", "a3", "drop"), ("w:pawn", "g3", "drop"),
    # promotions (only 1 spare-queen episode so far)
    ("spare", "spare", "g1"), ("spare", "spare", "c1"),      # take each spare queen off by hand afterwards
    # long straight / diagonal moves (only 14 moves of 3+ squares)
    ("rook", "a8", "a4"), ("rook", "a4", "a8"), ("queen", "d8", "h4"), ("queen", "h4", "d8"),
    # pawn pushes into white's half (e3 never touched)
    ("pawn", "e7", "e5"), ("pawn", "e5", "e4"), ("pawn", "e4", "e3"), ("pawn", "e3", "e7"),
    # long rook move along the back rank side and back
    ("rook", "h8", "h5"), ("rook", "h5", "h8"),
]


def build():
    board = {sq: ("black", p) for sq, p in SETUP.items()} | {sq: ("white", p) for sq, p in WHITE.items()}
    out, spare_used = [], 0
    for i, (piece, frm, to) in enumerate(DRILLS):
        if piece == "spare":
            assert to not in board, (i, to)                  # the user removes it by hand after the episode
            out.append(dict(i=i, task=f"place the spare black queen on {to}", frm=None, to=to, to_drop=False, from_spare=True)); continue
        col, p = ("white", piece[2:]) if piece.startswith("w:") else ("black", piece)
        assert board.get(frm) == (col, p), (i, frm, board.get(frm), (col, p))
        if to == "drop":
            del board[frm]
            out.append(dict(i=i, task=f"move the {col} {p} on {frm} to the drop zone", frm=frm, to=None, to_drop=True, from_spare=False))
        else:
            assert to not in board, (i, to, board.get(to)); board[to] = board.pop(frm)
            out.append(dict(i=i, task=f"move the {col} {p} from {frm} to {to}", frm=frm, to=to, to_drop=False, from_spare=False))
    return out, board


if __name__ == "__main__":
    eps, final = build()
    json.dump(dict(setup_black=SETUP, setup_white=WHITE, episodes=eps), open("drill_40.json", "w"), indent=1)
    print(len(eps), "drill episodes OK; board at the end differs from the setup by:",
          {sq: v for sq, v in final.items() if ({**{s: ('black', p) for s, p in SETUP.items()}}.get(sq) != v and sq not in WHITE)})
