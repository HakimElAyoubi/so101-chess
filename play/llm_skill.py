"""Claude as the grasp / place skill of the staged player (play/policy_io.load_policy picks it for --grasp-repo /
--model-repo "claude:<model>"; "claude:mock" is a stand-in that reads the twin's truth, for dry runs without the API).

The player still does everything around the skill: IK approach to the hover (sag removed), carry, the hold / miss /
release rules, retries, logging. At each skill stage this class runs a look-move-verify loop in which Claude is the
EYES only and the calibrated camera model does the measuring:

  LOOK     the arm is settled; the raw wrist frame is cropped around the landing point, annotated and sent to Claude,
           who answers with ONE PIXEL: the centre of the target piece's top (grasp) or the centre of the printed
           target square (place), plus flags (visible, clear, knight roll, drawn-vs-real fingertip offset).
  MEASURE  that pixel is back-projected with the real fisheye model onto the plane where the pointed feature lives
           (the piece's top at its known height, the board for a square), and compared with LAND: the point where
           the feature must be for a vertical descent to put the stem between the pads, from the camera-to-jaw
           geometry only (the twin's board / arm bias of up to ~8 mm never enters). No metric estimate is asked of
           the model: pointing is what a vision model does well, counting millimetres is not. Pointing at the TOP
           matters: from the wrist camera (about 29 deg off vertical) only the head of a piece is visible, and
           back-projecting the head onto the stem's height would land 4-5 mm beyond the stem.
  MOVE     the fingertips are moved laterally by the measured offset (numerical Jacobian on the measured joints, tilt
           held, min-jerk, the integral sag compensation of the player's hold stage kept running), then LOOK again,
           until the offset is within tolerance (2 mm) or the moves per phase are used up.
  DESCEND  straight down to the humans' close height for the piece (drill medians), slowly (P=16 lag), jaws 12 %,
           one or two LOOKs at the bottom with a last correction, then CLOSE (0.8 %): the player detects the hold or
           the miss; a miss retries the stage.
  PLACE    the same loop on the printed target square with the held piece's base, LOWER to ~3 mm, RELEASE.

Every look is logged with its image, the answer, the measurement and (twin dry runs) the truth under
play/results*/claude/<trial>/, for the journal and for tuning. Timing: ~10-15 s per look (claude-opus-5-5, effort
low), 2-4 looks per attempt; the stage timeouts are raised accordingly (timeout_s).
"""
import base64
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "goal"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30
W_RAW, H_RAW = 1280, 720
# the humans' close heights above the board (cm, drill medians, data/gate_20261105/drill_human_closes.csv) and the
# real stems they held (mm, the piece manifest), which the pads must sit on
CLOSE_H_CM = {"pawn": 1.35, "rook": 2.03, "knight": 1.71, "bishop": 2.27, "queen": 2.81, "king": 2.68}
STEM_MM = {"pawn": 5, "rook": 10, "knight": 10, "bishop": 7, "queen": 7, "king": 6}
PIECE_H_CM = {"pawn": 2.5, "rook": 2.8, "knight": 3.1, "bishop": 3.6, "queen": 4.0, "king": 5.0}
# height above the board of the point Claude is asked to point at: the centre of the piece's top as seen from above
# (chess_sim/assets/pieces/manifest.json profiles: pawn ball 20-25 mm, rook top face 28, bishop ball 31-36, queen crown
# 34-40, king cross 45-50); the knight's top is off its axis, so its round base on the board is used instead
TOP_Z_CM = {"pawn": 2.15, "rook": 2.7, "knight": 0.2, "bishop": 3.3, "queen": 3.7, "king": 4.5}
# the real set may differ from the twin's meshes: data/piece_tops_cm.json {"pawn": 2.3, ...} (height of the pointed
# feature above the board, cm: the ball / crown / cross centre as seen from above) overrides the table
try:
    TOP_Z_CM.update({k: float(v) for k, v in json.loads((ROOT / "data/piece_tops_cm.json").read_text()).items() if k in TOP_Z_CM})
    print(f"[claude skill] piece top heights from data/piece_tops_cm.json: {TOP_Z_CM}", file=sys.stderr, flush=True)
except FileNotFoundError:
    pass
except Exception as e:
    print(f"[claude skill] data/piece_tops_cm.json ignored: {e!r}", file=sys.stderr, flush=True)
TOP_NAME = {"pawn": "the middle of its round head (the ball on top)", "rook": "the middle of its flat crenellated top",
            "knight": "the centre of its round BASE where it stands on the board", "bishop": "the middle of the small ball on top",
            "queen": "the middle of its crown (the top)", "king": "the middle of the cross on top"}
# Opening (real arm, 2026-11-10 evening): the twin's fingertip sites are 6.6 mm apart with the pads touching and 16.7 mm at
# 15 %, so the pad FACES were only ~10 mm apart during the descent: a pawn's head (10-12 mm) does not fit. The pads rode
# over the head, shoved it 3-6 mm (what the bottom look saw) and the close shut on nothing; the two holds were lucky
# squeezes of the ball's tip (follower 3 %). The humans opened 9-13 % and squeezed with the sponge, which a blind
# descent cannot repeat. 30 % = 27.3 mm between the sites = 20.7 mm between the faces: +-5 mm around a pawn head. The
# jaws close along the FILE and the MOVING pad swings toward the square in front (empty for a 7th-rank pawn, 4 mm clear
# of a piece there); the fixed pad stays on the rank-8 side. The release opens as wide so the piece is not dragged.
HOVER_GRIP, OPEN_GRIP, CLOSE_GRIP, RELEASE_GRIP = 10.0, 30.0, 0.8, 30.0
LAND_FROM_FIXED_MM = 10.0   # the aim point: 3.3 mm of pad + a pawn's half width 5 + 1.7 mm margin from the FIXED fingertip site
GRID_MM, GRID_HALF_MM = 10, 30                   # the faint scale grid: spacing and half-extent around LAND
CROP, SCALE = 560, 1.5                           # px of the raw frame around the landing point, upscale for Claude
MAX_LOOKS = 14                                    # per stage attempt (the player's retry starts a fresh count)
# mid-descent look (real arm, 2026-11-10): a camera-converged hover does not survive the 5 cm descent. The joint traces
# show the commanded and measured fingertips 3-8 mm apart at the bottom (the model's pose error differs between the two
# configurations), the bottom look then finds the pawn 3-6 mm off, and every first e7 attempt missed. So the descent
# stops where the pads are MID_CLEAR_M above the piece's top (the top still visible between them), looks and corrects
# again, and only then goes to the close height. Not yet for the bishop / queen verified straight descent (their
# recorded-fixture checks encode that flow; the f8 bishop showed the same drift at 20:03: model 0.6 mm, closed on nothing).
MID_LOOK, MID_CLEAR_M = True, 0.012
MID_LOOK_VERIFIED = True    # also for the bishop / queen verified straight descent (real arm 2026-11-10: it missed every time;
# the recorded-fixture checks switch this off because they encode the straight path)
LOOK_CLEAR_M = 0.035                             # grasp: the pads this far above the piece's top for the hover looks (else the
                                                 # sponge pads hide the top of a tall piece in the wrist camera, real arm 2026-11-10)
MAX_HOVER_MOVES, MAX_LOW_MOVES = 4, 2            # lateral corrections before descending / before closing
MAX_STEP_MM = 15.0                               # per correction
TOL_MM = {"from_hover": 2.0, "from_low": 2.0, "to_hover": 2.0, "to_low": 2.5}
TIPS_OFFSET_MAX_PX = 25                          # a reported drawn-vs-real fingertip offset is applied only up to this
REG_MATCH_MM, REG_MAX_MM = 9.0, 12.0             # grid intersections: max distance to the believed lattice; max total board shift
HAND_EYE_MM = None                               # (dx, dy) mm added to LAND once the real pads' offset from the model is measured
PADS_APPLY_MAX_MM = 12.0                         # the measured pad-gap offset is applied up to this (real arm 2026-11-10: the
# model reports the real pads ~(4.5, 4.5) mm = 6.4 mm off the twin's at nearly every hover look; the old 6 mm gate dropped
# it by 0.4 mm and the pawn sat at the pad's edge)
DRAW_LAND = False                                # the goal marker anchored Claude's answers on itself (real arm): not drawn
MAX_DESCEND_MM = 3.0
# At the bottom the sponge pads hide a pawn's top, so the bottom look often answers 'target None' (real arm, 2026-11-10:
# every pawn after a7/b7). When the hover had converged (<= MAX_DESCEND_MM) that means the piece is between the pads: close.
# A real miss still ends in 'closed on nothing' (staged.py), which retries.
CLOSE_ON_HIDDEN_TARGET = True                             # never descend with the hover estimate farther off than this
TRY_OFFSET_MM = (0.0, 3.0, -3.0)                 # retries probe the hand-eye along the jaw axis (a hold reveals the bias)
SAG_GAIN, SAG_MAX_DEG = 0.15, 5.0                # the player's own sag integrator (play/staged.py)
SETTLE_TOL_DEG, SETTLE_S, SETTLE_MAX_S = 0.25, 0.35, 1.5
MOVE_S, DESCEND_MM_S = 0.7, 25.0                 # lateral move duration; descent speed (slow: the P=16 lag)
MODEL_DEFAULT = "claude-opus-5-5"
EFFORT = os.environ.get("CLAUDE_EFFORT", "low")   # thinking cannot be switched off on Opus 5.5; low = 300-600 output tokens
MOCK_NOISE_PX = float(os.environ.get("CLAUDE_MOCK_NOISE_PX", "0"))
SESSION_DIRS = {}                                # one results directory per provider for this player process

SYSTEM_PROMPT = """You are the eyes of a small robot arm (an SO-101) that plays chess as Black on a real wooden board with
dark and light squares. Each message is ONE annotated crop of the arm's wrist camera (a fisheye lens next to the
gripper, looking down along the jaws) plus a short text. The two jaws come into the image from its right edge, one
above the other, each ending in a 10 mm white sponge pad; the pads face each other and close vertically in the image
(the fixed jaw is the lower one). You do not steer the arm: you locate things in
the image as precisely as you can, and the software converts your pixels into motion with the calibrated camera model.
A pixel error of 10 px is about 1 mm, so aim for the exact pixel.

Overlays drawn on the image:
  - two small cyan circles labelled tip, and a yellow X labelled PADS: where the software expects the fingertips and
    the centre of the gap between the two sponge pads (at the pads' own height). In every look, find the real centre of
    that gap, midway between the two pads' facing surfaces at their tips, and report real minus drawn PADS in
    pads_offset_px (pixels; [0, 0] only if it really coincides). This calibrates the hand-eye: be careful with it.
  - the target SQUARE outlined from the software's belief. For a grasp (green): a dashed outline on the printed square
    and a solid outline lifted to the height of the piece tops, joined by an arrow; because the camera looks obliquely
    the tops of the pieces standing on that square appear inside the SOLID outline, a couple of cm away from the
    printed square. For a place (magenta): the printed square. The belief can be off by up to 10 mm (a third of a
    square): the outline says WHICH square or piece is meant, never where it exactly is.
  - a faint white grid, 10 mm spacing, for scale.
The board's printed file letters and rank numbers along its edges, when in view, confirm which square is which.

In EVERY look, report in grid_px up to 4 intersections of the PRINTED board grid that you can see clearly: points
where the corners of four squares meet (two dark, two light), anywhere in the image, preferring well-lit ones near the
target and away from the gripper's shadow. Give only intersections you really see, as precisely as you can; an empty
list is fine. Never report the drawn outline's corners, a square's edge midpoint, a piece's base or a shadow edge as an
intersection: the software matches each point to the board model, and a wrong point moves the arm the wrong way.

GRASP look: the text names the target piece (type, colour, square). It is the piece of that kind whose top is inside
or closest to the green outline. Pieces of the same kind often stand on the neighbouring squares, 28 mm away: do not
pick them. Set target_px to the pixel of the CENTRE OF ITS TOP as seen from above, as the text specifies (for example
the middle of a pawn's ball). Set clear=false if another piece or a jaw stands in the gripper's way within about
12 mm of the target (the jaws descend around the target). For a knight, also give roll_deg: the angle (-90..90) to
turn the jaws so that they close across its narrow neck; otherwise 0.

PLACE look: the arm holds a piece above the board; the text names the target square. The grid intersections you report
are what locates it, so look for them carefully (the square's own corners are intersections too, when visible). Also
set target_px to your estimate of the real square's centre (squares are 28 mm wide; infer it from the visible edges
when the centre itself is hidden by the held piece or the jaws).

Answer with ONE JSON object only, no prose outside it:
{"visible": true, "target_px": [u, v], "grid_px": [[u, v], [u, v]], "pads_offset_px": [0, 0], "clear": true,
 "roll_deg": 0.0, "confidence": 0.0, "notes": "under 20 words"}
visible=false (with your best target_px or null) when the target piece / square cannot be found at all."""


def _b64_png(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in the answer")
    return json.loads(m.group(0))


def _put(img, text, org, col, scale=0.5):
    org = tuple(int(round(v)) for v in org)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


class ClaudeSkill:
    """Policy interface of play/policy_io.Policy: reset / act(q6, img, task) / chunk / fixed_noise; goal and session set
    by the player; `staged` and `player` set by the SkillDriver (the move in progress)."""
    type = "claude"
    model_default = MODEL_DEFAULT
    max_looks = MAX_LOOKS
    strict_measurements = False

    def __init__(self, checkpoint="final", variant="skills", repo=None):
        from targets import LATEST, Projector
        import staged as S
        model = (repo or "").split(":", 1)[1] if repo and ":" in repo else self.model_default
        self.mock = model == "mock"
        self.model = self.model_default if self.mock else model
        self.checkpoint, self.variant = checkpoint, variant
        self.proj, self.session = Projector(), LATEST
        self.Pl = S.Planner(self.proj)
        self.scene = self.proj.s
        self.goal, self.staged, self.player = None, None, None
        self.n_steps, self.timeout_s = 1, 90.0
        self.client = None
        self.log_dir = None
        self.looks_total, self.cost_calls = 0, 0
        self.reset()

    # ------------------------------------------------------------------ player interface
    def reset(self):
        self._look_generation = getattr(self, "_look_generation", 0) + 1
        self.phase = "init"
        self.desired, self.bias, self.grip = None, np.zeros(6), HOVER_GRIP
        self.move_from, self.move_to, self.move_t0, self.move_T = None, None, None, 0
        self.settle_t0, self.settle_start = None, None
        self.look_n, self.pending, self.answer, self.err = 0, None, None, None
        self.moves_hover, self.moves_low = 0, 0
        self.history = []
        self.z_plane, self.z_board = None, None
        self.low = False                                   # True once the descent (grasp) / the lowering (place) is done
        self.gave_up = None                                # the reason this attempt stopped (the player retries at once)
        self.last_dist, self.jumps = None, 0               # the previous look's offset; jumps refused after a converged look
        self.descended_from_mm = None                      # the hover offset at the descend decision (blind close at the bottom)
        self.mid_done = False                              # the mid-descent look has been taken (or does not apply)
        self.lost_relooks = 0                              # relooks from higher up after the target vanished behind the pads
        self.hint_ring = False                             # recovery looks: a thin ring around the expected position (never a dot)
        self.low_relooks = 0                               # bottom look saw nothing -> back up to the mid height and look again
        self.lowconf_prev, self.lowconf_relooks = None, 0  # a low-confidence look is re-taken; two that agree count as confirmed
        self.measured_from_xy = None                       # world xy of the piece from the latest visible look (the player re-plans a retry there)
        self.raised = False                                # the first look is taken from higher up (the pads hide the top)
        self.t_phase = time.monotonic()
        self.last_look_img = None

    def fixed_noise(self, seed=0):
        return None

    def chunk(self, state6, image_hwc_uint8, task, noise=None):
        return np.array([self.act(state6, image_hwc_uint8, task)], np.float32)

    # ------------------------------------------------------------------ geometry on the calibrated twin
    def _set(self, q6):
        self.scene.set_joints(np.asarray(q6, float))

    def tips(self, q6):
        """(LAND, tip A, tip B). LAND is the aim point for the piece: LAND_FROM_FIXED_MM along the jaw axis from tip A, the
        FIXED jaw (the twin: A moves 0 mm, B 9.9 mm between 15 and 1 %). The piece must sit against the fixed pad, so the
        moving pad pinches it instead of shoving it across the gap; the geometric midpoint moves 0.35 mm per % of
        opening and left a 30 %-open pawn 10 mm from the fixed pad (twin 2026-11-10: every first attempt failed)."""
        self._set(q6)
        a, b = (np.asarray(v, float) for v in self.scene.jaw_tip_positions())
        axis = b - a; gap = float(np.linalg.norm(axis))
        land = a + axis * (min(LAND_FROM_FIXED_MM / 1000.0, 0.5 * gap) / max(gap, 1e-6))
        return land, a, b

    def _tips_tilt(self, q6):
        t, _, _ = self.tips(q6)
        tilt, _ = self.Pl.orientation(q6)
        return np.array([t[0], t[1], t[2], np.radians(tilt)])

    def jacobian_step(self, q6, d_tips, max_deg=3.0, damping=0.3):
        """Joint deltas (pan, lift, elbow, wrist flex) moving the fingertips by d_tips (m) with the jaw tilt held:
        damped least squares on rows scaled to mm / deg, the step shrunk as a whole (never clipped per joint: that
        changes its direction, which sent one 11 mm move 18 mm the wrong way near the board's edge)."""
        q6 = np.asarray(q6, float); base = self._tips_tilt(q6)
        J = np.zeros((4, 4)); h = 0.5
        for j in range(4):
            qp = q6.copy(); qp[j] += h
            J[:, j] = (self._tips_tilt(qp) - base) / h
        W = np.diag([1000.0, 1000.0, 1000.0, 180.0 / np.pi])        # m -> mm, rad -> deg: comparable rows
        Jw = W @ J; bw = W @ np.array([d_tips[0], d_tips[1], d_tips[2], 0.0])
        dq = np.linalg.solve(Jw.T @ Jw + damping ** 2 * np.eye(4), Jw.T @ bw)
        m = np.abs(dq).max()
        return dq * (max_deg / m) if m > max_deg else dq

    def joints_for_tips(self, q6, target_tips, iters=10, tol=1.5e-3):
        """Joint vector whose fingertips are at target_tips (m), same tilt and roll, by Jacobian iterations; None when
        the solution does not reach within tol (the caller shortens the move or skips it)."""
        q = np.asarray(q6, float).copy()
        best, best_err = q.copy(), np.inf
        for _ in range(iters):
            t, _, _ = self.tips(q)
            d = np.asarray(target_tips) - t
            err = float(np.linalg.norm(d))
            if err < best_err:
                best, best_err = q.copy(), err
            if err < 2e-4:
                break
            q[:4] += self.jacobian_step(q, d)
        t, _, _ = self.tips(q)
        if np.linalg.norm(np.asarray(target_tips) - t) < best_err:
            best, best_err = q, float(np.linalg.norm(np.asarray(target_tips) - t))
        return best if best_err <= tol else None

    def project(self, q6, pts):
        uv, vis = self.proj.project(q6, np.atleast_2d(np.asarray(pts, float)), W_RAW)
        return uv, vis

    def backproject(self, q6, uv_raw, z):
        """A raw-frame pixel -> world xy on the horizontal plane at height z (the real fisheye model)."""
        return np.asarray(self.proj.backproject(q6, np.atleast_2d(np.asarray(uv_raw, float)), W_RAW, z=z), float)[0]

    def crop_to_raw(self, meta, uv):
        return np.asarray(uv, float) / meta["scale"] + np.array([meta["x0"], meta["y0"]], float)

    def squares(self):
        """(from square, to square or 'drop zone') of the move in progress."""
        task = self.staged.task if self.staged is not None else (self.goal or {}).get("move", "")
        m = re.search(r"(?:from|on) ([a-h][1-8]) to (?:the )?([a-h][1-8]|drop zone)", task or "")
        return (m.group(1), m.group(2)) if m else (None, None)

    # ---- the board model and its registration from the printed grid intersections Claude reports
    def _reg(self):
        """The board registration shared by both skill instances of a player (world xy shift of the believed board model,
        from the printed grid intersections seen in the looks); reset when the board is measured again (session)."""
        host = self.player if self.player is not None else self
        reg = getattr(host, "claude_reg", None)
        if reg is None or reg.get("session") != self.session:
            reg = {"session": self.session, "t": np.zeros(2), "n": 0, "looks": 0, "resid_mm": None}
            try:
                host.claude_reg = reg
            except Exception:
                pass
        return reg

    def bp(self, f, r, z=0.0, registered=True):
        """World xyz of a board point (continuous file / rank coordinates, see targets.Projector.board_point) under the
        believed board model, shifted by the current registration."""
        p = np.asarray(self.proj.board_point(f, r, self.session, z), float)
        if registered:
            p[:2] += self._reg()["t"]
        return p

    def lattice_w(self):
        """All 81 printed grid intersections of the believed board (world xy), registered."""
        if getattr(self, "_lat", None) is None:
            self._lat_fr = [(f + 0.5, r + 0.5) for f in range(-1, 8) for r in range(0, 9)]
            self._lat = np.array([self.proj.board_point(f, r, self.session, 0.0)[:2] for f, r in self._lat_fr], float)
        return self._lat + self._reg()["t"]

    def register(self, ans, meta):
        """Match Claude's grid_px to the nearest lattice intersections and shift the board model by their mean offset
        (outliers dropped). Returns a dict for the log."""
        pts = ans.get("grid_px") or []
        q = np.asarray(meta["q"], float); lat = self.lattice_w()
        offs, matched = [], []
        for px in pts:
            try:
                px = np.asarray(px, float).ravel()[:2]
            except Exception:
                continue
            if len(px) != 2 or not np.all(np.isfinite(px)):
                continue
            xy = self.backproject(q, self.crop_to_raw(meta, px), meta["z_board"])
            k = int(np.argmin(np.linalg.norm(lat - xy, axis=1)))
            off = xy - lat[k]
            if np.linalg.norm(off) <= REG_MATCH_MM / 1000.0:           # farther from every intersection than the belief can be off: not one
                offs.append(off); matched.append(self._lat_fr[k])
        info = {"grid_n": len(pts), "grid_matched": len(offs)}
        if not offs:
            return info
        O = np.array(offs)
        med = np.median(O, axis=0)
        keep = np.linalg.norm(O - med, axis=1) <= 0.004 if len(O) >= 3 else np.ones(len(O), bool)
        if len(O) == 2 and np.linalg.norm(O[0] - O[1]) > 0.005:
            info["grid_disagree_mm"] = float(1000 * np.linalg.norm(O[0] - O[1])); return info
        O = O[keep]
        reg = self._reg()
        step = O.mean(axis=0)
        t = reg["t"] + step
        if np.linalg.norm(t) > REG_MAX_MM / 1000.0:                      # the board model is not that far off: a wrong match
            info["grid_rejected_mm"] = (1000 * t).tolist(); return info
        reg["t"] = t; reg["n"] = int(len(O)); reg["looks"] += 1
        reg["resid_mm"] = float(1000 * np.linalg.norm(O - step, axis=1).max()) if len(O) > 1 else 0.0
        info.update(grid_used=int(len(O)), grid_step_mm=(1000 * step).tolist(), reg_t_mm=(1000 * reg["t"]).tolist(), reg_resid_mm=reg["resid_mm"])
        return info

    def square_centre_w(self, sq, z=0.0):
        return self.bp("abcdefgh".index(sq[0]), int(sq[1]), z)

    def anchor(self, q6, meta):
        """The pads close on the target piece, which stands on the target square: the strongest observation of where the
        board really is (pieces sit within a few mm of their square's centre). Shifts the board model (capped)."""
        try:
            sq = meta.get("square")
            if not sq:
                return
            tips_xy = self.tips(q6)[0][:2]
            off = tips_xy - self.square_centre_w(sq)[:2]
            if np.linalg.norm(off) > 0.014:                               # not on that square: do not learn from it
                return
            reg = self._reg(); t = reg["t"] + 0.8 * off                   # 0.8: pieces are not perfectly centred
            if np.linalg.norm(t) <= REG_MAX_MM / 1000.0:
                reg["t"] = t; reg["looks"] += 1; reg["anchor_mm"] = (1000 * off).tolist()
                print(f"[{self.type} skill] board model anchored on the grasped piece: shift ({1000 * off[0]:+.1f}, {1000 * off[1]:+.1f}) mm -> model ({1000 * t[0]:+.1f}, {1000 * t[1]:+.1f}) mm", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[{self.type} skill] anchor failed: {e!r}", file=sys.stderr, flush=True)

    def _targets(self):
        from_xyz = np.asarray(self.staged.from_xyz if self.staged is not None else self.proj.target_xyz(self.goal["move"], self.session)[0], float)
        to_xyz = np.asarray(self.goal.get("to_xyz") if self.goal is not None and self.goal.get("to_xyz") is not None else
                            (self.staged.to_xyz if self.staged is not None else self.proj.target_xyz(self.goal["move"], self.session)[1]), float)
        piece = self.staged.piece if self.staged is not None else "pawn"
        return from_xyz, to_xyz, piece

    # ------------------------------------------------------------------ the annotated image
    def annotate(self, q6, img_rgb, kind):
        """Crop + overlays on the RAW frame (the projector uses the real fisheye model). Returns (bgr, meta)."""
        from_xyz, to_xyz, piece = self._targets()
        tips_mid, tip_a, tip_b = self.tips(q6)
        sq_from, sq_to = self.squares()
        if kind == "from":
            z_board = float(from_xyz[2])
            z_plane = z_board + TOP_Z_CM[piece] / 100.0           # where the pointed feature (the piece's top) lives
            c = self.square_centre_w(sq_from)[:2] if sq_from else from_xyz[:2]
            planned = np.array([c[0], c[1], z_plane])             # the believed piece position, registered
        else:
            z_board = float(to_xyz[2])
            z_plane = z_board                                     # the printed square centre
            c = self.square_centre_w(sq_to)[:2] if sq_to and sq_to != "drop zone" else to_xyz[:2]
            planned = np.array([c[0], c[1], z_plane])
        self.z_plane, self.z_board = z_plane, z_board
        land_w = np.array([tips_mid[0], tips_mid[1], z_plane])    # the feature must be here for a vertical descent to fit
        uv_land = self.project(q6, land_w)[0][0]
        uv_plan = self.project(q6, planned)[0][0]
        uv_tips = self.project(q6, np.stack([tip_a, tip_b]))[0]
        if not np.all(np.isfinite(uv_land)):
            uv_land = np.array([W_RAW * 0.55, H_RAW * 0.5])
        c = uv_land if not np.all(np.isfinite(uv_plan)) else 0.5 * (uv_land + uv_plan)
        x0 = int(np.clip(c[0] - CROP / 2, 0, W_RAW - CROP)); y0 = int(np.clip(c[1] - CROP / 2, 0, H_RAW - CROP))
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
        # the faint scale grid on the pointed plane, world axes, centred on LAND
        n = GRID_HALF_MM // GRID_MM
        layer = img.copy()
        for axis in (0, 1):
            for k in range(-n, n + 1):
                pts = []
                for m in np.linspace(-GRID_HALF_MM, GRID_HALF_MM, 13):
                    p = land_w.copy(); p[axis] += k * GRID_MM / 1000.0; p[1 - axis] += m / 1000.0
                    pts.append(p)
                uv = self.project(q6, np.array(pts))[0]
                for i in range(len(uv) - 1):
                    if np.all(np.isfinite(uv[i])) and np.all(np.isfinite(uv[i + 1])):
                        cv2.line(layer, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[i + 1]).astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.addWeighted(layer, 0.35, img, 0.65, 0)
        pa = land_w.copy(); pa[0] -= (GRID_HALF_MM + 5) / 1000.0; pb = pa.copy(); pb[1] += 0.010
        ua, ub = self.project(q6, np.stack([pa, pb]))[0]
        if np.all(np.isfinite(ua)) and np.all(np.isfinite(ub)):
            cv2.line(img, tuple(np.round(ua).astype(int)), tuple(np.round(ub).astype(int)), (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(img, tuple(np.round(ua).astype(int)), tuple(np.round(ub).astype(int)), (255, 255, 255), 2, cv2.LINE_AA)
            _put(img, "10 mm", 0.5 * (ua + ub) + [0, -8], (255, 255, 255))
        # markers: LAND as a hollow circle (the feature under it stays visible), the tips, the planned dot / ring
        if DRAW_LAND:
            u, v = np.round(uv_land).astype(int)
            cv2.circle(img, (u, v), 11, (0, 0, 0), 4, cv2.LINE_AA); cv2.circle(img, (u, v), 11, (0, 0, 255), 2, cv2.LINE_AA)
            _put(img, "LAND", (u + 13, v - 8), (0, 0, 255), 0.55)
        for t in uv_tips:
            if np.all(np.isfinite(t)):
                cv2.circle(img, tuple(np.round(t).astype(int)), 6, (255, 255, 0), 2, cv2.LINE_AA)
                _put(img, "tip", t + [8, 4], (255, 255, 0), 0.45)
        uv_pads = self.project(q6, tips_mid)[0][0]                    # the model's gap centre between the pads, at their own height
        if np.all(np.isfinite(uv_pads)):
            u, v = np.round(uv_pads).astype(int)
            cv2.drawMarker(img, (u, v), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 16, 4, cv2.LINE_AA)
            cv2.drawMarker(img, (u, v), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
            _put(img, "PADS", (u + 10, v + 16), (0, 255, 255), 0.5)
        # the believed (registered) square: its outline, at the piece tops' height for a grasp (the heads of the pieces on
        # it appear inside; on the board plane a neighbour's head would, from this oblique camera) and on the board for a
        # place; sampled edges because the fisheye bends straight lines
        sq = sq_from if kind == "from" else (sq_to if sq_to != "drop zone" else None)
        col = (0, 200, 0) if kind == "from" else (255, 0, 255)
        centre_w = None
        if sq is not None:
            centre_w = self.square_centre_w(sq)
            f, r = "abcdefgh".index(sq[0]), int(sq[1])
            z_out = z_plane - z_board if kind == "from" else 0.0
            ring_pts = []
            for (df0, dr0), (df1, dr1) in (((-0.5, -0.5), (0.5, -0.5)), ((0.5, -0.5), (0.5, 0.5)), ((0.5, 0.5), (-0.5, 0.5)), ((-0.5, 0.5), (-0.5, -0.5))):
                for t in np.linspace(0, 1, 6, endpoint=False):
                    ring_pts.append(self.bp(f + df0 + t * (df1 - df0), r + dr0 + t * (dr1 - dr0), z_out))
            uv_ring = self.project(q6, np.array(ring_pts))[0]
            for i in range(len(uv_ring)):
                a, b = uv_ring[i], uv_ring[(i + 1) % len(uv_ring)]
                if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                    cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, 1, cv2.LINE_AA)
            if kind == "from" and z_out > 0:
                # the same square printed on the board (dashed) and a line to where the tops of its pieces appear: the
                # camera is oblique, so the lifted outline sits a couple of cm away from the printed square
                uv_b = self.project(q6, np.array([self.bp(f + df, r + dr, 0.0) for df, dr in
                                                  [(-0.5 + t, -0.5) for t in np.linspace(0, 1, 6, endpoint=False)] +
                                                  [(0.5, -0.5 + t) for t in np.linspace(0, 1, 6, endpoint=False)] +
                                                  [(0.5 - t, 0.5) for t in np.linspace(0, 1, 6, endpoint=False)] +
                                                  [(-0.5, 0.5 - t) for t in np.linspace(0, 1, 6, endpoint=False)]]))[0]
                for i in range(0, len(uv_b), 2):
                    a, b = uv_b[i], uv_b[(i + 1) % len(uv_b)]
                    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, 1, cv2.LINE_AA)
                c_b = self.project(q6, self.bp(f, r, 0.0))[0][0]; c_t = self.project(q6, self.bp(f, r, z_out))[0][0]
                if np.all(np.isfinite(c_b)) and np.all(np.isfinite(c_t)):
                    cv2.arrowedLine(img, tuple(np.round(c_b).astype(int)), tuple(np.round(c_t).astype(int)), (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.2)
                    cv2.arrowedLine(img, tuple(np.round(c_b).astype(int)), tuple(np.round(c_t).astype(int)), col, 1, cv2.LINE_AA, tipLength=0.2)
                    _put(img, f"{sq} printed", c_b + [6, 14], col, 0.45)
                    _put(img, "tops of its pieces appear here", c_t + [6, -8], col, 0.45)
            uv_cc = self.project(q6, self.bp(f, r, z_out))[0][0]; uv_a = uv_ring[0]
            if np.all(np.isfinite(uv_a)) and np.all(np.isfinite(uv_cc)):
                lab = uv_a + (uv_a - uv_cc) / max(1e-6, np.linalg.norm(uv_a - uv_cc)) * 30
                _put(img, f"{sq} (believed)", lab + [-20, 0], col, 0.5)
        if np.all(np.isfinite(uv_plan)) and DRAW_LAND:
            if kind == "from":
                cv2.circle(img, tuple(np.round(uv_plan).astype(int)), 5, (0, 200, 0), -1, cv2.LINE_AA)
                _put(img, "planned", uv_plan + [7, 12], (0, 200, 0), 0.45)
            else:
                cv2.circle(img, tuple(np.round(uv_plan).astype(int)), 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, tuple(np.round(uv_plan).astype(int)), 3, (255, 0, 255), -1, cv2.LINE_AA)
        crop = img[y0:y0 + CROP, x0:x0 + CROP]
        crop = cv2.resize(crop, (int(CROP * SCALE), int(CROP * SCALE)), interpolation=cv2.INTER_CUBIC)
        meta = dict(x0=x0, y0=y0, scale=SCALE, land_px=((uv_land - [x0, y0]) * SCALE).tolist(),
                    plan_px=((uv_plan - [x0, y0]) * SCALE).tolist() if np.all(np.isfinite(uv_plan)) else None,
                    z_plane=z_plane, z_board=z_board, land_w=land_w.tolist(), tips_mid=tips_mid.tolist(),
                    tips_px=[((t - [x0, y0]) * SCALE).tolist() for t in uv_tips], tips_raw=[t.tolist() for t in uv_tips],
                    pads_px=((uv_pads - [x0, y0]) * SCALE).tolist() if np.all(np.isfinite(uv_pads)) else None,
                    q=np.asarray(q6, float).tolist(), piece=piece, square=sq,
                    centre_w=centre_w.tolist() if centre_w is not None else None, reg_t_mm=(1000 * self._reg()["t"]).tolist())
        return crop, meta

    # ------------------------------------------------------------------ the measurement
    def measure(self, ans, meta, kind="from"):
        """Claude's target pixel -> (dx, dy) mm that move LAND onto the feature, plus a dict for the log. A place uses
        the registered board model (from the grid intersections) when there is one; a grasp checks that the pointed
        top stands on the target square."""
        q = np.asarray(meta["q"], float)
        px = ans.get("target_px")
        if not px or len(px) != 2 or not all(np.isfinite(np.asarray(px, float))):
            raise ValueError("no target_px")
        raw = self.crop_to_raw(meta, px)
        feat_xy = self.backproject(q, raw, meta["z_plane"])
        land_xy = np.asarray(meta["land_w"][:2], float)
        info = {"target_px": [float(px[0]), float(px[1])], "target_raw": raw.tolist(), "feat_xy": feat_xy.tolist(), "method": "centre"}
        if meta.get("square") and meta.get("centre_w") is not None and kind == "to":
            reg = self._reg()
            if reg["n"] > 0 and reg["resid_mm"] is not None:           # a measured grid, not only a grasp anchor
                info["centre_px_xy"] = feat_xy.tolist()
                feat_xy = self.square_centre_w(meta["square"])[:2]
                info["method"] = f"registered ({reg['n']} intersections, resid {reg['resid_mm']:.1f} mm)"
                info["feat_xy"] = feat_xy.tolist()
        if kind == "from" and meta.get("square") and self.square_identity_trusted(ans):
            # the pointed top should stand on the (registered) target square
            dc = float(1000 * np.linalg.norm(feat_xy - self.square_centre_w(meta["square"])[:2]))
            info["top_from_square_mm"] = dc
            if dc > 15.0:
                info["wrong_piece"] = True
        off = ans.get("pads_offset_px") or [0, 0]
        try:
            off = np.asarray(off, float).ravel()[:2]
        except Exception:
            off = np.zeros(2)
        if np.all(np.isfinite(off)) and meta.get("pads_px") is not None and 0 < np.linalg.norm(off) <= 120:
            # the real gap between the pads sits off the model's: logged in mm on the pads' plane (applied once the
            # real-arm data shows it is a constant: HAND_EYE_MM below)
            z_tips = float(meta["tips_mid"][2]); pr = self.crop_to_raw(meta, meta["pads_px"])
            dl = self.backproject(q, pr + off / meta["scale"], z_tips) - self.backproject(q, pr, z_tips)
            info["pads_offset_mm"] = (1000 * dl).tolist()
            if self.low and kind == "from" and np.linalg.norm(dl) <= PADS_APPLY_MAX_MM / 1000.0:
                # at the bottom the real gap is what the piece must sit in: move LAND to where the pads really are
                land_xy = land_xy + dl; info["pads_applied"] = True
        if HAND_EYE_MM is not None:
            land_xy = land_xy + np.asarray(HAND_EYE_MM, float) / 1000.0
            info["hand_eye_mm"] = list(HAND_EYE_MM)
        k = int(getattr(self.staged, "retries", 0) or 0)
        if kind == "from" and 0 < k < len(TRY_OFFSET_MM) and TRY_OFFSET_MM[k] != 0.0:
            _, ta, tb = self.tips(q); jaw = (np.asarray(tb)[:2] - np.asarray(ta)[:2]); jaw = jaw / max(1e-9, np.linalg.norm(jaw))
            land_xy = land_xy + jaw * TRY_OFFSET_MM[k] / 1000.0
            info["try_offset_mm"] = [TRY_OFFSET_MM[k], (jaw * TRY_OFFSET_MM[k]).round(2).tolist()]
        d = (feat_xy - land_xy) * 1000.0
        info["d_mm"] = d.tolist()
        return float(d[0]), float(d[1]), info

    def truth_xy(self, kind):
        """Twin dry runs only: the true feature position (the FakeArm's pieces / its shifted board), else None."""
        try:
            robot = getattr(self.player, "robot", None)
            if robot is None or not hasattr(robot, "pieces"):
                return None
            from_xyz, to_xyz, piece = self._targets()
            if kind == "from":
                best = None
                for k, (t, p) in robot.pieces.items():
                    d = float(np.linalg.norm(np.asarray(p, float)[:2] - from_xyz[:2]))
                    if d < 0.03 and (best is None or d < best[0]):
                        best = (d, np.asarray(p, float)[:2])
                return None if best is None else best[1]
            sq = re.search(r"to (?:the )?([a-h][1-8])\b", self.staged.task) if self.staged is not None else None
            if sq is None or not hasattr(robot, "Pl"):
                return None
            return np.asarray(robot.Pl.s.square_xyz(sq.group(1)), float)[:2]
        except Exception:
            return None

    # ------------------------------------------------------------------ the look (Claude or the mock)
    def _situation_text(self, kind, meta):
        piece, colour = self.staged.piece, self.staged.colour
        w = int(CROP * SCALE)
        prev = ""
        if self.history:
            prev = " Earlier in this attempt: " + "; ".join(self.history[-3:]) + "."
        if self.staged.retries:
            prev += f" This is retry {self.staged.retries} of the stage (the previous attempt failed)."
        sq_from, sq_to = self.squares()
        if kind == "from":
            return (f"GRASP look {self.look_n + 1}. Target: the {colour} {piece} standing on square {sq_from}. The green outline is that square from the "
                    f"belief, drawn at the height of the piece TOPS ({TOP_Z_CM[piece]:.0f} cm up), so it appears about {TOP_Z_CM[piece] * 0.55:.0f} mm "
                    f"away from the square's printed edges and letters (the dashed outline is the printed square, the arrow leads to where the tops "
                    f"of its pieces appear). Judge pieces by their tops inside the solid outline, not by the printed letters. Pick the {piece} whose "
                    f"top is inside or closest to the solid outline; point at {TOP_NAME[piece]}. Image {w}x{w} px. The jaws are open {self.grip:.0f} %. "
                    f"Phase: {'the pads are down at the grasp height around the piece; last check before closing' if self.low else 'at the hover above the piece, before the descent'}." + prev)
        sq = sq_to or "the target square"
        base_mm = 1000 * max(0.0, float(meta["tips_mid"][2]) + self.site_above_tips(meta["q"]) - (self.staged.close_h or 0.0) - meta["z_board"])
        return (f"PLACE look {self.look_n + 1}. The arm holds the {colour} {piece}; target: the centre of square {sq} (outlined in magenta from the belief). "
                f"Report the printed grid intersections you see clearly in grid_px. Image {w}x{w} px. "
                f"The base is now about {base_mm:.0f} mm above the board. "
                f"Phase: {'lowered to the release height; last check' if self.low else 'at the place hover'}." + prev)

    def _ask(self, q6, img_rgb, kind):
        crop, meta = self.annotate(q6, img_rgb, kind)
        text = self._situation_text(kind, meta)
        self.last_look_img = crop
        if self.mock:
            return self._mock_answer(q6, meta, kind), meta, crop, text
        return self._model_answer(text, crop), meta, crop, text

    def _model_answer(self, text, crop):
        """Provider transport; Astra overrides this while sharing the exact control loop."""
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(timeout=40.0, max_retries=2)
        t0 = time.time()
        img_b64 = _b64_png(crop)
        resp = None
        for attempt, (budget, effort) in enumerate(((3000, EFFORT), (8000, "low"))):
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=budget, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": effort},
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                                                       {"type": "text", "text": text}]}])
            self.cost_calls += 1
            txt = "".join(b.text for b in resp.content if b.type == "text")
            if resp.stop_reason in ("end_turn", "stop_sequence") and txt.strip():
                break
            detail = f"stop_reason {resp.stop_reason}, model {getattr(resp, 'model', '?')}, blocks {[b.type for b in resp.content]}, out {resp.usage.output_tokens} tokens, text {txt[-120:]!r}"
            print(f"[{self.type} skill] look attempt {attempt + 1} unusable ({effort}, {budget}): {detail}", file=sys.stderr, flush=True)
            if attempt == 1:
                raise RuntimeError(detail)
        ans = extract_json(txt)
        ans["_latency_s"] = round(time.time() - t0, 2)
        ans["_usage"] = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens, "cached": getattr(resp.usage, "cache_read_input_tokens", 0), "effort": effort}
        return ans

    def _mock_answer(self, q6, meta, kind):
        """The twin's truth instead of Claude: the true feature projected into the crop (+ optional pixel noise)."""
        xy = self.truth_xy(kind)
        if xy is None:
            if kind == "to":
                xy = np.asarray(self._targets()[1], float)[:2]      # the drop zone / no fake arm: the belief
            else:
                return {"visible": False, "target_px": None, "tips_offset_px": [0, 0], "clear": True, "roll_deg": 0.0, "confidence": 0.0, "notes": "mock: no piece"}
        q = np.asarray(meta["q"], float)
        uv = self.project(q, [xy[0], xy[1], meta["z_plane"]])[0][0]
        px = (uv - [meta["x0"], meta["y0"]]) * meta["scale"]
        if MOCK_NOISE_PX > 0:
            px = px + np.random.normal(0, MOCK_NOISE_PX, 2)
        ans = {"visible": True, "target_px": [float(px[0]), float(px[1])], "tips_offset_px": [0, 0], "clear": True, "roll_deg": 0.0,
               "confidence": 0.95, "notes": "mock: the twin's truth", "_latency_s": 0.0}
        # the true grid intersections nearest to the target (the FakeArm's shifted board), three of them, as pixels
        robot = getattr(self.player, "robot", None)
        if robot is not None and hasattr(robot, "Pl"):
            sc = robot.Pl.s; pts = []
            sq = meta.get("square") or self.squares()[0]
            f, r = "abcdefgh".index(sq[0]), int(sq[1])
            o = np.asarray(sc.cfg.square_xy(3, 4), float)                       # the lattice is affine: a centre + two steps
            e_f = np.asarray(sc.cfg.square_xy(4, 4), float) - o; e_r = np.asarray(sc.cfg.square_xy(3, 5), float) - o
            for df, dr in ((-0.5, -0.5), (0.5, 0.5), (1.5, -0.5)):
                wxy = np.asarray(sc.board_to_world(o + (f + df - 3) * e_f + (r + dr - 4) * e_r), float)
                u = self.project(q, [wxy[0], wxy[1], meta["z_board"]])[0][0]
                gpx = (u - [meta["x0"], meta["y0"]]) * meta["scale"]
                if MOCK_NOISE_PX > 0:
                    gpx = gpx + np.random.normal(0, MOCK_NOISE_PX, 2)
                if 0 <= gpx[0] < CROP * SCALE and 0 <= gpx[1] < CROP * SCALE:
                    pts.append([float(gpx[0]), float(gpx[1])])
            ans["grid_px"] = pts
        return ans

    def _start_look(self, q6, img_rgb, kind):
        # A timeout/reset must not let the old worker write into a new attempt's pending look.
        pending = {"done": False}
        self.pending = pending
        generation = self._look_generation
        q = np.asarray(q6, float).copy(); img = np.asarray(img_rgb).copy()

        def work():
            try:
                ans, meta, crop, text = self._ask(q, img, kind)
                result = dict(ans=ans, meta=meta, crop=crop, text=text, q=q)
            except Exception as e:
                result = dict(error=repr(e))
                if getattr(e, "api_status", None) is not None:
                    result.update(api_status=e.api_status, api_retry_after=e.retry_after_s, api_fatal=e.fatal)
            if generation == self._look_generation and self.pending is pending:
                pending.update(result)
                pending["done"] = True
        threading.Thread(target=work, daemon=True).start()

    def _log_look(self, kind, record, crop):
        try:
            if self.type not in SESSION_DIRS:                            # one folder per player process (both skill instances)
                results = getattr(sys.modules.get("__main__"), "RESULTS", HERE / "results")
                SESSION_DIRS[self.type] = Path(results) / self.type / time.strftime("%Y%m%d_%H%M%S")
            d = SESSION_DIRS[self.type]
            d.mkdir(parents=True, exist_ok=True); self.log_dir = d
            n = len(list(d.glob("look_*.png")))
            cv2.imwrite(str(d / f"look_{n:02d}_{kind}.png"), crop)
            record = {"n": n, "kind": kind, "time": time.strftime("%H:%M:%S"), **record}
            with open(d / "looks.jsonl", "a") as f:
                f.write(json.dumps(record, default=float) + "\n")
            self._progress(kind)
        except Exception as e:
            print(f"[{self.type} skill] log failed: {e!r}", file=sys.stderr, flush=True)

    def _progress(self, kind):
        """One-line bar for data/progress.html (the user's progress rule): stage, look, elapsed and a rough ETA."""
        try:
            ev = getattr(self.staged, "events", None) or []
            elapsed = (time.monotonic() - ev[0][0]) if ev else 0.0
            frac = min(0.98, ((0 if kind == "from" else 1) + min(self.look_n, 3) / 3.0) / 2.0)
            eta = elapsed * (1 - frac) / max(frac, 0.05)
            n = int(round(30 * frac))
            task = getattr(self.staged, "task", "") or ""
            line = (f"[{'#' * n}{'.' * (30 - n)}] {100 * frac:4.0f}%  {task}: {'grasp' if kind == 'from' else 'place'} look {self.look_n}"
                    f"  elapsed {int(elapsed // 60)}:{int(elapsed % 60):02d}  ETA ~{int(eta // 60)}:{int(eta % 60):02d}  "
                    f"(retry {self.staged.retries}, {time.strftime('%H:%M:%S')}, looks logged in {self.log_dir})\n")
            (ROOT / "data" / f"progress_{self.type}.txt").write_text(line)
        except Exception:
            pass

    # ------------------------------------------------------------------ motion helpers
    def _settled(self, q6, now):
        err = np.abs(self.desired[:5] - np.asarray(q6)[:5]).max()
        if err <= SETTLE_TOL_DEG:
            if self.settle_t0 is None:
                self.settle_t0 = now
            return now - self.settle_t0 >= SETTLE_S or now - self.settle_start >= SETTLE_MAX_S
        self.settle_t0 = None
        return now - self.settle_start >= SETTLE_MAX_S

    def _begin_move(self, q_new, now, seconds):
        q_new = np.asarray(q_new, float).copy(); q_new[5] = self.grip             # the gripper is commanded separately
        self.move_from, self.move_to, self.move_t0, self.move_T = self.desired.copy(), q_new, now, seconds

    def _moving(self, now):
        if self.move_t0 is None:
            return False
        u = min(1.0, (now - self.move_t0) / max(1e-3, self.move_T))
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        self.desired = self.move_from + (self.move_to - self.move_from) * s
        if u >= 1.0:
            self.move_t0 = None; self.settle_start, self.settle_t0 = now, None
            return False
        return True

    def _command(self, q6):
        """Command = desired + the sag bias (integral of the error, the player's hold-stage logic), gripper separate."""
        self.desired[5] = self.grip                           # keep the commanded pose's jaw opening current: the tips'
        # midpoint (LAND) moves 0.35 mm per % toward the moving jaw, and the IK of a correction must use the same
        # opening as the camera measurement (10 -> 30 %: a 7 mm standing error, twin 2026-11-10)
        err = self.desired[:5] - np.asarray(q6, float)[:5]
        if self.move_t0 is None:                              # integrate only while not travelling
            self.bias[:5] = np.clip(self.bias[:5] + SAG_GAIN * err, -SAG_MAX_DEG, SAG_MAX_DEG)
        cmd = self.desired + self.bias; cmd[5] = self.grip
        return cmd.astype(np.float32)

    def _lateral(self, q_meas, dx, dy, now, roll_deg=0.0):
        """Move the fingertips by (dx, dy) mm from where they were measured at the look, tilt held. Returns the move
        really commanded (mm), shorter when the IK only reaches part of it, (0, 0) when it reaches none."""
        tips_now, _, _ = self.tips(q_meas)
        for f in (1.0, 0.5, 0.25):
            q_new = self.joints_for_tips(self.desired, tips_now + np.array([dx * f, dy * f, 0.0]) / 1000.0)
            if q_new is not None:
                break
        if q_new is None:
            print(f"[{self.type} skill] IK cannot move the tips by ({dx:+.1f}, {dy:+.1f}) mm here: no move", file=sys.stderr, flush=True)
            return 0.0, 0.0
        if abs(roll_deg) > 3:
            q_new[4] += float(np.clip(roll_deg, -45, 45))
        self._begin_move(q_new, now, MOVE_S)
        return dx * f, dy * f

    def site_above_tips(self, q6):
        """The planner's grasp site (play/staged.Planner.fk, the reference of the humans' close heights and of the
        player's close_h) sits above the fingertip sites: 6.0 mm on the twin. Heights are converted through it."""
        try:
            site, _ = self.Pl.fk(np.asarray(q6, float)); tips, _, _ = self.tips(q6)
            return float(site[2] - tips[2])
        except Exception:
            return 0.006

    def square_identity_trusted(self, ans):
        return self._reg()["looks"] > 0

    def grasp_height_cm(self, piece):
        return CLOSE_H_CM[piece]

    def _down(self, kind, now):
        """Grasp: the grasp SITE to the humans' close height (the pads then sit where the humans' pads sat; with the
        fingertips there instead they were 6 mm too high and closed on the queen's crown, real arm 2026-11-10), jaws
        opened. Place: the held piece's base to 3 mm above the board (close_h is a site height too)."""
        tips_mid, _, _ = self.tips(self.desired)
        off = self.site_above_tips(self.desired)
        if kind == "from":
            piece = self._targets()[2]
            z_goal = self.z_board + self.grasp_height_cm(piece) / 100.0 - off
            self.grip = OPEN_GRIP
            verified = getattr(self, "_uses_verified_grasp", None)     # bishop / queen: Codex's verified straight descent
            if MID_LOOK and not self.mid_done and (MID_LOOK_VERIFIED or not (verified and verified())):
                self.mid_done = True
                z_mid = float(self._targets()[0][2]) + TOP_Z_CM[piece] / 100.0 + MID_CLEAR_M
                if z_mid > z_goal + 0.008 and tips_mid[2] - z_mid > 0.008:
                    dz = z_mid - tips_mid[2]
                    q_new = self.joints_for_tips(self.desired, tips_mid + np.array([0.0, 0.0, dz]), tol=4e-3)
                    if q_new is not None:
                        self.last_dist = None                     # the drift seen from here is expected: no jump guard
                        self._begin_move(q_new, now, max(0.6, abs(dz) * 1000.0 / DESCEND_MM_S))
                        print(f"[{self.type} skill] mid-descent look: {-1000 * dz:.0f} mm down, pads {1000 * MID_CLEAR_M:.0f} mm above the {piece}'s top", file=sys.stderr, flush=True)
                        return False
        else:
            base_now = tips_mid[2] + off - (self.staged.close_h or 0.02)
            z_goal = tips_mid[2] - (base_now - (self.z_board + 0.003))
        dz = z_goal - tips_mid[2]
        q_new = self.joints_for_tips(self.desired, tips_mid + np.array([0.0, 0.0, dz]), tol=4e-3)
        if q_new is None:                                             # best effort: the vertical move must happen
            q_new = self.joints_for_tips(self.desired, tips_mid + np.array([0.0, 0.0, dz]), tol=1.0)
        self.low = True
        self._begin_move(q_new, now, max(0.6, abs(dz) * 1000.0 / DESCEND_MM_S))
        return True

    def _up_to_mid(self, now):
        """From the bottom back to the mid-descent height (pads MID_CLEAR_M above the top) for another look."""
        tips_mid, _, _ = self.tips(self.desired)
        piece = self._targets()[2]
        z_mid = float(self._targets()[0][2]) + TOP_Z_CM[piece] / 100.0 + MID_CLEAR_M
        dz = z_mid - tips_mid[2]
        if dz < 0.005:
            return False
        q_new = self.joints_for_tips(self.desired, tips_mid + np.array([0.0, 0.0, dz]), tol=4e-3)
        if q_new is None:
            return False
        self.low = False; self.mid_done = True; self.last_dist = None
        getattr(self, "_forget_tracker", lambda: None)()
        self._begin_move(q_new, now, max(0.6, abs(dz) * 1000.0 / DESCEND_MM_S))
        print(f"[{self.type} skill] bottom look saw no target: back up {1000 * dz:.0f} mm to the mid height and look again", file=sys.stderr, flush=True)
        return True

    def grasp_capture(self, ans, meta, dx, dy):
        """Optional provider-specific body-height capture window; the normal 2 mm aim stays unchanged."""
        return None

    def visual_grasp_ready(self, ans, meta, q6, kind):
        return False

    def handle_look_error(self, pending, now, kind):
        return False

    def review_action(self, action, why, ans, meta, kind, dist):
        """Provider-specific bounded verification before committing a decision."""
        return action, why

    # ------------------------------------------------------------------ the tick
    def act(self, state6, image_hwc_uint8, task):
        now = time.monotonic()
        q6 = np.asarray(state6, float)
        kind = "from" if "from" in (self.goal or {}).get("which", ("from",)) else "to"
        if self.phase == "init":
            self.desired = q6.copy(); self.bias = np.zeros(6)
            self.grip = HOVER_GRIP if kind == "from" else CLOSE_GRIP            # place: holding the piece
            self.settle_start, self.settle_t0 = now, None
            self.no_look = kind == "to" and self.squares()[1] == "drop zone"    # no printed square there: the belief is the target
            self.phase = "settle_look"
        if self._moving(now):
            return self._command(q6)
        if self.phase == "settle_look" and kind == "from" and not self.raised and not self.low:
            self.raised = True
            _, _, piece = self._targets()
            z_top = float(self._targets()[0][2]) + TOP_Z_CM[piece] / 100.0
            tips_mid, _, _ = self.tips(self.desired)
            dz = (z_top + LOOK_CLEAR_M) - tips_mid[2]
            if dz > 0.003:
                # near the base (the back rank) the full raise can have no IK solution: settle for half of it rather
                # than looking with the pads filling the frame (bishop f8, 2026-11-10: 'target None' three times)
                for frac in (1.0, 0.5):
                    q_new = self.joints_for_tips(self.desired, tips_mid + np.array([0.0, 0.0, dz * frac]), tol=4e-3)
                    if q_new is not None:
                        self._begin_move(q_new, now, max(0.6, dz * frac * 1000.0 / DESCEND_MM_S))
                        print(f"[{self.type} skill] looking from {1000 * dz * frac:.0f} mm higher (pads {1000 * (LOOK_CLEAR_M - dz * (1 - frac)):.0f} mm above the {piece}'s top)", file=sys.stderr, flush=True)
                        return self._command(q6)
                print(f"[{self.type} skill] no reachable raised look for the {piece}: looking from the hover", file=sys.stderr, flush=True)
        if self.phase in ("settle_look", "settle_look_low"):
            if self._settled(q6, now):
                if getattr(self, "no_look", False):
                    if not self.low:
                        self._down(kind, now); self.phase = "settle_look_low"
                    else:
                        self.grip = RELEASE_GRIP; self.phase = "released"; self.t_phase = now
                    return self._command(q6)
                if self.look_n >= self.max_looks:
                    if self.low and not self.strict_measurements:                 # legacy skill behaviour
                        self.grip = CLOSE_GRIP if kind == "from" else RELEASE_GRIP
                        self.phase = "closed" if kind == "from" else "released"; self.t_phase = now
                    else:
                        self.phase = "give_up"; self.gave_up = f"out of looks ({self.max_looks}) without a verified final look"
                else:
                    self._start_look(q6, image_hwc_uint8, kind); self.phase = "look"
            return self._command(q6)
        if self.phase == "look":
            if not self.pending["done"]:
                return self._command(q6)
            self.look_n += 1; self.looks_total += 1
            if "error" in self.pending:
                self.err = self.pending["error"]; self.history.append(f"look {self.look_n} failed ({self.err[:60]})")
                print(f"[{self.type} skill] look failed: {self.err[:300]}", file=sys.stderr, flush=True)
                self._log_look(kind, {"phase": "low" if self.low else "hover", "error": self.err},
                               self.last_look_img if self.last_look_img is not None else np.zeros((8, 8, 3), np.uint8))
                if self.handle_look_error(self.pending, now, kind):
                    return self._command(q6)
                self.phase = "settle_look_low" if self.low else "settle_look"; self.settle_start, self.settle_t0 = now, None
                return self._command(q6)
            ans, meta, q_meas = self.pending["ans"], self.pending["meta"], self.pending["q"]
            self.answer = ans
            rec = {"phase": "low" if self.low else "hover", "text": self.pending["text"], "answer": ans, "q": meta["q"],
                   "land_px": meta["land_px"], "land_w": meta["land_w"], "z_plane": meta["z_plane"], "tips_px": meta["tips_px"]}
            if self.visual_grasp_ready(ans, meta, q6, kind):
                self.desired = q6.copy(); self.bias = np.zeros(6)
                self.grip, self.phase, self.t_phase = CLOSE_GRIP, "closed", now
                rec.update(action="close", method="fresh visual body between contact pads", used_mm=[0., 0.])
                self._log_look(kind, rec, self.pending["crop"])
                print(f"[{self.type} skill] close: target body visibly between contact pads", file=sys.stderr, flush=True)
                self.anchor(q6, meta)
                return self._command(q6)
            truth = self.truth_xy(kind)
            if truth is not None:
                rec["truth_mm"] = ((truth - np.asarray(meta["land_w"][:2])) * 1000.0).tolist()
                try:
                    robot = getattr(self.player, "robot", None)
                    if meta.get("square") and hasattr(robot, "Pl"):
                        true_c = np.asarray(robot.Pl.s.square_xyz(meta["square"]), float)[:2]
                        rec["reg_err_before_mm"] = (1000 * (true_c - self.square_centre_w(meta["square"])[:2])).tolist()
                except Exception:
                    pass
            visible = bool(ans.get("visible", True)) and ans.get("target_px") is not None
            dx = dy = 0.0; dist = None; why = ""
            if visible:
                try:
                    rec["reg"] = self.register(ans, meta)
                    if "reg_err_before_mm" in rec:
                        true_c = np.asarray(self.player.robot.Pl.s.square_xyz(meta["square"]), float)[:2]
                        rec["reg_err_after_mm"] = (1000 * (true_c - self.square_centre_w(meta["square"])[:2])).tolist()
                    dx, dy, info = self.measure(ans, meta, kind)
                    rec.update(info); dist = float(np.hypot(dx, dy))
                    if kind == "from" and np.isfinite([dx, dy]).all():
                        self.measured_from_xy = (np.asarray(meta["land_w"][:2], float) + np.array([dx, dy]) / 1000.0).tolist()
                    if info.get("wrong_piece"):
                        # the pointed piece is not on the target square: look again with the registered outline
                        u = self.project(np.asarray(meta["q"], float), self.square_centre_w(meta["square"], meta["z_plane"] - meta["z_board"]))[0][0]
                        upx = (u - [meta["x0"], meta["y0"]]) * meta["scale"]
                        self.history.append(f"look {self.look_n}: the piece you pointed at ({info['top_from_square_mm']:.0f} mm from the square) is not the target; "
                                            f"the target square's centre is at pixel ({upx[0]:.0f}, {upx[1]:.0f}) in that image")
                        visible = None
                except Exception as e:
                    rec["measure_error"] = repr(e); visible = False
            if visible is None:                                      # wrong piece: just look again (the outline moved)
                rec.update(action="relook", used_mm=[0.0, 0.0]); self._log_look(kind, rec, self.pending["crop"])
                print(f"[{self.type} skill] {kind} look {self.look_n}: top {rec.get('top_from_square_mm', 0):.0f} mm from the registered square -> look again", file=sys.stderr, flush=True)
                self.phase = "settle_look_low" if self.low else "settle_look"; self.settle_start, self.settle_t0 = now, None
                return self._command(q6)
            if not visible and kind == "to" and meta.get("plan_px") is not None and not self.strict_measurements:
                # the square cannot be found: put the piece down at the believed centre (within ~10 mm of the real one)
                land_xy = np.asarray(meta["land_w"][:2]); plan_xy = np.asarray(self._targets()[1][:2])
                dx, dy = (float(v) for v in (plan_xy - land_xy) * 1000.0); dist = float(np.hypot(dx, dy)); visible = True
                rec["fallback"] = "believed centre"; self.history.append(f"look {self.look_n}: square not found, aiming at the ring")
            conf = float(ans.get("confidence", 0) or 0)
            # A look under the 0.7 confidence floor is not thrown away: it is re-taken, and two consecutive looks that agree
            # within 3 mm are a measurement whatever their confidence (f8 bishop, 2026-11-10: found 3 times, converged to
            # 1.4 mm, then the attempt died on a 0.64).
            agree = bool(visible and self.lowconf_prev is not None and dist is not None
                         and np.hypot(dx - self.lowconf_prev[0], dy - self.lowconf_prev[1]) <= 3.0)
            conf_ok = conf >= 0.7 or agree
            if agree:
                rec["lowconf_agreement"] = True; ans["_agreement"] = True
            self.lowconf_prev = (dx, dy) if (visible and conf < 0.7) else None
            clear = bool(ans.get("clear", True))
            tol = TOL_MM[f"{kind}_{'low' if self.low else 'hover'}"]
            capture = self.grasp_capture(ans, meta, dx, dy) if self.strict_measurements and visible and self.low and kind == "from" else None
            if capture is not None:
                rec["capture_window"] = capture
            capture_ok = bool(capture and capture.get("ok"))
            if visible and kind == "from" and self.last_dist is not None and self.last_dist <= 3.0 and dist > 8.0 and self.jumps < 2:
                # the previous look had the target within 3 mm; a sudden 8+ mm jump is another piece (or the sponge
                # hiding the target): do not move, look again
                self.jumps += 1
                rec.update(action="relook", used_mm=[0.0, 0.0], jump_from_mm=self.last_dist); self._log_look(kind, rec, self.pending["crop"])
                self.history.append(f"look {self.look_n}: your point was {dist:.0f} mm from the previous look's, which had the target within {self.last_dist:.0f} mm of LAND: ignored, look again")
                print(f"[{self.type} skill] {kind} look {self.look_n}: jump {dist:.1f} mm after a converged look -> look again", file=sys.stderr, flush=True)
                self.phase = "settle_look_low" if self.low else "settle_look"; self.settle_start, self.settle_t0 = now, None
                return self._command(q6)
            if visible:
                self.last_dist = dist
            # the decision is geometric; Claude's flags only veto
            blind_close = (CLOSE_ON_HIDDEN_TARGET and not visible and self.low and kind == "from" and self.descended_from_mm is not None
                           and self.descended_from_mm <= MAX_DESCEND_MM)
            if (not visible and self.low and kind == "from" and self.low_relooks < 1 and self.look_n < self.max_looks
                    and self._up_to_mid(now)):
                # Every blind close tonight shut on nothing: a hidden target at the bottom means the piece is NOT in the gap.
                # Back up to the mid height, look again with the hint ring, correct, descend again.
                self.low_relooks += 1; self.hint_ring = True; blind_close = False
                action = "relook"; rec["relook"] = "bottom: target hidden, back up to the mid height"
                self.history.append(f"look {self.look_n}: nothing in view at the bottom; backing up to look again")
            elif blind_close:
                action = "close"; rec["blind_close"] = {"hover_mm": self.descended_from_mm}
                self.history.append(f"look {self.look_n}: no target in view at the bottom; the hover had it within {self.descended_from_mm:.1f} mm, so the piece is between the pads: closing")
            elif (not visible and not self.low and kind == "from" and self.last_dist is not None and self.last_dist > 5.0
                  and self.lost_relooks < 1 and self.look_n < self.max_looks):
                # the piece was in view a moment ago and vanished after the centring move: a tall piece's top now sits under
                # the near pad (f8 bishop, 2026-11-10, looked at from only 25 mm). Look again from higher up before giving up.
                self.lost_relooks += 1; self.raised = False; self.hint_ring = True; action = "relook"; rec["relook"] = "lost after the move: re-raise"
                self.history.append(f"look {self.look_n}: the target vanished after the move; looking again from higher up")
            elif not visible:
                action = "give_up"; why = "target not in view" if ans.get("target_px") is None or not ans.get("visible", True) else "no measurement"
            elif self.strict_measurements and not conf_ok:
                if self.lowconf_relooks < 2 and self.look_n < self.max_looks:
                    self.lowconf_relooks += 1; getattr(self, "_forget_tracker", lambda: None)()
                    action = "relook"; rec["relook"] = f"low confidence {conf:.2f}: look again"
                else:
                    action = "give_up"; why = "uncertain target measurement"
            elif self.low:
                if capture_ok:
                    action = "close"
                elif dist > tol and self.moves_low < MAX_LOW_MOVES:
                    action = "move"
                else:
                    action = "close" if kind == "from" else "release"
            else:
                if dist > tol and self.moves_hover < MAX_HOVER_MOVES:
                    action = "move"
                elif dist > MAX_DESCEND_MM:
                    action = "give_up"                                           # cannot converge: a bottom look will not fix it
                    why = f"still {dist:.0f} mm off after {self.moves_hover} moves"
                else:
                    action = "descend" if kind == "from" else "lower"
            if self.strict_measurements:
                if action in ("descend", "lower", "close", "release") and not blind_close and (not clear or (dist > tol and not (action == "close" and capture_ok))):
                    action = "give_up"; why = "blocked or not converged: no blind descent/close"
            action, why = self.review_action(action, why, ans, meta, kind, dist)
            if action == "relook":
                dx = dy = 0.0
            if action == "move":
                step = min(1.0, MAX_STEP_MM / max(dist, 1e-6)); dx, dy = dx * step, dy * step
            rec.update(action=action, used_mm=[dx, dy], truth_after_mm=None)
            if truth is not None and action == "move":
                rec["truth_after_mm"] = ((truth - np.asarray(meta["land_w"][:2])) * 1000.0 - [dx, dy]).tolist()
            self._log_look(kind, rec, self.pending["crop"])
            tr = f", truth ({rec['truth_mm'][0]:+.1f}, {rec['truth_mm'][1]:+.1f})" if "truth_mm" in rec else ""
            rg = rec.get("reg", {}); ra = rec.get("reg_err_after_mm")
            rtxt = f", grid {rg.get('grid_used', 0)}/{rg.get('grid_n', 0)}" + (f" reg err ({ra[0]:+.1f}, {ra[1]:+.1f})" if ra else "")
            print(f"[{self.type} skill] {kind} look {self.look_n} ({rec['phase']}): target {ans.get('target_px')} -> d ({dx:+.1f}, {dy:+.1f}) mm{tr}{rtxt}, "
                  f"{rec.get('method', '')}, conf {conf:.2f}, {ans.get('_latency_s', 0)} s -> {action}", file=sys.stderr, flush=True)
            self.history.append(f"look {self.look_n}: {action}" + (f" by {dist:.1f} mm" if action == "move" else ""))
            if action == "relook":
                self.phase = "settle_look_low" if self.low else "settle_look"
                self.settle_start, self.settle_t0 = now, None
            elif action == "give_up":
                self.phase = "give_up"; self.gave_up = why or "gave up"
            elif action == "move":
                roll = float(ans.get("roll_deg", 0.0) or 0.0) if (kind == "from" and self._targets()[2] == "knight" and not self.low) else 0.0
                mdx, mdy = self._lateral(q_meas, dx, dy, now, roll)
                if (mdx, mdy) != (dx, dy):
                    rec["used_mm"] = [mdx, mdy]; self.history.append(f"(the arm could only move {np.hypot(mdx, mdy):.1f} mm of it)")
                if self.low:
                    self.moves_low += 1; self.phase = "settle_look_low"
                else:
                    self.moves_hover += 1; self.phase = "settle_look"
            elif action in ("descend", "lower"):
                self.descended_from_mm = dist
                low = self._down(kind, now)
                self.phase = "settle_look" if low is False else "settle_look_low"     # False: the mid-descent look first
            elif action == "close":
                self.grip = CLOSE_GRIP; self.phase = "closed"; self.t_phase = now
                if not rec.get("blind_close"):                   # a blind close may be a miss: never anchor the model on it
                    self.anchor(q6, meta)
            else:
                self.grip = RELEASE_GRIP; self.phase = "released"; self.t_phase = now
            return self._command(q6)
        if self.phase in ("closed", "released"):
            return self._command(q6)                          # the player's hold / release rules take it from here
        return self._command(q6)                              # give_up: the player's timeout retries or fails the stage
