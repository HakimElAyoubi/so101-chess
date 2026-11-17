"""Staged move execution, plan step 2 (train/PLAN_TO_95.md): geometry takes the gripper to the square, the learned skill
policy (trained on LeHak0/so101_chess_skills) does the grasp and the place.

  ready --APPROACH (joint min-jerk to above the piece, then straight down)--> grasp hover
        --GRASP (skill: "pick up the black bishop", green dot)--> holding detected
        --CARRY (IK: straight up, across at travel height, straight down)--> place hover
        --PLACE (skill: "put the black bishop down on the ring", magenta ring)--> release detected
        --LIFT (straight up) + RETURN (joint min-jerk)--> ready
A grasp that closes on nothing, or takes longer than GRASP_TIMEOUT_S, goes back up to the hover pose and tries again
(at most MAX_RETRIES times). Losing the piece during the carry, or a place timeout, ends the move as a failure.

Template measured on the 395 skill clips (twin FK on the human demos, 2026-10-26):
- grasp hover: 2.5 s before closing the human's gripper is ~3.5 cm above the piece's top (king 8.7 cm, pawn 5.9 cm above the
  board), within ~0.5 cm of straight above it, tilted -2 deg near the arm (< 14 cm from the pan axis) and ~12 deg further
  out, wrist roll ~20 deg from radial; the gripper is still CLOSED there (median 1.5 %): it opens on the way down.
- place hover: ~1.5 cm above the height the piece was grasped at; tilt grows with reach (1.08 deg/cm * r - 19 deg).
- gripper while holding: the follower reads 3.95 % median (p5 2.7 %) while the command is ~0.8 %; closed on nothing it
  reads <= 1.47 %.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "goal")]
from chess_sim import config as C                                           # noqa: E402
from chess_sim.kinematics import Kinematics, frame_params, radial_lat, rotate_lat   # noqa: E402

# ---- hover template (see the docstring)
R_CM = [12.0, 15.5, 18.5, 21.5, 24.5, 28.0]          # distance of the target from the shoulder-pan axis (cm), bin centres
# The grasp hover is where the humans' hand was 2.0 s before they closed the jaws (177 demos), i.e. half a second into
# every grasp clip the skill trained on: 1-2 cm above the piece's top with the jaws already opening, instead of the clip
# start (-2.5 s: 3 cm up, jaws closed). Lower would put the open jaws around the piece's head (-1.5 s), where a few mm
# of hover error hits it.
GRASP_TILT = [-6.9, 4.9, 4.1, 7.6, 7.3, 14.3]        # human median jaw tilt 2.0 s before the close, per bin (deg)
PSI_DEG = 20.0                                       # wrist roll of the jaw opening axis from radial (deg)
# grasp hover: gripper site this far above the piece's top (m): the human medians at -2.0 s, knight and king (3-4 mm)
# held at MIN_ABOVE_TOP
ABOVE_TOP = {"pawn": 0.020, "rook": 0.019, "knight": 0.010, "bishop": 0.018, "queen": 0.012, "king": 0.010}
HOVER_GRIP = 10.0                                    # ...with the jaws this open (%; humans 9-13 % at -2.0 s)
MIN_GRASP_H = 0.030
PLACE_ABOVE_GRASP = 0.015                            # place hover: this far above the height the piece was grasped at
MIN_PLACE_H = 0.025
TRAVEL_CLEAR = 0.060                                 # carry: the held piece's base this far above the board (king 50 mm + 1 cm)
APPROACH_Z = 0.110                                   # from the ready pose, first go above the piece at this height (if reachable)
LIFT_AFTER_RELEASE = 0.040
MIN_ABOVE_TOP = 0.010                                # where the template is out of reach: no lower than this above the top
HOVER_MAX_TILT = 35.0                                # humans tilted up to ~33 deg over the rank-8 kings
WRIST_SOFT_MAX, WRIST_SOFT_PENALTY = 95.0, 6.0       # prefer hovers with the wrist flex below 95 deg (cost units ~ 30 deg of tilt)
TILT_TRIES = [0, -5, 5, -10, 10, -15, 15, -20, 20, -25, 25]   # IK fallbacks around the template tilt (deg), |tilt| <= 30
MAX_TILT = 35.0
# ---- motion
PEAK_DEG_S = 30.0                                    # min-jerk peak joint speed, as the recorder's auto-moves
SETTLE_S = 0.3
# path clock: the time along a trajectory only runs as fast as the arm follows it. Raising the arm against gravity the
# shoulder lift trails its target by 5-7 deg at full speed (P=16), the player clips every command to 4 deg from the
# measured joints, and the elbow kept to its schedule: the jaws swung down and skimmed the board (real arm, 2026-10-27,
# planned 7-9 cm up, measured 1-3 mm). Full speed while every joint is within LAG_OK of the path, MIN_RATE beyond LAG_SLOW.
LAG_OK, LAG_SLOW, MIN_RATE = 3.0, 6.0, 0.05
SKILL_STAGES = ("grasp", "place")                   # the model drives; every other stage follows a planned path
# gravity sag: at P=16 the follower hangs below its command when stretched out (real arm, 2026-10-27: elbow ~2.8 deg low,
# the jaws ~9 mm low and 4-7 mm short at 28-30 cm from the pan axis, ~0.3 deg near it). Before a skill takes over, the
# command is pushed (integral of the joint error, per tick) until the measured joints reach the hover.
# Done within ~1 encoder tick: gravity leaves the last of the error on the low side, and 0.3 deg on the lift and the
# elbow is ~2.5 mm low at a1 / h1 (the user saw the jaws ~3 mm lower there than planned).
SAG_GAIN, SAG_TOL_DEG, SAG_MAX_DEG, SAG_MAX_S = 0.15, 0.12, 5.0, 2.0
# ---- gripper signal (follower % vs command %)
HOLD_MIN, EMPTY_MAX, CLOSED_CMD, OPENED_CMD = 2.0, 1.7, 3.0, 6.0
HOLD_GAP, STALL_TOL = 1.0, 0.6    # holding = the follower stays >= 1 % above the command and has stopped moving (stalled on the piece)
HOLD_CONFIRM_S, MISS_CONFIRM_S, RELEASE_CONFIRM_S, DROP_CONFIRM_S = 0.5, 0.5, 0.3, 0.3
GRASP_TIMEOUT_S, PLACE_TIMEOUT_S, MAX_RETRIES = 12.0, 12.0, 2
# ---- keep going (the guided game, 2026-11-10): the drop detector (follower <= EMPTY_MAX during the carry) misfires on a
# thin pawn stem in the sponge pads (read 1.2 % while visibly held), so it is advisory: noted in the events, never a
# failure. A place that gives up or times out opens the jaws where they are (BLIND_RELEASE_PCT) instead of ending the
# move with the piece held. KEEP_GOING = False restores the old behaviour (fail on both).
KEEP_GOING, BLIND_RELEASE_PCT = True, 14.0

MOVE = re.compile(r"(black|white) (pawn|knight|bishop|rook|queen|king) (?:from|on) ([a-h][1-8]) to (?:the )?([a-h][1-8]|drop zone)")


def minjerk(u):
    u = min(1.0, max(0.0, u))
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class Trajectory:
    """Joint waypoints [n, 6] (LeRobot deg / %) played with one min-jerk time scaling over their joint-space length."""

    def __init__(self, waypoints, peak=PEAK_DEG_S):
        self.W = np.asarray(waypoints, float)
        d = np.abs(np.diff(self.W[:, :5], axis=0)).max(1) if len(self.W) > 1 else np.zeros(0)
        self.L = np.concatenate([[0.0], np.cumsum(d)])
        self.T = max(0.8, 1.875 * self.L[-1] / peak)

    def at(self, t):
        s = minjerk(t / self.T) * self.L[-1]
        if self.L[-1] <= 0:
            return self.W[-1].copy()
        return np.array([np.interp(s, self.L, self.W[:, j]) for j in range(6)])

    def done(self, t):
        return t >= self.T + SETTLE_S

    def advance(self, tau, q6, dt):
        """Path clock: tau after dt, slowed while the arm trails the path (see LAG_OK)."""
        err = float(np.abs(self.at(tau)[:5] - np.asarray(q6, float)[:5]).max())
        return tau + dt * float(np.clip((LAG_SLOW - err) / (LAG_SLOW - LAG_OK), MIN_RATE, 1.0))


class Planner:
    """IK on the calibrated twin (joint offsets + hand-eye), board pose from goal/board_poses.json via the Projector."""

    def __init__(self, proj):
        self.P, self.s = proj, proj.s
        self.K = Kinematics(proj.s.model)
        self.z0 = float(proj.s.cfg.board_thickness)
        self.pan = self.K.pan_axis_xy

    def fk(self, q6):
        p, R, _ = self.K.fk(C.lerobot_to_qpos(np.asarray(q6, float))[:5])
        return p, R

    def orientation(self, q6):
        p, R = self.fk(q6)
        return frame_params(R, radial_lat(p, 1.0, self.pan))

    def r_cm(self, xyz):
        return 100 * float(np.linalg.norm(np.asarray(xyz, float)[:2] - self.pan))

    def ik(self, xyz, tilt, psi, seed6, gripper):
        """Gripper site at xyz with jaw tilt / roll (deg); returns [5 joints deg, gripper %] or None."""
        lat = rotate_lat(radial_lat(xyz, 1.0, self.pan), psi)
        seed = C.lerobot_to_qpos(seed6)[:5] if seed6 is not None else None
        r = self.K.solve(np.asarray(xyz, float), lat, tool_site=np.zeros(3), tilt_deg=tilt, q_init=seed)
        if not r.ok:
            return None
        q = C.qpos_to_lerobot(np.append(r.q5, 0.0))       # back to REAL-arm degrees (joint zero offsets)
        q[5] = gripper
        return q

    def ik_near(self, xyz, tilt, psi, seed6, gripper):
        """ik() with the template tilt first, then the TILT_TRIES fallbacks. Returns (q6, tilt used) or (None, None)."""
        for dt in TILT_TRIES:
            t = tilt + dt
            if abs(t) > MAX_TILT:
                continue
            q = self.ik(xyz, t, psi, seed6, gripper)
            if q is not None:
                return q, t
        return None, None

    # ---- hover poses
    def grasp_hover(self, target_xyz, piece, seed6, gripper):
        """Above the piece standing at target_xyz (board-top z). Returns (q6, info) or (None, info).
        The template pose first; where it is out of reach (tall pieces on rank 8, next to the arm) the nearest pose
        that is reachable, as the human did there: tilted up to HOVER_MAX_TILT, or lower (down to 1 cm above the piece's
        top), preferring a wrist flex <= WRIST_SOFT_MAX (the leader is awkward near its limit)."""
        top = C.PIECE_SPECS[piece][0]
        h0 = max(MIN_GRASP_H, top + ABOVE_TOP[piece])
        tilt0 = float(np.interp(self.r_cm(target_xyz), R_CM, GRASP_TILT))
        cands = []
        for h in np.arange(h0, top + MIN_ABOVE_TOP - 1e-9, -0.005):
            for dt in range(-40, 41, 5):
                t = tilt0 + dt
                if abs(t) <= HOVER_MAX_TILT:
                    cands.append((abs(dt) / 5 + (h0 - h) / 0.005, h, t))
        best = None
        for cost, h, t in sorted(cands):
            if best is not None and cost >= best[0]:
                break
            q = self.ik([target_xyz[0], target_xyz[1], target_xyz[2] + h], t, PSI_DEG, seed6, gripper)
            if q is None:
                continue
            c = cost + (WRIST_SOFT_PENALTY if q[3] > WRIST_SOFT_MAX else 0)
            if best is None or c < best[0]:
                best = (c, q, h, t)
        if best is None:
            return None, dict(h=h0, tilt=None, tilt_template=tilt0)
        return best[1], dict(h=best[2], tilt=best[3], tilt_template=tilt0, h_template=h0)

    def place_hover(self, target_xyz, close_h, seed6, gripper):
        h = max(MIN_PLACE_H, close_h + PLACE_ABOVE_GRASP)
        tilt = float(np.clip(1.08 * self.r_cm(target_xyz) - 19.0, -MAX_TILT, MAX_TILT))
        xyz = np.array([target_xyz[0], target_xyz[1], target_xyz[2] + h])
        q, t = self.ik_near(xyz, tilt, PSI_DEG, seed6, gripper)
        return q, dict(h=h, tilt=t, tilt_template=tilt)

    # ---- paths
    def cart_path(self, q_start, q_end, z_travel, step=0.01):
        """Straight up from q_start to z_travel (or as high as the arm reaches there), across, straight down onto q_end.
        Orientation and gripper are interpolated from start to end. Returns (waypoints [n, 6], info)."""
        p0, _ = self.fk(q_start); p1, _ = self.fk(q_end)
        t0, s0 = self.orientation(q_start); t1, s1 = self.orientation(q_end)
        ds = (s1 - s0 + 180) % 360 - 180
        za, zb = max(p0[2], z_travel), max(p1[2], z_travel)
        corners = [p0, np.array([p0[0], p0[1], za]), np.array([p1[0], p1[1], zb]), p1]
        seg = [np.linalg.norm(b - a) for a, b in zip(corners, corners[1:])]
        total = sum(seg) or 1.0
        pts, fr = [], []
        acc = 0.0
        for (a, b), L in zip(zip(corners, corners[1:]), seg):
            n = max(1, int(math.ceil(L / step)))
            for i in range(1, n + 1):
                pts.append(a + (b - a) * i / n); fr.append((acc + L * i / n) / total)
            acc += L
        W, prev, lowered, worst = [np.asarray(q_start, float)], np.asarray(q_start, float), 0, 0.0
        for p, f in zip(pts[:-1], fr[:-1]):
            g = q_start[5] + (q_end[5] - q_start[5]) * f
            tilt, psi = t0 + (t1 - t0) * f, s0 + ds * f
            q = None
            floor = max(min(p0[2], p1[2]), p[2] - 0.08)
            z = p[2]
            while q is None and z >= floor - 1e-9:          # reach ceiling: come down until the IK converges
                q, _ = self.ik_near([p[0], p[1], z], tilt, psi, prev, g)
                if q is None:
                    z -= 0.005
            if q is None:
                continue                                    # unreachable sample: skip it (the neighbours carry the path)
            if z < p[2] - 1e-9:
                lowered += 1; worst = max(worst, p[2] - z)
            W.append(q); prev = q
        W.append(np.asarray(q_end, float))
        return np.array(W), dict(z_travel=z_travel - self.z0, lowered=lowered, max_lowered_m=worst, n=len(W))

    def approach(self, ready, hover):
        """Ready -> above the target at APPROACH_Z (joint min-jerk; the ready pose is high and leans back) -> straight down."""
        p, _ = self.fk(hover); t, s = self.orientation(hover)
        pre = None
        z = max(p[2], self.z0 + APPROACH_Z)
        while pre is None and z > p[2] + 1e-9:
            pre, _ = self.ik_near([p[0], p[1], z], t, s, hover, hover[5])
            z -= 0.005
        if pre is None:
            return [ready, hover]
        down, _ = self.cart_path(pre, hover, z_travel=0.0)
        return [ready] + list(down)

    def rise_to(self, q6, z):
        """Straight up until the gripper site is z above the board (or as high as the arm reaches there); [q6] if it
        already is."""
        dz = self.z0 + z - self.fk(q6)[0][2]
        return self.above(q6, dz) if dz > 0.005 else np.array([q6], float)

    def above(self, q6, dz, step=0.01):
        """Straight up by dz from the current pose, same orientation (for the retreat after a release / a miss)."""
        p, _ = self.fk(q6); t, s = self.orientation(q6)
        W, prev = [np.asarray(q6, float)], np.asarray(q6, float)
        for i in range(1, int(math.ceil(dz / step)) + 1):
            q, _ = self.ik_near([p[0], p[1], p[2] + dz * i / math.ceil(dz / step)], t, s, prev, q6[5])
            if q is None:
                break
            W.append(q); prev = q
        return np.array(W)


class StagedMove:
    """One move, one control tick at a time: step(q6, img, now) -> joint target [6] (or None = hold where it is).

    skill: object with start(text, which, to_xyz) (new stage: reset the policy's queue, set the markers) and
    step(q6, img) -> action [6] or None. ready6: the ready pose. drop_xyz: where a capture goes (the ring).
    After the move, .result is ("placed" | "failed", reason) and .events the stage log [(t, stage, note)]."""

    def __init__(self, planner, skill, task, session, ready6, now, drop_xyz=None):
        m = MOVE.search(task)
        if not m:
            raise ValueError(f"not a move task: {task}")
        self.colour, self.piece, frm, to = m.groups()
        self.Pl, self.skill, self.task, self.ready = planner, skill, task, np.asarray(ready6, float)
        P = planner.P
        self.from_xyz = P.square_xyz(frm, session)
        if to == "drop zone":
            drop_xyz = P.drop_xyz(session) if drop_xyz is None else drop_xyz
        self.to_xyz = np.asarray(drop_xyz if to == "drop zone" else P.square_xyz(to, session), float)   # z: the model's board
        self.ring = self.to_xyz if to == "drop zone" else None        # drop zone: the ring marks the chosen spot
        self.result, self.events, self.retries, self.close_h = None, [], 0, None
        self.info = {}
        self.stage, self.t0, self.traj = None, now, None
        hover, hi = planner.grasp_hover(self.from_xyz, self.piece, self.ready, HOVER_GRIP)
        self.info["grasp_hover"] = hi
        if hover is None:
            self._fail(now, f"no IK solution for the hover above {frm}")
            return
        self.hover = hover
        self._go(now, "approach", Trajectory(self._approach_waypoints(hover)))

    def on_path(self):
        """True while the arm follows a planned path (not the skill model)."""
        return self.stage not in SKILL_STAGES

    # ---- helpers
    def _approach_waypoints(self, hover):
        return self.Pl.approach(self.ready, hover)

    def _go(self, now, stage, traj=None, note=""):
        self.stage, self.t0, self.traj = stage, now, traj
        self._tau, self._prev = 0.0, now                  # path clock of traj (see Trajectory.advance)
        self._hold_t = self._empty_t = self._rel_t = self._drop_t = None
        self._opened = False
        self.events.append((round(now, 2), stage, note))
        if stage == "grasp":
            self.skill.start(f"pick up the {self.colour} {self.piece}", ("from",), None)
        elif stage == "place":
            self.skill.start(f"put the {self.colour} {self.piece} down on the ring", ("to",), self.ring)

    def _timeout(self, default):
        """The skill's own stage timeout when it has one (play/llm_skill.py: several looks of a few seconds each)."""
        t = getattr(self.skill, "timeout_s", None)
        return float(t) if t else default

    def _fail(self, now, reason):
        self.result = ("failed", reason)
        self.events.append((round(now, 2), "failed", reason))
        self.stage = "failed"

    def _lost(self, now, q6):
        """The follower reads empty during the carry. KEEP_GOING: note it once and carry on (False); else fail (True)."""
        if KEEP_GOING:
            if "hold_doubt_pct" not in self.info:
                self.info["hold_doubt_pct"] = round(float(q6[5]), 2)
                self.events.append((round(now, 2), self.stage, f"gripper reads {q6[5]:.1f} % (empty?): carrying on"))
            return False
        self._fail(now, "lost the piece during the carry"); return True

    def _confirm(self, attr, cond, now, need):
        """Debounce: True once cond has held continuously for `need` seconds."""
        if not cond:
            setattr(self, attr, None); return False
        if getattr(self, attr) is None:
            setattr(self, attr, now)
        return now - getattr(self, attr) >= need

    def _retry(self, q6, now, why):
        if self.retries >= MAX_RETRIES:
            self._fail(now, f"{why}; gave up after {MAX_RETRIES} retries"); return None
        self.retries += 1
        measured = getattr(self.skill, "measured_from_xy", None)
        if measured is not None:
            # The camera has seen where the piece is: re-plan the hover THERE instead of returning to the planned one. A
            # sideways nudge at hover height can be out of reach near the base (f8 bishop, 2026-11-10: 6-8 mm corrections
            # never executed, three attempts); the approach planner searches tilt and height for a reachable pose.
            new_from = np.asarray(self.from_xyz, float).copy(); new_from[:2] = np.asarray(measured, float)[:2]
            shift = 1000.0 * float(np.linalg.norm(new_from[:2] - np.asarray(self.from_xyz, float)[:2]))
            if 1.0 <= shift <= 40.0:
                hover, hi = self.Pl.grasp_hover(new_from, self.piece, self.ready, HOVER_GRIP)
                if hover is not None:
                    self.from_xyz, self.hover = new_from, hover
                    self.info.setdefault("replanned_hover_mm", []).append(round(shift, 1))
                    why = f"{why}; hover re-planned {shift:.0f} mm away, at the seen piece"
        up = self.Pl.above(q6, 0.03)
        up[:, 5] = q6[5]                                  # rise with the jaws as they are: a miss holds nothing, and opening
        W = list(up) + [self.hover]                       # them between two pieces would push a neighbour (twin: -1.5 mm)
        self._go(now, "retreat", Trajectory(W), f"{why}: retry {self.retries}/{MAX_RETRIES}")
        return self.traj.at(0.0)

    # ---- the tick
    def step(self, q6, img, now):
        q6 = np.asarray(q6, float)
        st, t = self.stage, now - self.t0
        if self.result is not None:
            return None
        if st in ("approach", "retreat", "carry", "lift", "return"):
            if st == "carry" and self._confirm("_drop_t", q6[5] <= EMPTY_MAX, now, DROP_CONFIRM_S) and self._lost(now, q6):
                return None
            self._tau = self.traj.advance(self._tau, q6, now - self._prev); self._prev = now
            a = self.traj.at(self._tau)
            if self.traj.done(self._tau):
                if st in ("approach", "retreat", "carry"):        # at the hover: take out the sag, then the skill
                    self._next, self._target, self._bias = "place" if st == "carry" else "grasp", self.traj.W[-1].copy(), np.zeros(6)
                    self._go(now, "hold")
                elif st == "lift":
                    # the approach in reverse: straight up to APPROACH_Z, then the joint move home. A direct joint move
                    # from 4 cm above the board dipped: the real gripper skimmed the board at 1-3 mm (drop zone worst)
                    W = list(self.Pl.rise_to(q6, APPROACH_Z)) + [self.ready]
                    self._go(now, "return", Trajectory(W))
                elif st == "return":
                    self.result = ("placed", f"{self.retries} retries")
                    self.events.append((round(now, 2), "done", ""))
            return a
        if st == "hold":
            if self._next == "place" and self._confirm("_drop_t", q6[5] <= EMPTY_MAX, now, DROP_CONFIRM_S) and self._lost(now, q6):
                return None
            err = self._target[:5] - q6[:5]
            if np.abs(err).max() <= SAG_TOL_DEG or t >= SAG_MAX_S:
                self.info.setdefault("sag_deg", []).append([round(float(v), 2) for v in self._bias[:5]])
                self._go(now, self._next)
            else:
                self._bias[:5] = np.clip(self._bias[:5] + SAG_GAIN * err, -SAG_MAX_DEG, SAG_MAX_DEG)
            return self._target + self._bias
        if st == "grasp":
            a = self.skill.step(q6, img)
            if a is None:
                return None
            fatal = getattr(self.skill, "fatal_error", None)
            if fatal:
                self._fail(now, fatal); return None  # network failure must not trigger an arm retreat/retry
            why = getattr(self.skill, "gave_up", None)        # play/llm_skill.py: the target is not in view / the way is blocked
            if why:
                return self._retry(q6, now, f"skill gave up ({why})")
            cmd, fol = float(a[5]), float(q6[5])
            self._opened = self._opened or cmd > OPENED_CMD
            # holding: the jaws stall on the piece. While they are still closing the follower also trails the command
            # (100-240 ms lag at P=16), so require it to have stopped moving for HOLD_CONFIRM_S as well.
            blocked = fol >= HOLD_MIN and cmd <= fol - HOLD_GAP and cmd < OPENED_CMD
            if blocked and self._hold_t is not None and abs(fol - self._hold_f) > STALL_TOL:
                self._hold_t = None                         # still moving: restart the window
            if blocked and self._hold_t is None:
                self._hold_f = fol
            if self._confirm("_hold_t", blocked, now, HOLD_CONFIRM_S):
                p, _ = self.Pl.fk(q6)
                self.close_h = float(p[2] - self.from_xyz[2])
                grip = np.array(q6); grip[5] = cmd
                ph, pi = self.Pl.place_hover(self.to_xyz, self.close_h, grip, cmd)
                self.info["place_hover"] = pi
                if ph is None:
                    self._fail(now, "no IK solution for the place hover"); return None
                W, ci = self.Pl.cart_path(grip, ph, z_travel=max(self.from_xyz[2], self.to_xyz[2]) + self.close_h + TRAVEL_CLEAR)
                self.info["carry"] = ci
                self._go(now, "carry", Trajectory(W), f"holding (follower {fol:.1f} %, grasped {100 * self.close_h:.1f} cm above the board)")
                return self.traj.at(0.0)
            if self._opened and self._confirm("_empty_t", fol <= EMPTY_MAX and cmd <= CLOSED_CMD, now, MISS_CONFIRM_S):
                return self._retry(q6, now, "closed on nothing")
            if t > self._timeout(GRASP_TIMEOUT_S):
                return self._retry(q6, now, f"no grasp after {self._timeout(GRASP_TIMEOUT_S):.0f} s")
            return a
        if st == "place":
            a = self.skill.step(q6, img)
            if a is None:
                return None
            fatal = getattr(self.skill, "fatal_error", None)
            if fatal:
                self._fail(now, fatal); return None  # network failure must not trigger an arm retreat/retry
            why = getattr(self.skill, "gave_up", None)
            late = t > self._timeout(PLACE_TIMEOUT_S)
            if (why or late) and not KEEP_GOING:
                self._fail(now, f"place skill gave up ({why})" if why else f"no release after {self._timeout(PLACE_TIMEOUT_S):.0f} s"); return None
            if why or late:                                   # KEEP_GOING: open the jaws where they are, then lift as usual
                if "blind_release" not in self.info:
                    self.info["blind_release"] = why or "timeout"
                    self.events.append((round(now, 2), "place", f"releasing where it is ({why or 'timeout'})"))
                a = np.array(a, float); a[5] = max(float(a[5]), BLIND_RELEASE_PCT)
            cmd, fol = float(a[5]), float(q6[5])
            if self._confirm("_rel_t", cmd >= OPENED_CMD and fol >= OPENED_CMD - 1.0, now, RELEASE_CONFIRM_S):
                up = self.Pl.above(q6, LIFT_AFTER_RELEASE)
                up[:, 5] = cmd
                self._go(now, "lift", Trajectory(up), f"released (command {cmd:.1f} %)")
                return self.traj.at(0.0)
            return a
        return None
