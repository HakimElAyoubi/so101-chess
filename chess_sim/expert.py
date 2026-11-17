"""Scripted pick-and-place expert for one chess move in the MuJoCo scene.

plan_move(scene, move, board) -> MovePlan (a list of Waypoint) that implements the move with the SO-101:
  capture      : captured piece -> next free drop-zone cell first, then the mover
  en passant   : the captured pawn (behind the target square) -> drop cell, then the mover
  castling     : king, then rook (rook first when the king-first order has no collision-free rook grasp)
  promotion    : pawn -> drop cell, then the spare queen of that colour -> promotion square (v1: always a queen)
Every pick-and-place is: pre-grasp above the piece at lift height (jaws closed, opened there) -> descent (lean
settling, then a slow constant-orientation leg) to the grasp height -> close (settle) -> slow two-stage lift
(constant orientation first) -> straight-line transit at lift height -> descent -> open (settle) -> centre the
open jaws on the stem -> slow two-stage retreat.  The plan starts from the current arm pose (a far-away start
pose, e.g. the MJCF zero, is first brought to the watch pose by a collision-checked joint-space move) and ends
at the watch pose.

Every leg beside pieces is checked knot by knot (5 mm) on a scratch MjData: finger boxes vs every other piece
(>= CLEARANCE_PATH), finger boxes vs the TARGET piece itself (the one being grasped, or the one just released,
>= TARGET_CLEARANCE_MIN: a lean along the opening axis sweeps the finger root into the stem), and the CARRIED
piece (rigid along the jaw axis, or hanging vertical) vs every other piece (>= CARRY_CLEARANCE).  Grasp and
release candidates are ranked and the first with safe legs wins (GRASP_CANDIDATES_TRIED).  A lifted piece is
held rigidly, so it lands with the tilt between the grasp and release jaw orientations (landing_tilt_deg): the
release candidates are ranked by that predicted tilt (<= LANDING_TILT_MAX first) and, since a far corner (a1/h1)
leaves a single reachable release orientation, the GRASP candidates are ranked by release compatibility too
(rank_grasps_for_release).  A release uses the HELD stem position (kinematics.place_tool_site); at execution time
the in-jaw offset is measured just before the final descent and the descent / release / retreat legs are
re-planned so the base lands on the square (Expert._adapt_place, shifts >= ADAPT_MIN, capped to ADAPT_MAX).
Execution aborts when a carried piece is lost or a standing /
released piece is knocked over (KNOCKED_TILT_DEG / KNOCKED_DISP) so a game loop can stop.

Reach-limited squares.  The SO-101 cannot raise its vertical jaws above ~0.046 m within r ~ 0.08 m of the
shoulder-pan axis (wrist_flex limit, 0.159 m gripper): over c8..f8 (and the king on rank 7) the OPEN jaws cannot
pass over the piece's top.  Such pieces get a SIDE approach (arrive beside the piece at the grasp height, slide in
so the stem enters between the fingers), are dragged ("slide") rather than lifted when even a few mm of lift are
impossible, and after the release the jaws back out sideways (SIDE retreat) before closing and climbing, so the
placed piece is never touched.  Empty jaws travel CLOSED and open only above the target.  A blocked carry is
re-routed with an A* corridor search over the board (grid_route) and, failing that, retried with a lower grasp
(50 %, 40 % of the piece height: a lower-hanging piece passes over neighbours where the ceiling is low).  Every
path knot is checked exactly: finger boxes vs every piece's collision cylinders (jaw_piece_clearance) and arm
self-collision, on a scratch MjData (the live simulation is untouched).  Known physical limits (PlanError): a
rank-8 king/queen that must cross diagonally between two occupied squares (bases touch and cannot be lifted over
them), and castling with all of d7/e7/f7 (and f8/d8 neighbours) occupied.

Poses come from kinematics.Kinematics (damped-least-squares IK).  Grasp/release poses have near-vertical jaws
(<= grasp_max_lean, 8 deg; up to GRASP_MAX_LEAN_SLIDE for pieces that are slid anyway); the opening direction
(LAT_CANDIDATES_DEG relative to the radial direction), lean, grasp-height offset and jaw opening (OPENING_GAPS)
are searched and ranked by the clearance of the open fingers from the neighbouring pieces (>= CLEARANCE_GOOD
preferred: +-3 mm piece jitter + tracking error; >= CLEARANCE_MIN accepted).

execute(scene, plan, ...) runs the plan in the physics (position control at CONTROL_HZ, min-jerk time profile
per segment under the per-joint speed limit, settling waits after close/open) and returns a MoveResult with the
spec's success definition: moved piece(s) within SUCCESS_POS_TOL of their targets, tilt < SUCCESS_TILT_DEG,
no other piece displaced > DISTURB_TOL, gripper retreated to the planned lift height, the arm back at the watch
pose, and no arm-piece contact after the release.  action_generator(scene, plan, ...) yields the same actions one
control step at a time (LeRobot convention: [5 joints deg, gripper %]) for the gym environment.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import chess
import mujoco
import numpy as np

if __package__ in (None, ""):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chess_sim import config as C
    from chess_sim.kinematics import (Kinematics, IKResult, radial_lat, rotate_lat, grasp_height, frame_params,
                                      TRAVEL_TILT_DEG, grasp_max_lean, GRASP_SIDE_CLEARANCE, SPEC_DEVIATIONS, HELD_Z_OFFSET)
else:
    from . import config as C
    from .kinematics import (Kinematics, IKResult, radial_lat, rotate_lat, grasp_height, frame_params,
                             TRAVEL_TILT_DEG, grasp_max_lean, GRASP_SIDE_CLEARANCE, SPEC_DEVIATIONS, HELD_Z_OFFSET)

# --------------------------------------------------------------------------- parameters
# opening direction relative to the radial direction (0 = moving jaw on the far side).  180 (moving jaw toward
# the base) is NOT in the list: the wrist-roll range (-157..+162 deg) cannot realise it.
LAT_CANDIDATES_DEG = (0.0, 90.0, -90.0, 45.0, -45.0, 135.0, -135.0, 22.5, -22.5, 67.5, -67.5, 112.5, -112.5, 157.0, -157.0)
CLEARANCE_MIN = 0.002           # open jaws vs neighbouring piece collision geometry (m): accepted
CLEARANCE_MIN_SLIDE = 0.0015    # accepted for a piece that cannot be lifted anyway (side approach / slide: no alternative)
CLEARANCE_GOOD = 0.004          # preferred: covers the +-3 mm piece jitter plus ~1 mm tracking error
CLEARANCE_PATH = 0.002          # finger boxes vs pieces at every path knot (m)
NEIGHBOUR_RADIUS = 0.085        # pieces within this xy distance of the tool are checked (m)
KNOT_SPACING = 0.015            # Cartesian path sampling (m)
DESCENT_KNOT = 0.005            # knot spacing of the legs beside pieces (descents, lifts, retreats): the finger checks are per knot
MIN_SEGMENT_TIME = 0.25         # s
GRIPPER_MOVE_TIME = 0.30        # s, interpolation of a gripper open/close command
V_TOOL_MAX = 0.30               # m/s peak tool speed along a Cartesian path
V_DESCENT_MAX = 0.12            # m/s peak tool speed of the final vertical descents / slides (accuracy near neighbours)
# obstacle corridor: a piece counts as "under the path" when its STEM is within sweep_r + stem radius of the tool
# xy (full height) or its BASE disc within sweep_r + base radius (base height only).  The radii are chosen so that
# the pieces on the adjacent squares (28 mm away) never count: a carried piece or the closed jaws pass between them.
SWEEP_R_CARRY = 0.014           # carried piece base radius (<= 10 mm) + 4 mm
SWEEP_R_EMPTY = 0.016           # empty jaws (tips / finger boxes around the stem point) + margin
OBSTACLE_MARGIN = 0.004         # vertical clearance over pieces under the corridor (m)
PRE_DESCENT_MIN = 0.015         # the constant-orientation final descent is at least this long (m)
TIP_DROP = 0.008                # fingertips are this far below the stem point (m)
SIDE_APPROACH = 0.025           # side approach / retreat: arrive this far beside the piece at grasp height, then slide (m)
SIDE_APPROACH_ALT = (0.025, 0.020)   # side distances tried (the fingers extend ~10 mm along the slide direction: >= 18 mm)
SLIDE_KNOT_MIN = 0.0010         # neighbour clearance accepted at the knots of a slide-in / back-away leg (the side points
                                # themselves need CLEARANCE_MIN_SLIDE; the pieces are exactly where the plan sees them)
PLACE_EXTRA_Z = 0.0005          # release with the piece base 0.5 mm above the surface
SETTLE = dict(grasp=0.15, place=0.15, close_min=0.25, close_max=0.9, open=0.30, lift=0.0, transit=0.0, watch=0.3,
              pregrasp=0.0, retreat=0.0)
GRIPPER_SETTLE_VEL = 0.05       # rad/s, gripper joint considered settled below this
DROP_CHECK_TOL = 0.015          # carried piece must stay within this of its expected carried position (m)
LIFT_Z_TOL = 0.005
LIFT_MARGIN = 0.002             # stay this far below the reach ceiling (m; every knot is verified by the IK anyway)
LEAN_OK_Z = 0.085               # a leaning carry is used only where it reaches at least this tool height (m)
REACH_R_STEP = 0.005            # radial resolution of the reach-ceiling profile (m)
REACH_R_BAND = 0.005            # the ceiling at a point is the min over r +- this (tool lateral offset) (m)
GRASP_TILT_FALLBACK = (0.0, -2.5, 2.5, -5.0, 5.0, -7.5, 7.5, -8.0, 8.0, -10.0, 10.0, -12.5, 12.5, -15.0, 15.0)   # 8.0: the king at 29 mm on a1/h1
# Far-reach fallback tier: when NO pose within grasp_max_lean (8 deg) exists for a lifted grasp / release, leans up
# to GRASP_MAX_LEAN_FAR are tried (a separate, later tier: wherever 8 deg suffices the choice is unchanged).  It
# extends the reach at the far corners (a1/h1/b1/g1) by ~3.3 mm per degree, which the randomised board pose needs:
# a board shifted sideways / rotated pushes one far corner up to ~11 mm further out (README, reach analysis).
GRASP_MAX_LEAN_FAR = 12.5
FAR_TIER_MIN_RADIUS = 0.25      # m from the pan axis: the far tier only serves the far ranks (never the rank-8 side approaches)
GRASP_MAX_LEAN_SLIDE = 15.0     # lean accepted for a piece that is slid (not lifted) anyway: it stays upright on its base
GRASP_DZ_FALLBACK = C.GRASP_DZ_CYLINDERS                   # 'cylinders': grasp height offsets tried (m, still on the stem);
                                                           # 'bands' uses config.grasp_dz_candidates (inside the grasp zone)
GRASP_FRAC_FALLBACK = (None, 0.50, 0.40)                   # grasp height fractions tried per pick-place (None = spec); a lower
                                                           # grasp lets a carried piece pass over neighbours where the ceiling is low
OPENING_GAPS = (C.APPROACH_GAP, 0.036, 0.028, 0.024)       # jaw openings tried for a grasp/release, widest first (m)
OPENING_MIN_EXTRA = 0.012       # last resort opening: stem diameter + this (m); the jaw clearance is ~5.5 mm less than the gap
# The TARGET piece (the one being grasped, or the one just released) is included in the finger checks: the finger
# collision boxes must stay >= TARGET_CLEARANCE_MIN from its stem at the grasp/release pose and along the descent,
# slide-in, centring shift, retreat and back-away legs (the nominal clearance of the fixed finger is
# GRASP_SIDE_CLEARANCE - ~1 mm of box overhang = 1.5 mm; a 5 deg lean along the opening axis sweeps the finger root
# ~2.6 mm, which is how the g1 knight / c8 queen were knocked over before this check existed).
TARGET_CLEARANCE_MIN = 0.0008
# At a RELEASE pose the stem is where the held piece put it, HELD_Z_OFFSET (-1.1 mm) nearer the fixed finger, so the
# nominal clearance there is ~0.4 mm: release poses and un-centred post-release legs are checked against this
RELEASE_CLEARANCE_MIN = TARGET_CLEARANCE_MIN + HELD_Z_OFFSET
CARRY_CLEARANCE = 0.002         # the CARRIED piece's collision cylinders (held rigidly along the jaw axis, or hanging vertical) vs
                                # every other piece at each knot of the lift / carry / place-descent legs (m)
CENTRE_SHIFT_MAX = 0.008        # after a release: shift the open jaws along the opening axis so the stem is centred between
                                # the fingers (up to this, m) before any lean/roll change or sideways retreat
GRASP_CANDIDATES_TRIED = 6      # ranked grasp (and release) candidates tried when the descent/retreat legs fail the target check
KNOCKED_TILT_DEG = 20.0         # execution abort: a standing piece (not the one carried) tilted beyond this
# Execution-time place correction: just before the final descent the carried piece's offset in the jaws is measured
# (it depends on how the pads grabbed it: hysteresis, not predictable at plan time) and the descent / release /
# retreat legs are re-planned so the base lands on the square centre; applied for shifts in [ADAPT_MIN, ADAPT_MAX]
ADAPT_MIN = 0.0010
ADAPT_MAX = 0.015               # larger measured shifts are applied capped to this (then half of it), never ignored
# A lifted piece is held rigidly at the grasp lean / opening direction: released with another (lean, direction) it
# lands tilted by the angle between the two jaw-axis orientations (landing_tilt_deg) and a king topples beyond
# ~10 deg (Kg1-h1: grasp lean 7.5 deg at +22 deg, release at +112 deg -> 10.6 deg).  Release candidates whose
# predicted landing tilt exceeds this are ranked after the others (stable: the clearance ranking is kept inside
# each tier) instead of being chosen for their clearance alone.
LANDING_TILT_MAX = 4.0
KNOCKED_DISP = 0.015            # execution abort: a released piece displaced beyond this from its target (m)
ROUTE_CELL = 0.014              # grid cell of the A* corridor planner used when the straight transit is blocked (m)
ROUTE_RETRIES = 8               # re-routes around points where a routed leg fails the exact knot checks
TRAVEL_PCT = 0.0                # jaw opening (LeRobot %) while travelling with empty jaws: closed (7 mm gap)
START_NEAR_WATCH_DEG = 15.0     # a start pose farther than this (max joint) from the watch pose is first moved there
CONTACT_STEPS_FAIL = 10         # arm-piece contact with a RELEASED piece during >= this many control steps (0.33 s) fails the move
                                # (a brief brush while the jaw opens / the piece settles against the finger is tolerated)
SELF_COLLISION_TOL = 0.0005     # arm self-penetration below this (m) is treated as touching, not colliding (mesh tolerance)
KNIGHT_AXIS_TOL_DEG = 25.0      # 'bands': opening directions accepted around the knight's narrow axis (20 deg off: 1.2 mm shift, measured)
LEAN_IK_TOL_DEG = 0.1           # the IK realises a requested lean within rot_tol (0.06 deg): a candidate at the max lean is kept
KNIGHT_ALIGN_SNAP_DEG = 10.0    # a knight squeezed further off its narrow axis turns to align in the jaws (Expert.theta_held)


class PlanError(RuntimeError):
    """The move cannot be planned (unreachable pose, no collision-free jaw direction, no spare queen, ...).
    `where` (optional xyz) marks the path point that failed, so the corridor planner can route around it."""

    def __init__(self, msg: str, where=None):
        super().__init__(msg)
        self.where = where


def minjerk(tau: float) -> float:
    tau = min(1.0, max(0.0, tau))
    return tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)


@dataclass
class Waypoint:
    """A joint-space target (LeRobot convention) reached through `via` knots (joint space, piecewise linear in
    the path parameter with a min-jerk time profile), then held for `dwell` seconds (or, for kind 'close', until
    the gripper joint has settled)."""
    q6: np.ndarray
    kind: str                           # pregrasp, preopen, approach, descend_grasp, close, lift, transit, descend_place, open, retreat, watch
    label: str = ""
    via: List[np.ndarray] = field(default_factory=list)
    dwell: float = 0.0
    tool_path_len: float = 0.0          # Cartesian length of the tool path (m), for the speed limit
    meta: dict = field(default_factory=dict)


@dataclass
class PickPlace:
    piece: str
    ptype: str
    src_xyz: np.ndarray                 # base-centre position when planned
    dst_xyz: np.ndarray                 # target base-centre position (surface)
    dst_name: str
    grasp: Optional[IKResult] = None
    place: Optional[IKResult] = None
    grasp_lat_deg: float = 0.0
    place_lat_deg: float = 0.0
    grasp_clearance: float = 0.0
    place_clearance: float = 0.0
    open_pct: float = 0.0               # opening used for the grasp
    place_open_pct: float = 0.0         # opening used for the release
    close_pct: float = 0.0
    side_approach: bool = False
    side_retreat: bool = False
    slide: bool = False                 # the piece is dragged (no lift possible at its square)
    hang: float = 0.0                   # grasp height used (stem point above the piece base, m)
    landing_tilt_deg: float = 0.0       # predicted tilt of the released piece (grasp vs release jaw orientation), deg


class MovePlan(list):
    """list of Waypoint plus bookkeeping (steps, expected final piece positions, planning time)."""

    def __init__(self, move: Optional[chess.Move] = None):
        super().__init__()
        self.move = move
        self.steps: List[PickPlace] = []
        self.expected: Dict[str, np.ndarray] = {}      # piece -> target base position
        self.plan_seconds = 0.0
        self.notes: List[str] = []
        self.watch_error: Optional[str] = None         # set when the return to the watch pose could not be planned

    @property
    def waypoints(self) -> List[Waypoint]:
        return list(self)

    @property
    def ends_at_watch(self) -> bool:
        return bool(self) and self[-1].kind == "watch"


@dataclass
class MoveResult:
    success: bool
    reasons: List[str]
    metrics: dict
    timings: List[Tuple[str, float, float]]     # (label, sim seconds, wall seconds)
    sim_seconds: float
    control_steps: int


# --------------------------------------------------------------------------- expert
class Expert:
    def __init__(self, scene):
        self.scene = scene
        self.collision = scene.cfg.piece_collision          # 'bands' (mesh hulls) | 'cylinders': grasp tables + envelopes
        self.kin = Kinematics(scene.model, collision=self.collision)
        m = scene.model
        self.finger_geoms = [g for g in range(m.ngeom) if (m.geom(g).name or "").endswith(("finger_0", "finger_1", "finger_2"))
                             and ("fixed_" in m.geom(g).name or "moving_" in m.geom(g).name)]
        assert len(self.finger_geoms) == 6, self.finger_geoms
        # every collision geom of each piece body: the base + stem cylinders, or the convex hull bands of the mesh
        self.piece_col_geoms: Dict[str, List[int]] = {
            n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == scene.piece_body_id[n] and m.geom_contype[g] != 0]
            for n in scene.piece_names}
        assert all(len(v) >= 2 for v in self.piece_col_geoms.values()), "piece bodies without collision geoms"
        self.piece_bodies = set(scene.piece_body_id.values())
        self.gripper_jid = m.joint(C.GRIPPER_JOINT).id
        self.state: dict = dict(phase="idle", waypoint=-1, label="")
        self._planned_pos: Dict[str, np.ndarray] = {}     # piece -> position after the steps planned so far
        self._held_theta: Dict[str, Optional[float]] = {}  # piece -> direction (piece frame, deg) of the fixed finger at its grasp
        self._reach_r: Optional[np.ndarray] = None        # reach ceiling profiles (see reach_profile)
        self._reach_z: Optional[np.ndarray] = None
        self._reach_z_lean: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ geometry helpers
    def lift_z(self) -> float:
        return self.scene.board_top_z + C.LIFT_CLEARANCE

    def z_safe_above(self, piece_xyz, ptype: str) -> float:
        """Lowest tool z at which the open jaws clear the top of a piece standing at piece_xyz."""
        return float(piece_xyz[2] + C.PIECE_SPECS[ptype][0] + TIP_DROP + OBSTACLE_MARGIN)

    def reach_profile(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reach ceilings: the highest tool-point z (m) of a top-down pose as a function of the distance r from
        the shoulder-pan axis (rotational symmetry), for a jaw lean <= JAW_VERTICAL_TOL_DEG ('vertical') and
        <= TRAVEL_TILT_DEG ('lean').  Measured with the IK (pawn stem tool, 5 mm radial steps, 2 mm z
        resolution) and cached in data/reach_profile.json (keyed by the arm geometry).  The SO-101 wrist_flex
        limit (95 deg) gives vertical ceilings of ~0.04 m at r = 0.065 (rank 8 centre), 0.09 m at r = 0.15..0.19
        and 0.04 m again at r = 0.27; with a 30 deg lean the far side rises to ~0.16 m but the near side does not.
        Returns (r, z_vertical, z_lean)."""
        if self._reach_r is not None:
            return self._reach_r, self._reach_z, self._reach_z_lean
        k = self.kin
        tool = k.grasp_tool_site("pawn")
        key = dict(travel_tilt=TRAVEL_TILT_DEG, vert_tilt=C.JAW_VERTICAL_TOL_DEG, pan=[round(float(v), 6) for v in k.pan_axis_xy],
                   tool=[round(float(v), 6) for v in tool], roll_axis=[round(float(v), 6) for v in k.roll_axis_site],
                   lo=[round(float(v), 6) for v in k.lo], hi=[round(float(v), 6) for v in k.hi], step=REACH_R_STEP, v=2)
        path = os.path.join(C.DATA_DIR, "reach_profile.json")
        try:
            with open(path) as f:
                rec = json.load(f)
            if rec.get("key") == key:
                self._reach_r, self._reach_z, self._reach_z_lean = np.array(rec["r"]), np.array(rec["z"]), np.array(rec["z_lean"])
                return self._reach_r, self._reach_z, self._reach_z_lean
        except (OSError, ValueError):
            pass
        rs = np.arange(0.045, 0.33 + 1e-9, REACH_R_STEP)
        out = {}
        for name, max_tilt, step in (("z", C.JAW_VERTICAL_TOL_DEG, 2.5), ("z_lean", TRAVEL_TILT_DEG, 5.0)):
            zs = []
            z_prev, q_prev = 0.03, None
            for r in rs:
                p = np.array([k.pan_axis_xy[0] + r, k.pan_axis_xy[1], 0.0])
                lat = radial_lat(p, 1.0, k.pan_axis_xy)

                def ok(z):
                    nonlocal q_prev
                    rr = k.solve_reach(np.array([p[0], p[1], z]), lat, tool, max_tilt_deg=max_tilt, tilt_step=step,
                                       q_init=q_prev, seeds=q_prev is None)
                    if rr.converged:
                        q_prev = rr.q5
                    return rr.converged
                z = max(0.015, z_prev - 0.015)
                while not ok(z) and z > 0.015:
                    z -= 0.005
                if not ok(z):
                    zs.append(0.0)
                    continue
                while ok(z + 0.005):
                    z += 0.005
                for dz in (0.004, 0.002):
                    if ok(z + dz):
                        z += dz
                zs.append(z)
                z_prev = z
            out[name] = np.array(zs)
        self._reach_r, self._reach_z, self._reach_z_lean = rs, out["z"], out["z_lean"]
        try:
            os.makedirs(C.DATA_DIR, exist_ok=True)
            with open(path, "w") as f:
                json.dump(dict(key=key, r=rs.tolist(), z=out["z"].tolist(), z_lean=out["z_lean"].tolist()), f)
        except OSError:
            pass
        return self._reach_r, self._reach_z, self._reach_z_lean

    def reach_ceiling(self, xy, lean: bool = False) -> float:
        """Highest reachable tool z at a world xy (lean <= 5 deg, or <= TRAVEL_TILT_DEG with lean=True), from the
        profile; conservative over +-REACH_R_BAND of radius (the tool's lateral offset from the roll axis)."""
        rs, zv, zl = self.reach_profile()
        zs = zl if lean else zv
        r = float(np.hypot(xy[0] - self.kin.pan_axis_xy[0], xy[1] - self.kin.pan_axis_xy[1]))
        return float(min(np.interp(r - REACH_R_BAND, rs, zs), np.interp(r, rs, zs), np.interp(r + REACH_R_BAND, rs, zs)))

    def lift_z_at(self, xy) -> float:
        """Lift / carry height above a point: the nominal board + LIFT_CLEARANCE where reachable (with a lean of
        up to TRAVEL_TILT_DEG on the far ranks, where a hanging piece is high enough for the lean to be
        harmless), otherwise the ceiling of near-vertical jaws minus LIFT_MARGIN (ranks 7-8: 3..7 cm, a strong
        lean there would swing the low-hanging piece into its neighbours)."""
        z_v = self.reach_ceiling(xy, lean=False) - LIFT_MARGIN
        z_l = self.reach_ceiling(xy, lean=True) - LIFT_MARGIN
        if z_l >= LEAN_OK_Z:
            return float(min(self.lift_z(), z_l))
        return float(min(self.lift_z(), z_v))

    def _scratch_pose(self, q5, gripper_q: float) -> None:
        """Scratch MjData = live scene with the pieces of earlier plan steps at their planned spots and the arm at q5."""
        s, k = self.scene, self.kin
        d = k.data
        d.qpos[:] = s.data.qpos
        for n, pos in self._planned_pos.items():
            a = s.piece_qadr[n]
            d.qpos[a:a + 3] = pos
            d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[k.qpos_adr[:5]] = q5
        d.qpos[k.qpos_adr[5]] = gripper_q
        mujoco.mj_kinematics(s.model, d)

    def arm_self_collision(self, q5, gripper_q: float) -> List[Tuple[str, str, float]]:
        """Arm self-collisions and arm-vs-table/board contacts at a configuration (MuJoCo collision detection on
        the scratch data; pairs present at the zero pose are ignored).  The wrist camera module sticks out on
        the moving-jaw side of the gripper and hits the shoulder/upper arm when the arm is folded over the base
        (ranks 7-8) unless the wrist roll turns it away, which is why every candidate pose is checked.
        Returns [(body a, body b, penetration m)]."""
        s, m = self.scene, self.scene.model
        d = self.kin.data
        self._scratch_pose(q5, gripper_q)
        mujoco.mj_collision(m, d)
        out = []
        arm = s.arm_body_ids
        static = {0, s.board_id}
        for i in range(d.ncon):
            con = d.contact[i]
            if con.dist >= -SELF_COLLISION_TOL:
                continue
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if (b1 in arm and (b2 in arm or b2 in static)) or (b2 in arm and (b1 in arm or b1 in static)):
                pair = (min(b1, b2), max(b1, b2))
                if pair in s._baseline_pairs:
                    continue
                out.append((m.body(b1).name, m.body(b2).name, float(-con.dist)))
        return out

    def arm_contacts(self, q5, gripper_q: float) -> List[Tuple[str, str, float]]:
        """Every contact of the arm (self, board, table, pieces) at a configuration (scratch data)."""
        s, m = self.scene, self.scene.model
        d = self.kin.data
        self._scratch_pose(q5, gripper_q)
        mujoco.mj_collision(m, d)
        out = []
        arm = s.arm_body_ids
        for i in range(d.ncon):
            con = d.contact[i]
            if con.dist >= 0:
                continue
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if b1 in arm or b2 in arm:
                pair = (min(b1, b2), max(b1, b2))
                if pair in s._baseline_pairs:
                    continue
                out.append((m.body(b1).name, m.body(b2).name, float(-con.dist)))
        return out

    def joint_path_clear(self, qa5, qb5, gripper_q: float, n: int = 25) -> Tuple[bool, str]:
        """True when the straight joint-space interpolation from qa5 to qb5 is free of arm contacts (self, board,
        pieces) at n samples."""
        qa, qb = np.asarray(qa5, float), np.asarray(qb5, float)
        for i in range(n + 1):
            q = qa + (qb - qa) * i / n
            c = self.arm_contacts(q, gripper_q)
            if c:
                return False, f"sample {i}/{n}: {c[0][0]}/{c[0][1]} ({c[0][2]*1000:.1f} mm)"
        return True, ""

    def jaw_piece_clearance(self, q5, gripper_q: float, exclude: Iterable[str] = (), tool_xy=None,
                            only: Optional[str] = None) -> Tuple[float, str]:
        """Minimum distance (m) between the jaw finger boxes and the collision cylinders of every piece (except
        `exclude`; or of the single piece `only`) within NEIGHBOUR_RADIUS of tool_xy, at arm configuration q5 /
        gripper angle, with the pieces where they are in the live scene (pieces moved by earlier steps of the plan
        at their planned spots).  Negative = penetration.  Returns (distance, closest piece)."""
        s, k, m = self.scene, self.kin, self.scene.model
        d = k.data
        self._scratch_pose(q5, gripper_q)
        if tool_xy is None:
            tool_xy = d.site_xpos[k.site_id][:2]
        ex = set(exclude)
        best, who = 0.05, ""
        for n in ([only] if only is not None else s.piece_names):
            if n in ex:
                continue
            p = d.xpos[s.piece_body_id[n]]
            if np.hypot(p[0] - tool_xy[0], p[1] - tool_xy[1]) > NEIGHBOUR_RADIUS:
                continue
            for g in self.piece_col_geoms[n]:
                for f in self.finger_geoms:
                    dist = mujoco.mj_geomDistance(m, d, f, g, 0.05, None)
                    if dist < best:
                        best, who = dist, n
        return float(best), who

    def carried_piece_clearance(self, q5, gripper_q: float, piece: str, hang: float, tool, exclude: Iterable[str] = ()) -> Tuple[float, str]:
        """Minimum distance (m) between the collision cylinders of the CARRIED piece and those of every other piece
        (except `exclude`) within NEIGHBOUR_RADIUS, at arm configuration q5, for both hold hypotheses: the piece
        rigidly aligned with the jaw axis (its base swung by hang * sin(lean)) and hanging vertical under the stem
        point.  The piece's base is hang below the stem point (tool, site frame).  Returns (distance, closest)."""
        s, k, m = self.scene, self.kin, self.scene.model
        d = k.data
        self._scratch_pose(q5, gripper_q)
        p = d.site_xpos[k.site_id].copy()
        R = d.site_xmat[k.site_id].reshape(3, 3).copy()
        stem = p + R @ np.asarray(tool, float)
        a = s.piece_qadr[piece]
        ex = set(exclude) | {piece}
        best, who = 0.05, ""
        for mode in ("rigid", "vertical"):
            if mode == "rigid":
                zp = -R[:, 0]
                base = stem + np.asarray(R[:, 0]) * hang
            else:
                zp = np.array([0.0, 0.0, 1.0])
                base = stem - np.array([0.0, 0.0, hang])
            xp = np.cross(R[:, 1], zp)
            xp /= np.linalg.norm(xp) + 1e-12
            yp = np.cross(zp, xp)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.column_stack([xp, yp, zp]).reshape(9))
            d.qpos[a:a + 3] = base
            d.qpos[a + 3:a + 7] = quat
            mujoco.mj_kinematics(m, d)
            for n in s.piece_names:
                if n in ex:
                    continue
                pn = d.xpos[s.piece_body_id[n]]
                if np.hypot(pn[0] - stem[0], pn[1] - stem[1]) > NEIGHBOUR_RADIUS:
                    continue
                for g1 in self.piece_col_geoms[piece]:
                    for g2 in self.piece_col_geoms[n]:
                        dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.05, None)
                        if dist < best:
                            best, who = dist, f"{n} ({mode})"
        return float(best), who

    def opening_candidates(self, ptype: str, open_q: Optional[float] = None) -> List[float]:
        """Gripper angles to try for a grasp/release of a piece type, widest first: OPENING_GAPS (the first
        replaced by open_q when given) down to the piece's descent envelope diameter + OPENING_MIN_EXTRA (the open
        jaws must straddle the widest part above the grasp height: collar / head / crown, = the stem for
        'cylinders')."""
        env = C.grasp_envelope_diameter(ptype, self.collision)
        gaps = [C.gripper_gap(open_q)] if open_q is not None else [OPENING_GAPS[0]]
        gaps += [g for g in OPENING_GAPS[1:]]
        gaps.append(env + OPENING_MIN_EXTRA)
        out = []
        for g in sorted(set(round(g, 4) for g in gaps), reverse=True):
            if C.gripper_clearance(C.gripper_ctrl_for_gap(g)) < env + 2 * GRASP_SIDE_CLEARANCE + 0.0005:
                continue
            out.append(C.gripper_ctrl_for_gap(g))
        return out

    def piece_yaw_deg(self, piece: str) -> float:
        q = self.scene.piece_quat(piece)
        return math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))

    def theta_fixed(self, piece: Optional[str], ang: float) -> Optional[float]:
        """Direction (deg, piece frame) from the piece axis to the FIXED finger for an opening direction `ang`
        (deg relative to the radial direction at the piece): the fixed finger sits at -lat.  None for pieces
        without a directional extent table (solids of revolution / 'cylinders')."""
        if piece is None or self.collision != "bands" or C.GRASP_TABLE_BANDS[self.scene.piece_type[piece]].get("extent_mm") is None:
            return None
        p = self.scene.piece_pos(piece)
        lat = rotate_lat(radial_lat(p, 1.0, self.kin.pan_axis_xy), ang)
        return math.degrees(math.atan2(-lat[1], -lat[0])) - self.piece_yaw_deg(piece)

    def grasp_tool(self, ptype: str, piece: Optional[str], ang: float) -> np.ndarray:
        """Grasp tool point for an opening direction (direction-dependent for the knight, see Kinematics)."""
        return self.kin.grasp_tool_site(ptype, self.theta_fixed(piece, ang))

    def theta_held(self, piece: Optional[str], ang: float) -> Optional[float]:
        """Direction of the fixed finger in the piece frame once the piece is HELD: a knight squeezed more than
        KNIGHT_ALIGN_SNAP_DEG off its narrow axis turns in the closing jaws until the axis lies across the pads
        (measured: 43 deg off at b8, 6 mm of prediction error without the snap), so the held extents are those of
        the nearest axis direction; the descent (grasp_tool) keeps the true direction."""
        th = self.theta_fixed(piece, ang)
        if th is None:
            return None
        axis = C.GRASP_TABLE_BANDS[self.scene.piece_type[piece]]["axis_deg"]
        cands = [axis, axis + 180.0]
        off = [abs((th - a + 180.0) % 360.0 - 180.0) for a in cands]
        i = int(np.argmin(off))
        return cands[i] if off[i] > KNIGHT_ALIGN_SNAP_DEG else th

    def place_tool(self, ptype: str, piece: Optional[str], ang: float) -> np.ndarray:
        return self.kin.place_tool_site(ptype, self.theta_held(piece, ang))

    def piece_axis_dirs(self, piece: Optional[str]) -> Optional[List[float]]:
        """For a piece whose grasp table names a grasp axis (the knight: across its head, 'axis_deg' in the piece
        frame), the two world directions (deg relative to the radial direction at the piece) along which the jaws
        should open, from the piece's current yaw (the hold is rigid, so the same directions serve the release);
        None for solids of revolution / 'cylinders'."""
        if piece is None or self.collision != "bands":
            return None
        s = self.scene
        rec = C.GRASP_TABLE_BANDS[s.piece_type[piece]]
        if rec.get("axis_deg") is None:
            return None
        yaw = self.piece_yaw_deg(piece)
        p = s.piece_pos(piece)
        u = radial_lat(p, 1.0, self.kin.pan_axis_xy)
        base = math.degrees(math.atan2(u[1], u[0]))
        a = yaw + rec["axis_deg"] - base
        return [(a + 180.0) % 360.0 - 180.0, (a + 360.0) % 360.0 - 180.0]

    def lat_candidates(self, xyz, piece: Optional[str] = None) -> List[float]:
        """Opening directions (deg relative to the radial direction) tried at a point: the four BOARD-ALIGNED
        directions (+-x, +-y: the fingers then straddle the piece exactly between two ranks or two files, which
        is what the crowded near ranks need) followed by the LAT_CANDIDATES_DEG grid; duplicates (< 4 deg) and
        directions outside the wrist-roll range are dropped.  For a piece with a grasp axis (the knight, 'bands')
        the two directions along its narrow axis come first (across the head the stall width is 10 mm; along it
        the jaws close on the 19.7 mm head and the piece shifts 2..4 mm in the jaws, which the execution-time place
        correction absorbs); grasp_pose ranks aligned directions (within KNIGHT_AXIS_TOL_DEG) before the others at
        equal clearance tier, so a crowded square (b8/g8 in the start position: the finger roots would sweep the
        rook and bishop) still gets the free direction."""
        k = self.kin
        u = radial_lat(np.asarray(xyz, float), 1.0, k.pan_axis_xy)
        base = math.degrees(math.atan2(u[1], u[0]))
        psi_max = min(math.degrees(k.hi[4]), -math.degrees(k.lo[4])) - k.roll_offset_deg - 1.0     # ~ +-156 deg
        out: List[float] = []
        axis = self.piece_axis_dirs(piece)
        yaw_b = self.scene.board_yaw_deg            # board-aligned = along the (possibly rotated) ranks / files
        grid = (axis or []) + [(d + yaw_b - base + 180.0) % 360.0 - 180.0 for d in (0.0, 90.0, 180.0, -90.0)] + list(LAT_CANDIDATES_DEG)
        for ang in grid:
            if not k.psi_reachable(ang):
                # a board-aligned direction in the roll dead zone is clipped to the nearest reachable one
                ang = max(-psi_max, min(psi_max, ang))
                if not k.psi_reachable(ang):
                    continue
            if any(abs((ang - a + 180.0) % 360.0 - 180.0) < 4.0 for a in out):
                continue
            out.append(ang)
        return out

    def grasp_pose(self, piece_xyz, ptype: str, exclude: Iterable[str], open_q: Optional[float] = None,
                   lat_pref: Optional[float] = None, label: str = "", roll_from: Optional[float] = None,
                   all_candidates: bool = False, max_lean: Optional[float] = None, min_good: float = CLEARANCE_GOOD,
                   min_clr: float = CLEARANCE_MIN, exhaustive: bool = False, hang: Optional[float] = None,
                   target: Optional[str] = None, tilt_pref: Optional[float] = None, tool=None,
                   target_min: float = TARGET_CLEARANCE_MIN, piece: Optional[str] = None, _lean_floor: float = -1.0):
        """Top-down pose whose stem point is on the piece axis at the grasp height, with the OPEN jaws (finger
        boxes) >= CLEARANCE_MIN from every other piece's collision cylinders and, when `target` names the piece
        standing there, >= TARGET_CLEARANCE_MIN from the target's own stem (a lean along the opening axis sweeps
        the finger root into it).  Candidates: jaw lean GRASP_TILT_FALLBACK (<= max_lean, default
        grasp_max_lean(type)) x grasp-height offsets config.grasp_dz_candidates (GRASP_DZ_FALLBACK for 'cylinders';
        inside the grasp zone for 'bands', see config) x the opening directions LAT_CANDIDATES_DEG
        (relative to the radial direction) x jaw openings (opening_candidates, widest first).  Leans are searched
        nearest to tilt_pref first (default 0 = smallest lean; a RELEASE passes the grasp lean so the rigidly held
        piece lands upright) and the search stops at the first lean level that has a candidate with clearance >=
        min_good (CLEARANCE_GOOD, robust to the piece jitter); otherwise (or with exhaustive=True, used for pieces
        that need a side approach: the best grasp may have no free side) every level is searched and the best
        clearance >= min_clr wins.  Ranking: good clearance, lean nearest tilt_pref, widest opening, then
        clearance / least wrist-roll travel from roll_from (lat_pref bonus).  `tool` overrides the tool point
        (default grasp_tool_site; a release uses place_tool_site).  `piece` (the piece being grasped / carried)
        restricts the opening directions to its grasp axis when it has one (lat_candidates).  Returns (IKResult,
        lat_deg, clearance, open_q_used), or the whole ranked list with all_candidates=True."""
        k = self.kin
        tool_fixed = None if tool is None else np.asarray(tool, float)   # None: the grasp tool point, per direction
        if roll_from is None:
            roll_from = float(C.lerobot_to_qpos(self.scene.get_joints())[4])
        if max_lean is None:
            max_lean = grasp_max_lean(ptype)
        # far-reach tier (GRASP_MAX_LEAN_FAR) only for the default cap and only when nothing within it exists
        far_tier = (_lean_floor < 0 and abs(max_lean - grasp_max_lean(ptype)) < 1e-9 and GRASP_MAX_LEAN_FAR > max_lean
                    and float(np.hypot(*(np.asarray(piece_xyz, float)[:2] - k.pan_axis_xy))) > FAR_TIER_MIN_RADIUS)
        pref = 0.0 if tilt_pref is None else float(tilt_pref)
        oqs = self.opening_candidates(ptype, open_q)
        tilts = sorted([t for t in GRASP_TILT_FALLBACK if _lean_floor + 1e-9 < abs(t) <= max_lean + 1e-9],
                       key=lambda t: (round(abs(t - pref), 3), abs(t)))
        angs = self.lat_candidates(piece_xyz, piece)
        axis = self.piece_axis_dirs(piece)

        def axis_tier(ang):
            if axis is None:
                return 0
            return 0 if min(abs((ang - ax + 180.0) % 360.0 - 180.0) for ax in axis) <= KNIGHT_AXIS_TOL_DEG else 1
        best_bad = None
        found = []
        for tilt in tilts:
            level = []
            for dz in C.grasp_dz_candidates(ptype, self.collision):
                target_xyz = np.array([piece_xyz[0], piece_xyz[1], piece_xyz[2] + (k.grasp_height(ptype) if hang is None else hang) + dz])
                for ang in angs:
                    lat = rotate_lat(radial_lat(target_xyz, 1.0, k.pan_axis_xy), ang)
                    tool = tool_fixed if tool_fixed is not None else self.grasp_tool(ptype, piece, ang)
                    r = k.solve(target_xyz, lat, tool, tilt_deg=tilt)
                    if not r.converged or r.pos_err > 0.002 or abs(r.tilt_deg) > max_lean + LEAN_IK_TOL_DEG:
                        continue
                    for oq in oqs:
                        # the moving jaw / camera module can touch the shoulder with the arm folded over the base:
                        # a narrower opening is often free where the wide one collides, so check every opening
                        sc = self.arm_self_collision(r.q5, oq)
                        if sc:
                            if best_bad is None:
                                best_bad = (r, ang, -1.0, f"the arm itself ({sc[0][0]}/{sc[0][1]}, {sc[0][2]*1000:.1f} mm)")
                            continue
                        clr, who = self.jaw_piece_clearance(r.q5, oq, exclude, target_xyz[:2])
                        if clr >= min_clr and target is not None:
                            clr_t, _ = self.jaw_piece_clearance(r.q5, oq, (), target_xyz[:2], only=target)
                            if clr_t < target_min:
                                if best_bad is None or best_bad[3].endswith(")") and clr_t > best_bad[2]:
                                    best_bad = (r, ang, clr_t, f"{target} itself (lean {r.tilt_deg:+.1f} deg, opening direction {ang:+.0f} deg, "
                                                               f"minimum {target_min*1000:.1f} mm)")
                                continue
                        if clr >= min_clr:
                            gap = C.gripper_gap(oq)
                            roll_travel = abs(math.degrees(r.q5[4] - roll_from))
                            # a grasp height outside the 'bands' grasp zone (config.GRASP_DZ_LAST_RESORT) only when no
                            # in-zone candidate clears the neighbours: the pads would pinch a single feature
                            cost = (0 if clr >= min_good else 1, axis_tier(ang), C.grasp_dz_tier(ptype, dz, self.collision),
                                    round(abs(tilt - pref), 1), -round(gap, 3),
                                    -min(clr, 0.008) * 1000 + roll_travel / 45.0 - (1.0 if ang == lat_pref else 0.0))
                            level.append((cost, r, ang, clr, oq))
                            if clr >= min_good:
                                break               # a narrower opening cannot be better for the stem
                        elif best_bad is None or clr > best_bad[2]:
                            best_bad = (r, ang, clr, who)
            found += level
            if not exhaustive and any(c[0][0] == 0 for c in level):
                break
        if found:
            found.sort(key=lambda f: f[0])
            if all_candidates:
                return [(r, ang, clr, oq) for _, r, ang, clr, oq in found]
            _, r, ang, clr, oq = found[0]
            return r, ang, clr, oq
        if far_tier:
            try:
                return self.grasp_pose(piece_xyz, ptype, exclude, open_q=open_q, lat_pref=lat_pref, label=label, roll_from=roll_from,
                                       all_candidates=all_candidates, max_lean=GRASP_MAX_LEAN_FAR, min_good=min_good,
                                       min_clr=min_clr, exhaustive=exhaustive, hang=hang, target=target, tilt_pref=tilt_pref,
                                       tool=tool, target_min=target_min, piece=piece, _lean_floor=max_lean)
            except PlanError:
                pass
        if best_bad is None:
            raise PlanError(f"{label}: no reachable vertical-jaw pose at {np.round(piece_xyz, 3)}")
        r, ang, clr, who = best_bad
        raise PlanError(f"{label}: open jaws would hit {who} for every opening direction / lean / height / opening (best clearance {clr*1000:.1f} mm)")

    # ------------------------------------------------------------------ paths
    def obstacle_floor(self, xy, hang: float, exclude: Iterable[str] = (), sweep_r: float = SWEEP_R_CARRY,
                       finger_r: float = 0.0) -> Tuple[float, str]:
        """Lowest admissible tool z over a point: every piece (except `exclude`; pieces moved by earlier plan
        steps at their planned spots) whose base lies within sweep_r + its base radius of xy must stay
        OBSTACLE_MARGIN below the carried piece's base (hang = grasp height of the carried piece, 0 for empty
        jaws).  finger_r > 0 adds the FINGERS: pieces within finger_r + their radius must stay below the
        fingertips (tool z - TIP_DROP).  Returns (z_floor, limiting piece)."""
        s = self.scene
        ex = set(exclude)
        z_floor, who = 0.0, ""
        for n in s.piece_names:
            if n in ex:
                continue
            p = self._planned_pos.get(n)
            if p is None:
                p = s.piece_pos(n)
            pt = s.piece_type[n]
            dist = np.hypot(p[0] - xy[0], p[1] - xy[1])
            z = 0.0
            # the piece's envelope radius as a function of height (the real profile for 'bands': the highest
            # section whose radius reaches the sweep disc is the obstacle height; base / stem for 'cylinders')
            top = C.piece_height_within(pt, dist - sweep_r, self.collision)
            if top > 0.0:
                z = p[2] + top + hang + OBSTACLE_MARGIN
            if finger_r > 0:
                top_f = C.piece_height_within(pt, dist - finger_r, self.collision)
                if top_f > 0.0:
                    z = max(z, p[2] + top_f + TIP_DROP + OBSTACLE_MARGIN)
            if z > z_floor:
                z_floor, who = z, n
        return z_floor, who

    @staticmethod
    def finger_radius(gripper_pct: float) -> float:
        """Radius of the disc around the tool point that the finger boxes can occupy at a jaw opening (the
        moving finger is gap away along the opening axis, whose direction depends on the wrist roll)."""
        return C.gripper_gap_for_percent(gripper_pct) / 2.0 + 0.007

    def grid_route(self, start_xy, goal_xy, floor, cell: float = 0.014) -> List[np.ndarray]:
        """A* over a grid of `cell` m covering the board, drop zone and spare cells: a cell is passable when
        floor(xy)[0] <= lift_z_at(xy) (the tool can travel over it at the local reach ceiling or lift height).
        Returns the simplified polyline (line-of-sight) from start to goal as xy points (goal included, start
        excluded), or raises PlanError."""
        import heapq
        cfg = self.scene.cfg
        # the grid covers the nominal board, drop zone and spare cells plus the largest board / pad displacement
        # (the board pose is randomised: offset + yaw, the pads jitter on their own)
        corners = self.scene.board_corners()[:, :2]
        need_x0 = min(cfg.board_gap + C.DROP_X_SHIFT, corners[:, 0].min()) - 0.045
        need_x1 = max(cfg.board_gap + C.BOARD_SIZE, corners[:, 0].max()) + 0.045
        need_y0 = min(-(C.BOARD_SIZE / 2 + C.DROP_GAP + C.DROP_CELLS * C.SQUARE), corners[:, 1].min()) - 0.03
        need_y1 = max(C.BOARD_SIZE / 2 + C.DROP_GAP + C.SQUARE, corners[:, 1].max()) + 0.03
        # the lattice stays anchored where the first build had it (x = board_gap - 45 mm, y = the old drop-zone edge
        # - 30 mm) and is only EXTENDED by whole cells: the A* routes on the nominal layout keep their knots
        ax, ay = cfg.board_gap - 0.045, -(C.BOARD_SIZE / 2 + 0.010 + C.DROP_CELLS * C.SQUARE + 0.03)
        x0 = ax - cell * max(0, math.ceil((ax - need_x0) / cell - 1e-9))
        y0 = ay - cell * max(0, math.ceil((ay - need_y0) / cell - 1e-9))
        x1, y1 = max(need_x1, x0), max(need_y1, y0)
        nx, ny = int(math.ceil((x1 - x0) / cell)) + 1, int(math.ceil((y1 - y0) / cell)) + 1

        def xy(i, j):
            return np.array([x0 + i * cell, y0 + j * cell])

        def idx(p):
            return (int(round((p[0] - x0) / cell)), int(round((p[1] - y0) / cell)))
        passable: Dict[Tuple[int, int], bool] = {}

        def ok(i, j):
            if not (0 <= i < nx and 0 <= j < ny):
                return False
            k = (i, j)
            if k not in passable:
                p = xy(i, j)
                zc = self.lift_z_at(p)
                passable[k] = zc > 0.02 and floor(p)[0] <= zc + 1e-6
            return passable[k]
        s0, g0 = idx(start_xy), idx(goal_xy)
        passable[s0] = passable[g0] = True
        h = lambda a: math.hypot(a[0] - g0[0], a[1] - g0[1])
        openq = [(h(s0), 0.0, s0)]
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cost = {s0: 0.0}
        found = False
        while openq:
            f, gc, cur = heapq.heappop(openq)
            if cur == g0:
                found = True
                break
            if gc > cost.get(cur, np.inf):
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    nb = (cur[0] + di, cur[1] + dj)
                    if not ok(*nb):
                        continue
                    if di and dj and not (ok(cur[0] + di, cur[1]) and ok(cur[0], cur[1] + dj)):
                        continue            # no corner cutting
                    c2 = gc + math.hypot(di, dj)
                    if c2 < cost.get(nb, np.inf):
                        cost[nb] = c2
                        came[nb] = cur
                        heapq.heappush(openq, (c2 + h(nb), c2, nb))
        if not found:
            raise PlanError(f"no corridor over the board from {np.round(start_xy, 3)} to {np.round(goal_xy, 3)} at the reach ceiling")
        cells = [g0]
        while cells[-1] != s0:
            cells.append(came[cells[-1]])
        cells.reverse()
        pts = [np.asarray(start_xy, float)[:2]] + [xy(*c) for c in cells[1:-1]] + [np.asarray(goal_xy, float)[:2]]

        def clear(a, b):
            n = max(2, int(math.ceil(np.linalg.norm(b - a) / (cell / 2))))
            for t in range(n + 1):
                p = a + (b - a) * t / n
                if not ok(*idx(p)):
                    return False
            return True
        out = []
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def cart_path(self, q_start5, p1, lat1, tilt1: float, tool, gripper_pct: float, q_end5=None,
                  max_tilt: float = TRAVEL_TILT_DEG, spacing: float = KNOT_SPACING, label: str = "",
                  clamp_z: bool = False, floor=None, fingers: Optional[Iterable[str]] = None,
                  fingers_min: float = CLEARANCE_PATH, target: Optional[str] = None,
                  target_min: float = TARGET_CLEARANCE_MIN, carried: Optional[Tuple[str, float]] = None) -> Tuple[List[np.ndarray], float]:
        """Joint knots along the straight tool-point line from fk(q_start5) to p1, with the opening direction
        interpolated (wrist roll kept inside its range) and the lean interpolated from the start pose's to tilt1
        (each knot: the lean nearest the interpolated value that converges without arm self-collision, <= max_tilt;
        the final knot is exact / q_end5).  clamp_z=True lowers the line to the reach ceiling (lift_z_at) where the
        nominal height is unreachable (transits near the base); floor(xy) -> (z, who) raises the line over the
        pieces under the corridor and fails (PlanError) when the ceiling is below that floor.  fingers (a list of
        pieces to exclude, e.g. the carried one) enables the exact check at every knot: finger boxes >= fingers_min
        from every other piece; `target` (a piece standing at the start or end of the leg: the one being grasped,
        or the one just released, at its planned spot) adds the check finger boxes >= target_min from that piece;
        carried=(piece, hang) adds the check carried piece >= CARRY_CLEARANCE from every other piece
        (carried_piece_clearance).  Every knot is warm-started from the previous one.  Returns (knots as LeRobot
        q6, path length)."""
        k = self.kin
        q0 = np.asarray(q_start5, float)
        p0, R0, t0 = k.fk(q0, tool)
        u0 = k.radial_dir(t0, R0[:, 2], tool, 0.0)
        tilt0, psi0 = frame_params(R0, u0)
        p1 = np.asarray(p1, float)
        u1 = k.radial_dir(p1, lat1, tool, tilt1)
        psi1 = math.degrees(math.atan2(float(np.cross(np.array([0, 0, 1.0]), u1) @ lat1), float(u1 @ lat1)))
        # the wrist roll is psi + const (~2.8 deg): pick the representative of psi1 whose roll lies inside the
        # roll joint range and is nearest the start roll, and interpolate WITHOUT wrapping (the roll range
        # -157..+163 deg has a 40 deg dead zone around 180 that a shortest-angle interpolation would cross)
        roll0 = math.degrees(q0[4])
        off = roll0 - psi0
        lo, hi = math.degrees(k.lo[4]), math.degrees(k.hi[4])
        cands = [psi1 + off + 360.0 * j for j in (-1, 0, 1)]
        inside = [c for c in cands if lo - 1.0 <= c <= hi + 1.0]
        roll1 = min(inside or cands, key=lambda c: abs(c - roll0))
        dpsi = roll1 - roll0
        L = float(np.linalg.norm(p1 - t0))
        n = max(2, int(math.ceil(L / spacing)), int(math.ceil(abs(dpsi) / 20.0)), int(math.ceil(abs(tilt1 - tilt0) / 5.0)))
        knots = []
        q_prev = q0.copy()
        gq = C.gripper_percent_to_frac(gripper_pct)
        fx = None if fingers is None else list(fingers)
        for i in range(1, n + 1):
            s = i / n
            p = t0 + (p1 - t0) * s
            if i < n:
                if clamp_z:
                    p[2] = min(p[2], self.lift_z_at(p[:2]))
                if floor is not None:
                    zf, who = floor(p[:2])
                    if zf > p[2] + 1e-6:
                        if clamp_z and zf > self.lift_z_at(p[:2]) + 1e-6:
                            raise PlanError(f"{label}: cannot pass over {who} at {np.round(p[:2], 3)} (needs z {zf:.3f}, "
                                            f"reach ceiling {self.lift_z_at(p[:2]):.3f})", where=p.copy())
                        p[2] = zf
            psi = psi0 + dpsi * s
            u = radial_lat(p, 1.0, k.pan_axis_xy)
            lat = rotate_lat(u, psi)
            if i == n and q_end5 is not None:
                q = np.asarray(q_end5, float)
            elif i == n:
                r = self._knot(p1, lat1, tool, gripper_pct, max_tilt, q_prev, tilt1)
                if r is None:
                    rr = k.solve_reach(p1, lat1, tool, max_tilt_deg=max_tilt, tilt_step=2.5, q_init=q_prev, tilt_hint=tilt1)
                    sc = self.arm_self_collision(rr.q5, gq) if rr.converged else []
                    raise PlanError(f"{label}: end pose unreachable at {np.round(p1, 3)} (pos err {rr.pos_err*1000:.1f} mm, "
                                    f"rot err {math.degrees(rr.rot_err):.1f} deg, limits {rr.at_limit}"
                                    + (f", self-collision {sc[0][0]}/{sc[0][1]}" if sc else "") + ")")
                q = r.q5
            else:
                hint = tilt0 + (tilt1 - tilt0) * s
                r = self._knot(p, lat, tool, gripper_pct, max_tilt, q_prev, hint)
                if r is None and clamp_z:
                    for dz in (0.005, 0.010, 0.015, 0.020):
                        p2 = p.copy()
                        p2[2] -= dz
                        if floor is not None and p2[2] < floor(p2[:2])[0] - 1e-6:
                            break
                        r = self._knot(p2, lat, tool, gripper_pct, max_tilt, q_prev, hint)
                        if r is not None:
                            p = p2
                            break
                if r is None:
                    raise PlanError(f"{label}: path knot {i}/{n} unreachable or self-colliding at {np.round(p, 3)}", where=p.copy())
                q = r.q5
            if fx is not None:
                clr, who = self.jaw_piece_clearance(q, gq, fx, p[:2])
                if clr < fingers_min:
                    raise PlanError(f"{label}: fingers would hit {who} at knot {i}/{n} {np.round(p, 3)} (clearance {clr*1000:.1f} mm)",
                                    where=p.copy())
            if target is not None:
                clr, _ = self.jaw_piece_clearance(q, gq, (), p[:2], only=target)
                if clr < target_min:
                    raise PlanError(f"{label}: fingers would hit the target {target} at knot {i}/{n} {np.round(p, 3)} "
                                    f"(clearance {clr*1000:.1f} mm)", where=p.copy())
            if carried is not None:
                clr, who = self.carried_piece_clearance(q, gq, carried[0], carried[1], tool)
                if clr < CARRY_CLEARANCE:
                    raise PlanError(f"{label}: the carried {carried[0]} would hit {who} at knot {i}/{n} {np.round(p, 3)} "
                                    f"(clearance {clr*1000:.1f} mm)", where=p.copy())
            knots.append(np.concatenate([C.rad2deg(q), [gripper_pct]]))
            q_prev = q
        return knots, L

    def _knot(self, p, lat, tool, gripper_pct: float, max_tilt: float, q_prev, hint: float) -> Optional[IKResult]:
        """IK for a path knot: the lean nearest `hint` that converges AND is free of arm self-collision.  The exact
        hint (the continuously interpolated lean) is tried first: quantising the lean to the 2.5 deg grid made the
        elbow zig-zag between neighbouring knots, and the joint-space interpolation between them then swept the
        fingertips millimetres off the planned line (a king was brushed over during its own grasp descent)."""
        k = self.kin
        gq = C.gripper_percent_to_frac(gripper_pct)
        grid = [0.0]
        j = 1
        while j * 2.5 < max_tilt - 1e-9:
            grid += [j * 2.5, -j * 2.5]
            j += 1
        if max_tilt > 1e-9:
            grid += [max_tilt, -max_tilt]
        grid.sort(key=lambda t: round(abs(t - hint), 6))
        h = max(-max_tilt, min(max_tilt, float(hint)))
        if all(abs(h - t) > 1e-6 for t in grid):
            grid.insert(0, h)
        for tilt in grid:
            r = k.solve(p, lat, tool, tilt_deg=tilt, q_init=q_prev, seeds=False)
            if not r.converged:
                continue
            if self.arm_self_collision(r.q5, gq):
                continue
            return r
        return None

    def _above_pose(self, xy, z_wanted: float, z_min: float, lat, tool, gripper_pct: float, q_hint, tilt_hint: float,
                    floor_z: float, label: str) -> Tuple[IKResult, float]:
        """Pose above a point for the lift/carry: the highest reachable, self-collision-free z among the wanted
        height, the obstacle floor and a few fallbacks down to z_min (never below the floor)."""
        cands = sorted({z_wanted, floor_z, z_min + 0.010, z_min + 0.003, z_min}, reverse=True)
        for z in cands:
            if z < z_min - 1e-6 or z < floor_z - 1e-6:
                continue
            rr = self._knot(np.array([xy[0], xy[1], z]), lat, tool, gripper_pct, TRAVEL_TILT_DEG, q_hint, tilt_hint)
            if rr is not None:
                return rr, z
        raise PlanError(f"{label}: no reachable pose above {np.round(xy, 3)} between z {z_min:.3f} and {z_wanted:.3f} (floor {floor_z:.3f})")

    def _pre_height(self, xy, z_grasp: float, z_above: float, lat, tool, gripper_pct: float, tilt: float, q_hint) -> Optional[float]:
        """Highest z (5 mm grid, >= z_grasp + PRE_DESCENT_MIN) up to z_above at which the GRASP orientation
        (lat, lean=tilt) is reachable without self-collision: the final descent below it keeps the orientation
        constant so the finger root never sweeps sideways near the neighbours."""
        k = self.kin
        gq = C.gripper_percent_to_frac(gripper_pct)
        z = z_above
        while z > z_grasp + PRE_DESCENT_MIN + 1e-6:
            r = k.solve(np.array([xy[0], xy[1], z]), lat, tool, tilt_deg=tilt, q_init=q_hint, seeds=False)
            if r.converged and not self.arm_self_collision(r.q5, gq):
                return z
            z -= 0.005
        return None

    def _descend(self, plan: MovePlan, q_from5, grasp: IKResult, lat, tool, gripper_pct: float, xy, z_above: float,
                 kind: str, label: str, meta: dict, fingers: Iterable[str], fingers_min: float = CLEARANCE_PATH,
                 target: Optional[str] = None, carried: Optional[Tuple[str, float]] = None) -> None:
        """Two-stage descent: approach (lean settling to the grasp lean) down to the pre-height, then a slow
        constant-orientation descent onto the grasp/place pose.  `target` (the piece being grasped) is included in
        the finger check of both legs."""
        k = self.kin
        g_z = float(k.fk(grasp.q5, tool)[2][2])
        pre_z = self._pre_height(xy, g_z, z_above, lat, tool, gripper_pct, grasp.tilt_deg, grasp.q5)
        q_mid = q_from5
        if pre_z is None:
            pre_z = z_above                   # far corners: the grasp lean is only reachable at the grasp height itself
        if pre_z < z_above - 1e-6:
            r_pre = k.solve(np.array([xy[0], xy[1], pre_z]), lat, tool, tilt_deg=grasp.tilt_deg, q_init=grasp.q5)
            if not r_pre.converged:
                raise PlanError(f"{label}: pre-descent pose unreachable")
            knots, L = self.cart_path(q_from5, np.array([xy[0], xy[1], pre_z]), lat, grasp.tilt_deg, tool, gripper_pct,
                                      q_end5=r_pre.q5, spacing=DESCENT_KNOT, label=f"approach {label}", fingers=fingers, fingers_min=fingers_min,
                                      target=target, carried=carried)
            plan.append(Waypoint(knots[-1], "approach", f"approach {label}", via=knots[:-1], dwell=0.0, tool_path_len=L, meta=dict(meta)))
            q_mid = r_pre.q5
        knots, L = self.cart_path(q_mid, k.fk(grasp.q5, tool)[2], lat, grasp.tilt_deg, tool, gripper_pct, q_end5=grasp.q5,
                                  spacing=DESCENT_KNOT, label=f"descend {label}", fingers=fingers, fingers_min=fingers_min, target=target,
                                  carried=carried)
        plan.append(Waypoint(knots[-1], kind, label, via=knots[:-1], dwell=SETTLE["grasp" if kind == "descend_grasp" else "place"],
                             tool_path_len=L, meta=dict(meta, slow=True)))

    def _centre_shift(self, plan: MovePlan, piece: str, ptype: str, pl: IKResult, lat, tool, open_q: float, label: str) -> Tuple[np.ndarray, np.ndarray]:
        """After a release: shift the OPEN jaws along the opening axis (toward the fixed-jaw side, -z) so the stem of
        the released piece sits centred between the fingers (both >= ~CENTRE_SHIFT_MAX away instead of 1.5 mm from the
        fixed finger).  A later lean/roll change or sideways retreat then cannot sweep a finger into the piece,
        and a piece resting against the fixed finger after the release can settle upright.  Returns (q5, tool
        point) at the end (unchanged when no shift is possible)."""
        k = self.kin
        open_pct = C.gripper_frac_to_percent(open_q)
        p0, R0, t0 = k.fk(pl.q5, tool)
        # room between the open fingers and the piece: its extents toward the fixed finger and toward the moving one
        th = self._held_theta.get(piece)
        e_f = C.piece_side_extent(ptype, th, above=True, collision=self.collision)
        e_m = C.piece_side_extent(ptype, None if th is None else th + 180.0, above=True, collision=self.collision)
        d_max = (C.gripper_clearance(open_q) - e_f - e_m) / 2.0 - GRASP_SIDE_CLEARANCE
        for d in (min(CENTRE_SHIFT_MAX, d_max), min(CENTRE_SHIFT_MAX, d_max) / 2.0):
            if d < 0.002:
                break
            p1 = t0 - R0[:, 2] * d          # the stem is on the +z (moving-jaw) side of the fixed finger: move the fixed finger away
            try:
                r1 = k.solve(p1, lat, tool, tilt_deg=pl.tilt_deg, q_init=pl.q5)
                if not r1.converged or r1.pos_err > 0.001 or self.arm_self_collision(r1.q5, open_q):
                    continue
                knots, L = self.cart_path(pl.q5, p1, lat, pl.tilt_deg, tool, open_pct, q_end5=r1.q5, spacing=0.004,
                                          label=f"centre jaws on {piece}", fingers=[piece], fingers_min=CLEARANCE_MIN_SLIDE, target=piece)
            except PlanError:
                continue
            plan.append(Waypoint(knots[-1], "retreat", f"centre jaws on {piece}", via=knots[:-1], dwell=0.0, tool_path_len=L,
                                 meta=dict(slow=True, released=piece)))
            return r1.q5, p1
        return np.asarray(pl.q5, float), t0

    def _lift(self, plan: MovePlan, pp: PickPlace, g: IKResult, lat, tool, above_xyz, q_above5, tilt_above: float, hang: float,
              z_src: float) -> None:
        """Lift after the close, symmetric to the descent: a slow constant-orientation vertical leg up to the
        pre-height (the grasp orientation kept while the finger roots are beside the neighbours), then the climb
        to the lift height with the lean settling to the above pose's; fingers and the carried piece are checked
        at every knot."""
        k = self.kin
        g_z = float(k.fk(g.q5, tool)[2][2])
        xy = above_xyz[:2]
        meta = dict(piece=pp.piece, carrying=True, carried_offset=np.array([0, 0, -hang]), lift_z=z_src, slow=True, hang=hang)
        pre_z = self._pre_height(xy, g_z, float(above_xyz[2]), lat, tool, pp.close_pct, g.tilt_deg, g.q5)
        q_mid = g.q5
        if pre_z is not None and pre_z < above_xyz[2] - 1e-6:
            r_pre = k.solve(np.array([xy[0], xy[1], pre_z]), lat, tool, tilt_deg=g.tilt_deg, q_init=g.q5)
            if r_pre.converged and not self.arm_self_collision(r_pre.q5, C.gripper_percent_to_frac(pp.close_pct)):
                knots, L = self.cart_path(g.q5, np.array([xy[0], xy[1], pre_z]), lat, g.tilt_deg, tool, pp.close_pct, q_end5=r_pre.q5,
                                          spacing=DESCENT_KNOT, label=f"lift {pp.piece} (vertical)", fingers=[pp.piece], carried=(pp.piece, hang))
                plan.append(Waypoint(knots[-1], "lift", f"lift {pp.piece} (vertical)", via=knots[:-1], dwell=0.0, tool_path_len=L,
                                     meta=dict(meta, no_check=True)))
                q_mid = r_pre.q5
        knots, L = self.cart_path(q_mid, above_xyz, lat, tilt_above, tool, pp.close_pct, q_end5=q_above5, spacing=DESCENT_KNOT,
                                  label=f"lift {pp.piece}", fingers=[pp.piece], carried=(pp.piece, hang))
        plan.append(Waypoint(knots[-1], "lift", f"lift {pp.piece}", via=knots[:-1], dwell=SETTLE["lift"], tool_path_len=L, meta=meta))

    def _retreat(self, plan: MovePlan, piece: str, q_from5, lat, tilt: float, tool, open_pct: float, z_wanted: float, z_min: float,
                 label: str, target_min: float = TARGET_CLEARANCE_MIN) -> np.ndarray:
        """Post-release climb above a released piece: a constant-orientation vertical leg up to the pre-height
        (where the release orientation is still reachable), then the climb to the lift height with the lean
        settling to the above pose's; the released piece (at its planned spot) is included in the finger checks.
        Returns the arm joints at the top."""
        k = self.kin
        p0, R0, t0 = k.fk(q_from5, tool)
        xy = t0[:2]
        fz = self.obstacle_floor(xy, 0.0, [], SWEEP_R_EMPTY)[0]
        r_above, z_top = self._above_pose(xy, z_wanted, z_min, lat, tool, open_pct, q_from5, tilt, fz, label)
        pre_z = self._pre_height(xy, float(t0[2]), z_top, lat, tool, open_pct, tilt, q_from5)
        q_mid = np.asarray(q_from5, float)
        if pre_z is not None and pre_z < z_top - 1e-6:
            r_pre = k.solve(np.array([xy[0], xy[1], pre_z]), lat, tool, tilt_deg=tilt, q_init=q_from5)
            if r_pre.converged:
                knots, L = self.cart_path(q_from5, np.array([xy[0], xy[1], pre_z]), lat, tilt, tool, open_pct, q_end5=r_pre.q5,
                                          spacing=DESCENT_KNOT, label=f"{label} (vertical)", fingers=[piece], target=piece, target_min=target_min)
                plan.append(Waypoint(knots[-1], "retreat", f"{label} (vertical)", via=knots[:-1], dwell=0.0, tool_path_len=L,
                                     meta=dict(slow=True, released=piece)))
                q_mid = r_pre.q5
        knots, L = self.cart_path(q_mid, np.array([xy[0], xy[1], z_top]), lat, r_above.tilt_deg, tool, open_pct, q_end5=r_above.q5,
                                  spacing=DESCENT_KNOT, label=label, fingers=[piece], target=piece, target_min=target_min)
        plan.append(Waypoint(knots[-1], "lift", label, via=knots[:-1], dwell=SETTLE["lift"], tool_path_len=L,
                             meta=dict(lift_check=True, lift_z=z_top, released=piece, slow=True)))
        return r_above.q5

    def _solve_near_lat(self, p, lat, tool, tilt: float, q_init, max_rot: float = 10.0):
        """IK at p with the opening direction lat, or lat rotated by up to +-max_rot deg (2.5 deg steps) when the
        exact one does not converge (wrist-roll limit).  Along the 25 mm slide the roll interpolates back to the
        exact grasp direction; the stem is between the fingers only during the last ~8 mm, where the residual
        rotation is < 3.5 deg (0.5 mm lateral).  Returns (IKResult or None, lat used)."""
        k = self.kin
        for rot in (0.0, 2.5, -2.5, 5.0, -5.0, 7.5, -7.5, 10.0, -10.0):
            if abs(rot) > max_rot + 1e-9:
                continue
            l2 = rotate_lat(lat, rot)
            r = k.solve(p, l2, tool, tilt_deg=tilt, q_init=q_init)
            if r.converged and r.pos_err <= 0.002:
                return r, l2
        return None, lat

    def landing_tilt_deg(self, grasp: IKResult, place: IKResult, ptype: str) -> float:
        """Tilt (deg) a rigidly held piece lands with when grasped upright at `grasp` and released at `place`:
        the angle between the piece axis R_place R_grasp^T z and the vertical."""
        k = self.kin
        _, Rg, _ = k.fk(grasp.q5, k.grasp_tool_site(ptype))
        _, Rp, _ = k.fk(place.q5, k.place_tool_site(ptype))
        axis = Rp @ (Rg.T @ np.array([0.0, 0.0, 1.0]))
        return float(math.degrees(math.acos(max(-1.0, min(1.0, axis[2] / (np.linalg.norm(axis) + 1e-12))))))

    def rank_grasps_for_release(self, cands, dst, ptype: str, exclude: Sequence[str], open_q: float, hang: float, tool_p,
                                piece: str):
        """Stable re-ranking of grasp candidates (r, ang, clr, oq) of a piece that will be lifted to `dst`: the
        grasps for which some release candidate at dst lands the piece within LANDING_TILT_MAX first.  The release
        candidates are probed once (every lean level), independent of the grasp; when no grasp is compatible (or
        no release pose exists) the order is unchanged and the normal planning reports the problem."""
        if len(cands) < 2:
            return cands
        self._planned_pos[piece] = np.array([dst[0], dst[1], dst[2] - PLACE_EXTRA_Z])
        try:
            probe = self.grasp_pose(dst, ptype, exclude=exclude, open_q=open_q, label=f"release probe {piece}",
                                    roll_from=float(cands[0][0].q5[4]), all_candidates=True, max_lean=grasp_max_lean(ptype),
                                    min_clr=CLEARANCE_MIN, exhaustive=True, hang=hang, tilt_pref=cands[0][0].tilt_deg, tool=tool_p,
                                    target=piece, target_min=RELEASE_CLEARANCE_MIN, piece=piece)
        except PlanError:
            return cands
        finally:
            self._planned_pos.pop(piece, None)
        compat = [any(self.landing_tilt_deg(g, r, ptype) <= LANDING_TILT_MAX for r, _, _, _ in probe) for g, _, _, _ in cands]
        if not any(compat) or all(compat[:GRASP_CANDIDATES_TRIED]):
            return cands
        return [c for c, ok in zip(cands, compat) if ok] + [c for c, ok in zip(cands, compat) if not ok]

    def rank_release_candidates(self, cands, grasp: IKResult, ptype: str):
        """Stable re-ranking of release candidates (r, ang, clr, oq): those whose predicted landing tilt is
        <= LANDING_TILT_MAX first (the piece lands upright), the others after, each tier in its original order."""
        return sorted(cands, key=lambda c: self.landing_tilt_deg(grasp, c[0], ptype) > LANDING_TILT_MAX)

    def _side_options(self, cands, piece: str, ptype: str, piece_xyz, tool, label: str, why: Optional[list] = None,
                      min_clr: float = CLEARANCE_MIN_SLIDE, tilt_pref: float = 0.0, tool_fn=None) -> list:
        """For ranked grasp/release candidates (r, ang, clr, oq) of a piece whose top the jaws cannot clear: the
        side points SIDE_APPROACH beside the piece (perpendicular to the opening axis, either side) where the
        same pose is reachable, self-collision free, above nothing, and the fingers stay >= CLEARANCE_MIN from
        every other piece at the side point and half way.  Returns [(score, sign, pre_xy, r_pre, g, ang, clr, lat,
        g_z, g_pt, oq)] sorted best first (clearance, then the side farther from the base)."""
        k = self.kin
        options = []
        tool_given = tool
        for g, g_ang, g_clr, oq in cands:
            tool = tool_fn(g_ang) if tool_fn is not None else tool_given    # the grasp tool point depends on the direction (knight)
            g_lat = rotate_lat(radial_lat(np.array([piece_xyz[0], piece_xyz[1], 0]), 1.0, k.pan_axis_xy), g_ang)
            g_pt = k.fk(g.q5, tool)[2]
            g_z = float(g_pt[2])
            side = np.cross(np.array([0.0, 0.0, 1.0]), g_lat)
            side /= np.linalg.norm(side) + 1e-12
            for sign, dist in [(sg, dd) for dd in SIDE_APPROACH_ALT for sg in (1.0, -1.0)]:
                pre_xy = g_pt[:2] + sign * side[:2] * dist
                tag = f"lat {g_ang:+.0f} tilt {g.tilt_deg:+.1f} gap {C.gripper_gap(oq)*1000:.0f} side {sign:+.0f} {dist*1000:.0f} mm {np.round(pre_xy, 3)}"
                fz, who = self.obstacle_floor(pre_xy, 0.0, [piece], 0.004)
                if fz > g_z - 1e-6:
                    if why is not None:
                        why.append(f"{tag}: {who} stands there")
                    continue
                # the opening direction is a world direction: 25 mm away the roll it needs differs by a few degrees
                # (radial direction), which can cross the wrist-roll limit, so a slightly rotated direction is allowed
                r_pre, lat_pre = self._solve_near_lat(np.array([pre_xy[0], pre_xy[1], g_z]), g_lat, tool, g.tilt_deg, g.q5)
                if r_pre is None:
                    rr = k.solve(np.array([pre_xy[0], pre_xy[1], g_z]), g_lat, tool, tilt_deg=g.tilt_deg, q_init=g.q5)
                    if why is not None:
                        why.append(f"{tag}: unreachable ({rr.pos_err*1000:.1f} mm, {rr.at_limit})")
                    continue
                sc = self.arm_self_collision(r_pre.q5, oq)
                if sc:
                    if why is not None:
                        why.append(f"{tag}: self-collision {sc[0][0]}/{sc[0][1]}")
                    continue
                clr, who1 = self.jaw_piece_clearance(r_pre.q5, oq, [piece], pre_xy)
                mid, _ = self._solve_near_lat(np.array([(pre_xy[0] + g_pt[0]) / 2, (pre_xy[1] + g_pt[1]) / 2, g_z]), g_lat, tool, g.tilt_deg, g.q5)
                clr_mid, who2 = self.jaw_piece_clearance(mid.q5, oq, [piece], pre_xy) if mid is not None else (-1.0, "mid unreachable")
                c = min(clr, clr_mid, g_clr)
                if c < min_clr:
                    if why is not None:
                        why.append(f"{tag}: clearance pre {clr*1000:.1f} ({who1}) mid {clr_mid*1000:.1f} ({who2}) grasp {g_clr*1000:.1f}")
                    continue
                r_side = float(np.hypot(pre_xy[0] - k.pan_axis_xy[0], pre_xy[1] - k.pan_axis_xy[1]))
                options.append(((0 if c >= CLEARANCE_GOOD else 1, round(abs(g.tilt_deg - tilt_pref), 1), -round(min(c, 0.008), 3), -r_side),
                                sign, pre_xy, r_pre, g, g_ang, g_clr, g_lat, g_z, g_pt, oq))
        options.sort(key=lambda o: o[0])
        return options

    def _side_approach(self, plan: MovePlan, pp: PickPlace, q_cur5, cands, tool, floor_empty) -> None:
        """Pick approach for pieces whose top the jaws cannot clear: travel to above a point SIDE_APPROACH beside
        the piece (perpendicular to the opening axis, on the side free of pieces), descend there with the grasp
        orientation, then slide horizontally into the grasp pose (the stem passes between the open fingers)."""
        why: list = []
        options = self._side_options(cands, pp.piece, pp.ptype, pp.src_xyz, tool, f"grasp {pp.piece}", why,
                                     tool_fn=lambda a: self.grasp_tool(pp.ptype, pp.piece, a))
        if not options:
            raise PlanError(f"grasp {pp.piece}: no free side for a side approach (jaws cannot clear its top; {len(cands)} grasp "
                            f"candidates): " + "; ".join(why[:6]))
        last_err = None
        for _, sign, pre_xy, r_pre, g, g_ang, g_clr, g_lat, g_z, g_pt, oq in options[:8]:
            n0 = len(plan)
            tool = self.grasp_tool(pp.ptype, pp.piece, g_ang)
            try:
                open_pct = C.gripper_frac_to_percent(oq)
                fz_pre = self.obstacle_floor(pre_xy, 0.0, [pp.piece], SWEEP_R_EMPTY)[0]
                r_above, z_pre = self._above_pose(pre_xy, self.lift_z_at(pre_xy), g_z, g_lat, tool, open_pct, r_pre.q5, g.tilt_deg, fz_pre,
                                                  f"side pre-grasp {pp.piece}")
                above_pre = np.array([pre_xy[0], pre_xy[1], z_pre])
                self._transit(plan, q_cur5, above_pre, g_lat, r_above.tilt_deg, tool, TRAVEL_PCT, r_above.q5, floor_empty,
                              f"to {pp.piece} (side)", dict(lift_z=z_pre), kind="pregrasp", fingers=[])
                self._open_above(plan, r_above.q5, open_pct, [pp.piece], pre_xy, f"open beside {pp.piece}")
                self._descend(plan, r_above.q5, r_pre, g_lat, tool, open_pct, pre_xy, z_pre, "approach", f"beside {pp.piece}",
                              dict(piece=pp.piece), fingers=[pp.piece], fingers_min=SLIDE_KNOT_MIN, target=pp.piece)
                knots, L = self.cart_path(r_pre.q5, g_pt, g_lat, g.tilt_deg, tool, open_pct, q_end5=g.q5, spacing=0.005,
                                          label=f"slide {pp.piece}", fingers=[pp.piece], fingers_min=SLIDE_KNOT_MIN, target=pp.piece)
                plan.append(Waypoint(knots[-1], "descend_grasp", f"grasp {pp.piece} (slide in)", via=knots[:-1], dwell=SETTLE["grasp"],
                                     tool_path_len=L, meta=dict(piece=pp.piece, slow=True)))
                pp.grasp, pp.grasp_lat_deg, pp.grasp_clearance, pp.open_pct, pp.side_approach = g, g_ang, g_clr, open_pct, True
                return
            except PlanError as e:
                last_err = e
                del plan[n0:]
        raise PlanError(f"grasp {pp.piece}: side approach failed: {last_err}")

    def _side_retreat(self, plan: MovePlan, pp: PickPlace, pl: IKResult, p_ang: float, p_clr: float, p_oq: float, tool, dst) -> np.ndarray:
        """After the release of a piece whose top the jaws cannot clear: centre the open jaws on the stem, slide
        them out sideways at the place height (the stem leaves between the fingers; the released piece is in the
        finger check), then climb to the lift height there.  Returns the arm joints at the end (above the side
        point, jaws open)."""
        options = self._side_options([(pl, p_ang, p_clr, p_oq)], pp.piece, pp.ptype, dst, tool, f"retreat from {pp.dst_name}")
        if not options:
            raise PlanError(f"retreat from {pp.dst_name}: no free side to back away from {pp.piece}")
        last_err = None
        for (_, sign, pre_xy, r_pre, g, g_ang, g_clr, g_lat, g_z, g_pt, oq), centre in [(o, c) for o in options for c in (True, False)]:
            n0 = len(plan)
            try:
                open_pct = C.gripper_frac_to_percent(oq)
                if centre:
                    # centre the jaws on the stem first, then back away PARALLEL to the validated side direction
                    q_c, t_c = self._centre_shift(plan, pp.piece, pp.ptype, pl, g_lat, tool, oq, f"centre on {pp.piece}")
                    shift = t_c[:2] - g_pt[:2]
                    if np.linalg.norm(shift) < 1e-6:
                        continue                    # no shift possible: same as centre=False
                else:
                    q_c, shift = pl.q5, np.zeros(2)
                p_end = np.array([pre_xy[0] + shift[0], pre_xy[1] + shift[1], g_z])
                r_end = r_pre
                if centre:
                    r_end, _ = self._solve_near_lat(p_end, g_lat, tool, g.tilt_deg, r_pre.q5)
                    if r_end is None or self.arm_self_collision(r_end.q5, oq):
                        raise PlanError(f"back away from {pp.piece}: shifted side point unreachable")
                knots, L = self.cart_path(q_c, p_end, g_lat, g.tilt_deg, tool, open_pct, q_end5=r_end.q5,
                                          spacing=0.005, label=f"back away from {pp.piece}", fingers=[pp.piece], fingers_min=SLIDE_KNOT_MIN,
                                          target=pp.piece, target_min=TARGET_CLEARANCE_MIN if centre else RELEASE_CLEARANCE_MIN)
                plan.append(Waypoint(knots[-1], "retreat", f"back away from {pp.piece}", via=knots[:-1], dwell=SETTLE["retreat"],
                                     tool_path_len=L, meta=dict(slow=True, released=pp.piece)))
                xy_e = p_end[:2]
                fz_pre = self.obstacle_floor(xy_e, 0.0, [], SWEEP_R_EMPTY)[0]
                r_above, z_pre = self._above_pose(xy_e, self.lift_z_at(xy_e), g_z, g_lat, tool, open_pct, r_end.q5, g.tilt_deg, fz_pre,
                                                  f"retreat above {pp.dst_name}")
                knots, L = self.cart_path(r_end.q5, np.array([xy_e[0], xy_e[1], z_pre]), g_lat, r_above.tilt_deg, tool, open_pct,
                                          q_end5=r_above.q5, spacing=DESCENT_KNOT, label=f"retreat from {pp.dst_name}", fingers=[],
                                          fingers_min=CLEARANCE_MIN_SLIDE)
                plan.append(Waypoint(knots[-1], "lift", f"retreat from {pp.dst_name}", via=knots[:-1], dwell=SETTLE["lift"], tool_path_len=L,
                                     meta=dict(lift_check=True, lift_z=z_pre, slow=True)))
                pp.side_retreat = True
                return r_above.q5
            except PlanError as e:
                last_err = e
                del plan[n0:]
        raise PlanError(f"retreat from {pp.dst_name}: {last_err}")

    def _ascend(self, plan: MovePlan, q5, tool, gripper_pct: float, label: str, fingers, carried=None) -> Tuple[np.ndarray, np.ndarray]:
        """Vertical ascent from a pose to the local lift height (if higher).  Returns (q5 at the top, tool point)."""
        k = self.kin
        p, R, t = k.fk(q5, tool)
        z_top = self.lift_z_at(t[:2])
        if z_top <= t[2] + 0.002:
            return np.asarray(q5, float), t
        u = k.radial_dir(t, R[:, 2], tool, 0.0)
        tilt, psi = frame_params(R, u)
        lat = rotate_lat(u, psi)
        r_top = self._knot(np.array([t[0], t[1], z_top]), lat, tool, gripper_pct, TRAVEL_TILT_DEG, q5, tilt)
        if r_top is None:
            return np.asarray(q5, float), t
        knots, L = self.cart_path(q5, np.array([t[0], t[1], z_top]), lat, r_top.tilt_deg, tool, gripper_pct, q_end5=r_top.q5,
                                  spacing=DESCENT_KNOT, label=f"{label} (ascend)", fingers=fingers, carried=carried)
        plan.append(Waypoint(knots[-1], "lift", f"{label} (ascend)", via=knots[:-1], dwell=0.0, tool_path_len=L))
        return r_top.q5, k.fk(r_top.q5, tool)[2]

    def _transit(self, plan: MovePlan, q_from5, above_dst, lat, tilt1: float, tool, gripper_pct: float, q_end5, floor, label: str,
                 meta: dict, kind: str = "transit", fingers: Optional[Iterable[str]] = None,
                 carried: Optional[Tuple[str, float]] = None) -> None:
        """Carry at lift height: straight line (reach-ceiling clamped, obstacle floor, finger clearance); if the
        direct line is blocked, climb to the local ceiling and follow the A* corridor of grid_route (cells of
        ROUTE_CELL over the board where the floor is below the reach ceiling), leg by leg."""
        k = self.kin
        try:
            knots, L = self.cart_path(q_from5, above_dst, lat, tilt1, tool, gripper_pct, q_end5=q_end5, clamp_z=True, floor=floor,
                                      label=label, fingers=fingers, carried=carried)
            plan.append(Waypoint(knots[-1], kind, label, via=knots[:-1], dwell=SETTLE["transit"], tool_path_len=L, meta=meta))
            return
        except PlanError as e:
            direct_err = str(e)
        n0 = len(plan)
        try:
            q_up, t0 = self._ascend(plan, q_from5, tool, gripper_pct, label, fingers, carried)
        except PlanError as e:
            del plan[n0:]
            raise PlanError(f"{label}: direct: {direct_err}; ascent: {e}")
        n1 = len(plan)
        p1 = np.asarray(above_dst, float)
        p0, R0, _ = k.fk(q_up, tool)
        u_start = k.radial_dir(t0, R0[:, 2], tool, 0.0)
        psi0 = frame_params(R0, u_start)[1]
        blocked: List[np.ndarray] = []          # points where a routed leg failed the exact checks: re-route around them

        def floor_b(xy):
            for b in blocked:
                if np.hypot(xy[0] - b[0], xy[1] - b[1]) < ROUTE_CELL * 0.8:
                    return 10.0, "blocked"
            return floor(xy)
        errs = []
        for attempt in range(ROUTE_RETRIES):
            del plan[n1:]
            try:
                route = self.grid_route(t0[:2], p1[:2], floor_b, ROUTE_CELL)
            except PlanError as e:
                errs.append(str(e))
                break
            q = q_up
            try:
                for idx, pt_xy in enumerate(route):
                    last = idx == len(route) - 1
                    pt = p1 if last else np.array([pt_xy[0], pt_xy[1], min(self.lift_z(), self.lift_z_at(pt_xy))])
                    leg_lat = lat if last else rotate_lat(radial_lat(pt, 1.0, k.pan_axis_xy), psi0)
                    knots, L = self.cart_path(q, pt, leg_lat, tilt1 if last else 0.0, tool, gripper_pct, q_end5=q_end5 if last else None,
                                              clamp_z=True, floor=floor_b, label=f"{label} (route leg {idx + 1}/{len(route)})", fingers=fingers,
                                              carried=carried)
                    plan.append(Waypoint(knots[-1], kind, f"{label} (leg {idx + 1})", via=knots[:-1], dwell=0.0 if not last else SETTLE["transit"],
                                         tool_path_len=L, meta=meta if last else dict(meta, no_check=True)))
                    q = C.lerobot_to_qpos(knots[-1])[:5]
                plan.notes.append(f"{label}: routed in {len(route)} legs, {attempt} re-routes (direct path blocked: {direct_err})")
                return
            except PlanError as e:
                errs.append(str(e))
                if e.where is None:
                    break
                blocked.append(np.asarray(e.where, float)[:2])
        del plan[n0:]
        raise PlanError(f"{label}: direct: {direct_err}; route: " + " | ".join(errs[-3:]))

    @staticmethod
    def _q6(q5, gripper_pct: float) -> np.ndarray:
        return np.concatenate([C.rad2deg(q5), [gripper_pct]])

    def _open_above(self, plan: MovePlan, q5, open_pct: float, exclude, xy, label: str) -> None:
        """Open the jaws at a pose above the target (the transit was made with closed jaws); the open fingers must
        clear every other piece there."""
        clr, who = self.jaw_piece_clearance(q5, C.gripper_percent_to_frac(open_pct), exclude, xy)
        if clr < CLEARANCE_MIN_SLIDE:
            raise PlanError(f"{label}: the open jaws would hit {who} (clearance {clr*1000:.1f} mm)")
        plan.append(Waypoint(self._q6(q5, open_pct), "preopen", label, dwell=SETTLE["open"]))

    # ------------------------------------------------------------------ planning
    def _pick_place(self, plan: MovePlan, pp: PickPlace, q_cur5: np.ndarray, exclude_at_place: Sequence[str] = ()) -> np.ndarray:
        """Append the waypoints of one pick-and-place to plan; returns the arm joints at the end (lift height
        above the destination or beside it after a side retreat, jaws open).  Records the piece's destination in
        self._planned_pos so the clearance checks of later steps see it there.  The grasp heights
        config.grasp_height_candidates (the nominal one, then lower ones: 50 % / 40 % of the height for 'cylinders',
        the lower end of the grasp zone for 'bands') are tried in turn (a lower grasp lets the carried piece pass
        over neighbours where the reach ceiling is low)."""
        n0 = len(plan)
        errs = []
        nominal = self.kin.grasp_height(pp.ptype)
        for h in C.grasp_height_candidates(pp.ptype, self.collision):
            hang = nominal if h is None else float(h)
            if h is not None and hang >= nominal - 1e-6:
                continue
            try:
                q_end = self._pick_place_once(plan, pp, q_cur5, exclude_at_place, hang)
                if h is not None:
                    H = C.PIECE_SPECS[pp.ptype][0]
                    plan.notes.append(f"{pp.piece}: grasped at {hang / H * 100:.0f} % of its height ({hang*1000:.1f} mm) instead of the "
                                      f"nominal {nominal*1000:.1f} mm ({errs[-1][:90]}...)")
                pp.hang = hang
                return q_end
            except PlanError as e:
                errs.append(str(e))
                del plan[n0:]
                self._planned_pos.pop(pp.piece, None)
                plan.expected.pop(pp.piece, None)
        raise PlanError(errs[0] + ("" if len(errs) == 1 else f" [lower grasps: {' / '.join(e[:150] for e in errs[1:])}]"))

    def _pick_place_once(self, plan: MovePlan, pp: PickPlace, q_cur5: np.ndarray, exclude_at_place: Sequence[str], hang: float) -> np.ndarray:
        s, k = self.scene, self.kin
        pt = pp.ptype
        open_q = C.gripper_open_ctrl()
        close_q = C.gripper_close_ctrl(pt, self.collision)
        tool = k.grasp_tool_site(pt)            # direction-independent default; per grasp direction for the knight (grasp_tool)
        tool_p = k.place_tool_site(pt)          # the held piece sits HELD_Z_OFFSET toward the fixed pad
        pp.close_pct = C.gripper_frac_to_percent(close_q)
        self._held_theta.pop(pp.piece, None)
        src = np.asarray(pp.src_xyz, float)
        dst = np.array([pp.dst_xyz[0], pp.dst_xyz[1], pp.dst_xyz[2] + PLACE_EXTRA_Z])
        # can the open jaws get above the piece at its square / at its destination?
        top_src = self.lift_z_at(src[:2]) >= self.z_safe_above(src, pt)
        top_dst = self.lift_z_at(dst[:2]) >= self.z_safe_above(dst, pt)
        floor_empty = lambda xy: self.obstacle_floor(xy, 0.0, [pp.piece], SWEEP_R_EMPTY, self.finger_radius(TRAVEL_PCT))
        # --- grasp candidates (jaws open must clear the neighbours AND the piece's own stem; ranked)
        cands = self.grasp_pose(src, pt, exclude=[pp.piece], open_q=open_q, label=f"grasp {pp.piece}", roll_from=float(q_cur5[4]),
                                all_candidates=True, max_lean=grasp_max_lean(pt) if top_src else GRASP_MAX_LEAN_SLIDE,
                                min_clr=CLEARANCE_MIN if top_src else CLEARANCE_MIN_SLIDE, exhaustive=not top_src, hang=hang,
                                target=pp.piece, piece=pp.piece)
        n_start = len(plan)
        if top_src and top_dst:
            # a lifted piece lands with the tilt between the grasp and release jaw orientations (landing_tilt_deg);
            # the far corners (a1/h1) leave a single reachable release orientation, so the GRASP must match it:
            # probe the release candidates once and rank the grasps that can be released upright first
            cands = self.rank_grasps_for_release(cands, dst, pt, list(exclude_at_place) + [pp.piece], open_q, hang,
                                                 self.place_tool(pt, pp.piece, cands[0][1]), pp.piece)
        if top_src:
            errs = []
            for g, g_ang, g_clr, g_oq in cands[:GRASP_CANDIDATES_TRIED]:
                del plan[n_start:]
                tool = self.grasp_tool(pt, pp.piece, g_ang)
                try:
                    pp.grasp, pp.grasp_lat_deg, pp.grasp_clearance, pp.open_pct = g, g_ang, g_clr, C.gripper_frac_to_percent(g_oq)
                    g_lat = rotate_lat(radial_lat(np.array([src[0], src[1], 0]), 1.0, k.pan_axis_xy), g_ang)
                    g_z = float(k.fk(g.q5, tool)[2][2])
                    fz_src = self.obstacle_floor(src[:2], 0.0, [pp.piece], SWEEP_R_EMPTY)[0]
                    r_above, z_src = self._above_pose(src[:2], self.lift_z_at(src[:2]), g_z, g_lat, tool, pp.open_pct, g.q5, g.tilt_deg,
                                                      fz_src, f"pre-grasp {pp.piece}")
                    above_src = np.array([src[0], src[1], z_src])
                    # 1. approach from the current pose to above the piece (empty CLOSED jaws over the other pieces), open there
                    self._transit(plan, q_cur5, above_src, g_lat, r_above.tilt_deg, tool, TRAVEL_PCT, r_above.q5, floor_empty, f"to {pp.piece}",
                                  dict(lift_z=z_src), kind="pregrasp", fingers=[])
                    self._open_above(plan, r_above.q5, pp.open_pct, [pp.piece], src[:2], f"open above {pp.piece}")
                    # 2. descend (approach with the lean settling, then constant orientation) to the grasp height; the
                    #    piece's own stem is in the finger check (a lean sweeps the finger root)
                    self._descend(plan, r_above.q5, g, g_lat, tool, pp.open_pct, src[:2], z_src, "descend_grasp", f"grasp {pp.piece}",
                                  dict(piece=pp.piece), fingers=[pp.piece], target=pp.piece)
                    break
                except PlanError as e:
                    errs.append(f"lean {g.tilt_deg:+.1f} dir {g_ang:+.0f}: {e}")
            else:
                del plan[n_start:]
                raise PlanError(f"grasp {pp.piece}: no candidate with a safe descent: " + " | ".join(x[:160] for x in errs[:3]))
        else:
            # the arm cannot get the open jaws above this piece (reach ceiling near the base): arrive BESIDE it at
            # grasp height, perpendicular to the opening axis, and slide in so the stem enters between the fingers
            self._side_approach(plan, pp, q_cur5, cands, tool, floor_empty)
            g, g_ang = pp.grasp, pp.grasp_lat_deg
            tool = self.grasp_tool(pt, pp.piece, g_ang)
            g_lat = rotate_lat(radial_lat(np.array([src[0], src[1], 0]), 1.0, k.pan_axis_xy), g_ang)
            g_z = float(k.fk(g.q5, tool)[2][2])
            fz_src = self.obstacle_floor(src[:2], 0.0, [pp.piece], SWEEP_R_EMPTY)[0]
            r_above, z_src = self._above_pose(src[:2], self.lift_z_at(src[:2]), g_z, g_lat, tool, pp.open_pct, g.q5, g.tilt_deg,
                                              fz_src, f"pre-grasp {pp.piece}")
            above_src = np.array([src[0], src[1], z_src])
            plan.notes.append(f"{pp.piece}: side approach (tips cannot clear its top: ceiling {self.lift_z_at(src[:2]):.3f} < "
                              f"{self.z_safe_above(src, pt):.3f})")
        if z_src < g_z + 0.002:
            pp.slide = True
            plan.notes.append(f"{pp.piece}: no lift possible at its square (reach ceiling), the piece slides")
        # --- place pose: the carried piece is excluded and the pieces of earlier steps are at their planned spots.
        #     The piece is held rigidly at the grasp lean / opening direction, so the same (lean, direction) at the
        #     release lands it upright (tilt_pref / lat_pref); the place tool point is the HELD stem position.
        # the released piece will stand at dst when the jaws open/retreat: the target check sees it there (the
        # neighbour check excludes it).  The piece is held rigidly, so its orientation relative to the jaws (and
        # the held tool point) is the grasp's for every release direction
        self._held_theta[pp.piece] = self.theta_held(pp.piece, g_ang)
        tool_p = self.place_tool(pt, pp.piece, g_ang)
        self._planned_pos[pp.piece] = np.array(pp.dst_xyz, float)
        try:
            p_max_lean = max(grasp_max_lean(pt), abs(g.tilt_deg)) if top_dst else GRASP_MAX_LEAN_SLIDE
            pcands = self.grasp_pose(dst, pt, exclude=list(exclude_at_place) + [pp.piece], open_q=open_q, lat_pref=g_ang,
                                     label=f"place {pp.piece} at {pp.dst_name}", roll_from=float(g.q5[4]), all_candidates=True,
                                     max_lean=p_max_lean, min_clr=CLEARANCE_MIN if top_dst else CLEARANCE_MIN_SLIDE,
                                     # a corner grasp (lean beyond the 5 deg spec: a1/h1 ...) has few landing-compatible releases
                                     # and they can sit one lean level further out (Kg1-h1: the +8.0 deg level), so search all levels
                                     exhaustive=(not top_dst) or abs(g.tilt_deg) > C.JAW_VERTICAL_TOL_DEG, hang=hang, tilt_pref=g.tilt_deg, tool=tool_p, target=pp.piece,
                                     target_min=RELEASE_CLEARANCE_MIN, piece=pp.piece)
            if not top_dst:
                # the jaws must be able to back out sideways after the release: keep the candidates with a free side
                why: list = []
                opts = self._side_options(pcands, pp.piece, pt, dst, tool_p, f"retreat from {pp.dst_name}", why, tilt_pref=g.tilt_deg)
                if not opts:
                    raise PlanError(f"place {pp.piece} at {pp.dst_name}: no free side to back away from it after the release "
                                    f"({len(pcands)} release candidates): " + "; ".join(why[:6]))
                pcands = [(o[4], o[5], o[6], o[10]) for o in opts]
            pcands = self.rank_release_candidates(pcands, g, pt)
        finally:
            self._planned_pos.pop(pp.piece, None)
        # 3. close
        plan.append(Waypoint(self._q6(g.q5, pp.close_pct), "close", f"close on {pp.piece}", meta=dict(piece=pp.piece, carrying=True)))
        # 4. lift straight up (slow; constant orientation to the pre-height, then to the lift height)
        self._lift(plan, pp, g, g_lat, tool, above_src, r_above.q5, r_above.tilt_deg, hang, z_src)
        n_place = len(plan)
        errs = []
        for pl, p_ang, p_clr, p_oq in pcands[:GRASP_CANDIDATES_TRIED]:
            del plan[n_place:]
            self._planned_pos.pop(pp.piece, None)
            plan.expected.pop(pp.piece, None)
            try:
                pp.place, pp.place_lat_deg, pp.place_clearance, pp.place_open_pct = pl, p_ang, p_clr, C.gripper_frac_to_percent(p_oq)
                pp.landing_tilt_deg = self.landing_tilt_deg(g, pl, pt)
                p_lat = rotate_lat(radial_lat(np.array([dst[0], dst[1], 0]), 1.0, k.pan_axis_xy), p_ang)
                p_z = float(k.fk(pl.q5, tool_p)[2][2])
                # 5. transit at lift height (reach ceiling, obstacle floor, detour if needed) to above the destination
                self._planned_pos[pp.piece] = np.array([-1.0, -1.0, 0.0])
                try:
                    floor_carry = lambda xy: self.obstacle_floor(xy, hang, [pp.piece], SWEEP_R_CARRY, self.finger_radius(pp.close_pct))
                    fz_dst = self.obstacle_floor(dst[:2], hang, [pp.piece], SWEEP_R_CARRY, self.finger_radius(pp.close_pct))[0]
                    r_above2, z_dst = self._above_pose(dst[:2], self.lift_z_at(dst[:2]), p_z, p_lat, tool_p, pp.close_pct, pl.q5, pl.tilt_deg,
                                                       fz_dst, f"above {pp.dst_name}")
                    above_dst = np.array([dst[0], dst[1], z_dst])
                    carry_meta = dict(piece=pp.piece, carrying=True, carried_offset=np.array([0, 0, -hang]), lift_z=z_dst, slow=pp.slide, hang=hang)
                    self._transit(plan, r_above.q5, above_dst, p_lat, r_above2.tilt_deg, tool_p, pp.close_pct, r_above2.q5, floor_carry,
                                  f"carry {pp.piece} to {pp.dst_name}", carry_meta, fingers=[pp.piece], carried=(pp.piece, hang))
                finally:
                    self._planned_pos.pop(pp.piece, None)
                # 6-8. descend, release, retreat (re-planned at execution time with the measured in-jaw offset)
                n_tail = len(plan)
                q_end = self._place_tail(plan, pp, pl, p_ang, p_clr, p_oq, tool_p, hang, dst, z_dst, r_above2.q5, exclude_at_place, top_dst)
                plan.expected[pp.piece] = np.array(pp.dst_xyz, float)
                i_desc = next(i for i in range(n_tail, len(plan)) if plan[i].kind == "descend_place")
                plan[i_desc].meta["adapt"] = dict(pp=pp, pl=pl, p_ang=p_ang, p_clr=p_clr, p_oq=p_oq, tool_p=tool_p, hang=hang, dst=dst.copy(),
                                                  z_dst=z_dst, exclude=list(exclude_at_place), top_dst=top_dst, tail_len=len(plan) - i_desc)
                if pp.side_retreat:
                    plan.notes.append(f"{pp.piece}: side retreat from {pp.dst_name} (tips cannot clear its top)")
                return q_end
            except PlanError as e:
                errs.append(f"lean {pl.tilt_deg:+.1f} dir {p_ang:+.0f}: {e}")
        del plan[n_place:]
        self._planned_pos.pop(pp.piece, None)
        plan.expected.pop(pp.piece, None)
        raise PlanError(f"place {pp.piece} at {pp.dst_name}: no release candidate with a safe carry/retreat: " + " | ".join(x[:200] for x in errs[:3]))

    def _place_tail(self, plan: MovePlan, pp: PickPlace, pl: IKResult, p_ang: float, p_clr: float, p_oq: float, tool_p, hang: float,
                    dst, z_dst: float, q_from5, exclude_at_place: Sequence[str], top_dst: bool) -> np.ndarray:
        """Waypoints from above the destination down to the release and back up: descend (approach + constant
        orientation) onto the place pose pl, open, then centre the open jaws on the stem and retreat straight up
        (constant orientation first) when the tips can end above the placed piece's top, otherwise back out
        sideways first.  `dst` is the base target (PLACE_EXTRA_Z above the surface).  Returns the arm joints at the
        end (jaws open, at the local lift height)."""
        k = self.kin
        pt = pp.ptype
        p_lat = rotate_lat(radial_lat(np.array([dst[0], dst[1], 0]), 1.0, k.pan_axis_xy), p_ang)
        p_z = float(k.fk(pl.q5, tool_p)[2][2])
        standing = np.array([dst[0], dst[1], dst[2] - PLACE_EXTRA_Z])
        self._planned_pos[pp.piece] = np.array([-1.0, -1.0, 0.0])
        try:
            self._descend(plan, q_from5, pl, p_lat, tool_p, pp.close_pct, dst[:2], z_dst, "descend_place",
                          f"place {pp.piece} on {pp.dst_name}", dict(piece=pp.piece, carrying=True), fingers=[pp.piece], carried=(pp.piece, hang))
        finally:
            self._planned_pos.pop(pp.piece, None)
        plan.append(Waypoint(self._q6(pl.q5, pp.place_open_pct), "open", f"release {pp.piece}", dwell=SETTLE["open"], meta=dict(piece=pp.piece)))
        self._planned_pos[pp.piece] = standing
        if z_dst >= self.z_safe_above(dst, pt) - 1e-6:
            q_c, t_c = self._centre_shift(plan, pp.piece, pt, pl, p_lat, tool_p, p_oq, f"centre on {pp.piece}")
            centred = bool(np.linalg.norm(t_c - k.fk(pl.q5, tool_p)[2]) > 1e-6)
            pp.side_retreat = False
            return self._retreat(plan, pp.piece, q_c, p_lat, pl.tilt_deg, tool_p, pp.place_open_pct, self.lift_z_at(dst[:2]), p_z,
                                 f"retreat from {pp.dst_name}", target_min=TARGET_CLEARANCE_MIN if centred else RELEASE_CLEARANCE_MIN)
        return self._side_retreat(plan, pp, pl, p_ang, p_clr, p_oq, tool_p, dst)

    def _adapt_place(self, adapt: dict, q_from6) -> Optional[List[Waypoint]]:
        """Execution-time place correction (see ADAPT_MIN): measure the carried piece's base offset in the gripper
        frame at the current pose, predict where it would land at the planned release pose, and re-plan the
        descent / release / retreat (same orientation, target shifted by the prediction error) from the current
        arm target.  Returns the new tail waypoints, or None (no correction needed / possible: the planned tail
        stays).  self.state['place_correction'] records (piece, shift xy, applied)."""
        s, k = self.scene, self.kin
        pp, pl, tool_p = adapt["pp"], adapt["pl"], adapt["tool_p"]
        piece, dst = pp.piece, np.asarray(adapt["dst"], float)
        q5 = C.lerobot_to_qpos(np.asarray(q_from6, float))[:5]
        p_now, R_now, stem_now = k.fk(q5, tool_p)
        off_site = R_now.T @ (s.piece_pos(piece) - stem_now)
        p_pl, R_pl, stem_pl = k.fk(pl.q5, tool_p)
        base_pred = stem_pl + R_pl @ off_site
        shift = dst[:2] - base_pred[:2]
        n = float(np.linalg.norm(shift))
        self.state["place_correction"] = (piece, shift.copy(), False)
        self.state["adapt_reason"] = "within tolerance" if n < ADAPT_MIN else ""
        if n < ADAPT_MIN:
            return None
        cap = min(1.0, ADAPT_MAX / n)           # a shift beyond ADAPT_MAX is applied capped (then half of the cap)
        p_ang, p_oq, hang, z_dst, top_dst = adapt["p_ang"], adapt["p_oq"], adapt["hang"], adapt["z_dst"], adapt["top_dst"]
        p_lat = rotate_lat(radial_lat(np.array([dst[0], dst[1], 0]), 1.0, k.pan_axis_xy), p_ang)
        saved = self._planned_pos
        side0 = pp.side_retreat
        full = shift.copy()
        try:
            for frac in (cap, 0.5 * cap):       # a crowded / reach-limited square may only allow part of the correction
                shift = full * frac
                dst_s = np.array([dst[0] + shift[0], dst[1] + shift[1], dst[2]])
                r = k.solve(stem_pl + np.array([shift[0], shift[1], 0.0]), p_lat, tool_p, tilt_deg=pl.tilt_deg, q_init=pl.q5)
                if not r.converged or r.pos_err > 0.001 or abs(r.tilt_deg - pl.tilt_deg) > 1.0 or self.arm_self_collision(r.q5, p_oq):
                    self.state["adapt_reason"] = f"shifted release pose unreachable (pos err {r.pos_err*1000:.1f} mm, lean {r.tilt_deg:+.1f}, limits {r.at_limit})"
                    continue
                self._planned_pos = {piece: np.array([dst_s[0], dst_s[1], dst_s[2] - PLACE_EXTRA_Z])}
                clr, _ = self.jaw_piece_clearance(r.q5, p_oq, [piece], dst_s[:2])
                clr_t, _ = self.jaw_piece_clearance(r.q5, p_oq, (), dst_s[:2], only=piece)
                if clr < (CLEARANCE_MIN if top_dst else CLEARANCE_MIN_SLIDE) or clr_t < RELEASE_CLEARANCE_MIN:
                    self.state["adapt_reason"] = f"shifted release pose: clearance {clr*1000:.1f} mm, target {clr_t*1000:.1f} mm"
                    continue
                self._planned_pos = {}
                tail = MovePlan(None)
                try:
                    self._place_tail(tail, pp, r, p_ang, clr, p_oq, tool_p, hang, dst_s, z_dst, q5, adapt["exclude"], top_dst)
                except PlanError as e:
                    pp.side_retreat = side0
                    self.state["adapt_reason"] = f"tail re-plan failed: {e}"
                    continue
                self.state["place_correction"] = (piece, shift.copy(), True)
                self.state["adapt_reason"] = (f"applied ({frac*100:.0f} % of the measured {np.linalg.norm(full)*1000:.1f} mm"
                                              f"{', capped to %.0f mm' % (ADAPT_MAX*1000) if cap < 1.0 else ''})")
                for w in tail:
                    w.meta["adapted"] = True
                return list(tail)
        finally:
            self._planned_pos = saved
        return None

    def plan_move(self, move, board: chess.Board, q_start: Optional[np.ndarray] = None, return_to_watch: bool = True) -> MovePlan:
        """Plan `move` (chess.Move, UCI string or (from_sq, to_sq) names) on `board` (the position BEFORE the
        move) using the pieces where they physically are in the scene.  q_start: LeRobot 6-vector the plan
        starts from (default: the scene's current joints).  A start pose far from the watch pose is first brought
        there by a collision-checked joint-space move.  If the return to the watch pose cannot be planned the
        plan still contains the executable move and plan.watch_error says why (call plan_to_watch() later)."""
        t0 = time.time()
        s = self.scene
        if isinstance(move, (tuple, list)):
            move = chess.Move.from_uci(move[0] + move[1])
        elif isinstance(move, str):
            move = chess.Move.from_uci(move)
        if move not in board.legal_moves and not board.is_pseudo_legal(move):
            raise PlanError(f"{move.uci()} is not legal in {board.fen()}")
        state = s.board_state()                      # square -> body name (physical)
        frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
        if frm not in state:
            raise PlanError(f"no piece standing on {frm} in the scene (board_state)")
        mover = state[frm]
        plan = MovePlan(move)
        q_start6 = np.asarray(s.get_joints() if q_start is None else q_start, float)
        q_start5 = C.lerobot_to_qpos(q_start6)[:5]
        prefix: List[Waypoint] = []
        watch = np.asarray(s.watch_pose(), float)
        qw5 = C.lerobot_to_qpos(watch)[:5]
        if np.max(np.abs(np.degrees(q_start5 - qw5))) > START_NEAR_WATCH_DEG:
            gq = C.lerobot_to_qpos(q_start6)[5]
            ok, why = self.joint_path_clear(q_start5, qw5, gq)
            if ok:
                via = [np.concatenate([C.rad2deg(q_start5 + (qw5 - q_start5) * i / 8), [q_start6[5]]]) for i in range(1, 8)]
                prefix.append(Waypoint(np.concatenate([watch[:5], [q_start6[5]]]), "watch", "to watch pose (start)", via=via,
                                       dwell=SETTLE["watch"]))
                q_start5 = qw5.copy()
                plan.notes.append("start pose far from the watch pose: a joint-space move to the watch pose is prepended")
            else:
                plan.notes.append(f"start pose far from the watch pose and the joint-space move there is not collision-free ({why}): "
                                  f"planning from the start pose")
        steps: List[PickPlace] = []
        # --- capture (incl. en passant): captured piece -> drop zone first
        if board.is_capture(move):
            if board.is_en_passant(move):
                cap_sq = chess.square_name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
            else:
                cap_sq = to
            if cap_sq not in state:
                raise PlanError(f"capture on {cap_sq} but no piece stands there in the scene")
            cell = s.free_drop_cell()
            if cell is None:
                raise PlanError("drop zone full")
            cap = state[cap_sq]
            steps.append(PickPlace(cap, s.piece_type[cap], s.piece_pos(cap), s.drop_cell_xyz(*cell), f"drop{cell[0]}{cell[1]}"))
        # --- the mover (promotion: pawn -> drop, spare queen -> target)
        if move.promotion:
            if move.promotion != chess.QUEEN:
                plan.notes.append(f"promotion to {chess.piece_symbol(move.promotion)} executed as a queen (v1 has spare queens only)")
            colour = s.piece_colour[mover]
            spare = f"{colour}_queen_spare_1"
            sp = s.piece_pos(spare)
            if np.linalg.norm(sp[:2] - s.spare_xyz(colour)[:2]) > 0.012:
                raise PlanError(f"spare queen {spare} is not parked on its cell (already used)")
            cell = s.free_drop_cell() if not steps else self._next_free_cell_after(steps)
            if cell is None:
                raise PlanError("drop zone full")
            steps.append(PickPlace(mover, s.piece_type[mover], s.piece_pos(mover), s.drop_cell_xyz(*cell), f"drop{cell[0]}{cell[1]}"))
            steps.append(PickPlace(spare, "queen", sp, s.square_xyz(to), to))
        else:
            steps.append(PickPlace(mover, s.piece_type[mover], s.piece_pos(mover), s.square_xyz(to), to))
        # --- castling: the rook after the king
        if board.is_castling(move):
            rank = frm[1]
            if chess.square_file(move.to_square) > chess.square_file(move.from_square):
                r_from, r_to = f"h{rank}", f"f{rank}"
            else:
                r_from, r_to = f"a{rank}", f"d{rank}"
            if r_from not in state:
                raise PlanError(f"castling but no rook on {r_from}")
            rook = state[r_from]
            steps.append(PickPlace(rook, s.piece_type[rook], s.piece_pos(rook), s.square_xyz(r_to), r_to))
        # --- build the waypoints (pieces moved by earlier steps are seen at their new spots: _planned_pos)
        orders = [steps]
        if board.is_castling(move):
            # spec order is king then rook; with the king already beside the rook the rook's grasp may have no
            # collision-free jaw direction (finger root vs the king's stem), so rook-first is the fallback
            orders.append(steps[::-1])
        last_err = None
        errs: List[str] = []
        final_pos: Dict[str, np.ndarray] = {}
        q_cur = q_start5.copy()
        for order in orders:
            plan.clear()
            plan.extend(prefix)
            plan.expected = {}
            self._planned_pos = {}
            q_cur = q_start5.copy()
            try:
                for pp in order:
                    pp.grasp = pp.place = None
                    pp.side_approach = pp.side_retreat = pp.slide = False
                    q_cur = self._pick_place(plan, pp, q_cur)
                steps = order
                last_err = None
                final_pos = dict(self._planned_pos)
                break
            except PlanError as e:
                last_err = e
                errs.append(f"{'-'.join(p.piece for p in order)}: {e}")
            finally:
                self._planned_pos = {}
        if last_err is not None:
            raise PlanError(" || ".join(errs)) if len(errs) > 1 else last_err
        if board.is_castling(move) and steps[0].ptype == "rook":
            plan.notes.append("castling executed rook first (king-first has no collision-free rook grasp)")
        plan.steps = steps
        if return_to_watch:
            self._planned_pos = final_pos            # the moved pieces are seen at their destinations
            n0 = len(plan)
            try:
                self._to_watch(plan, q_cur)
            except PlanError as e:
                del plan[n0:]
                plan.watch_error = str(e)
                plan.notes.append(f"return to the watch pose not planned: {e}")
            finally:
                self._planned_pos = {}
        plan.plan_seconds = time.time() - t0
        return plan

    def _next_free_cell_after(self, steps: List[PickPlace]):
        taken = {p.dst_name for p in steps}
        pieces = np.array([self.scene.piece_pos(n)[:2] for n in self.scene.piece_names])
        for i, j in self.scene.drop_zone_cells():
            if f"drop{i}{j}" in taken:
                continue
            c = self.scene.drop_cell_xyz(i, j)[:2]
            if np.min(np.linalg.norm(pieces - c, axis=1)) > 0.014:
                return (i, j)
        return None

    def _to_watch(self, plan: MovePlan, q_cur5: np.ndarray) -> None:
        """Close the jaws (the tips are above / beside every piece at this point), then a straight tool-point
        path to the watch pose (with the climb-and-detour fallback of _transit)."""
        s, k = self.scene, self.kin
        watch = np.asarray(s.watch_pose(), float)
        qw5 = C.lerobot_to_qpos(watch)[:5]
        tool = k.fixed_tip_site
        plan.append(Waypoint(self._q6(q_cur5, watch[5]), "close", "close jaws", dwell=0.0))
        p_w, R_w, t_w = k.fk(qw5, tool)
        u_w = k.radial_dir(t_w, R_w[:, 2], tool, 0.0)
        tilt_w, psi_w = frame_params(R_w, u_w)
        lat_w = rotate_lat(u_w, psi_w)
        self._transit(plan, q_cur5, t_w, lat_w, tilt_w, tool, watch[5], qw5,
                      lambda xy: self.obstacle_floor(xy, 0.0, (), SWEEP_R_EMPTY, self.finger_radius(watch[5])),
                      "to watch", dict(), kind="watch", fingers=[])
        plan[-1].q6 = np.asarray(watch, float)
        plan[-1].dwell = SETTLE["watch"]
        plan[-1].label = "watch pose"

    def plan_to_watch(self) -> MovePlan:
        """Plan the return to the watch pose from the scene's current arm pose (pieces where they are)."""
        plan = MovePlan(None)
        self._planned_pos = {}
        self._to_watch(plan, C.lerobot_to_qpos(self.scene.get_joints())[:5])
        return plan

    # ------------------------------------------------------------------ execution
    def segment_duration(self, q_from: np.ndarray, wp: Waypoint, max_joint_speed_deg_s: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """Min-jerk duration of a waypoint's path.  The path parameter advances by the per-interval max joint
        motion (cum), so the joint speed in interval k is |dq_jk| / share_k * ds/dt <= cum_total * 1.875 / T:
        T = 1.875 * cum_total / v_max bounds the PEAK joint speed of the piecewise-linear profile (not just the
        average).  Also: peak tool speed <= V_TOOL_MAX (V_DESCENT_MAX for slow segments), at least
        MIN_SEGMENT_TIME (gripper-only segments: GRIPPER_MOVE_TIME)."""
        seq = np.array([q_from] + [np.asarray(v, float) for v in wp.via] + [np.asarray(wp.q6, float)])
        d = np.abs(np.diff(seq[:, :5], axis=0))
        seg = d.max(axis=1)                      # per-interval "length" = largest joint motion
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] < 1e-9:
            T = GRIPPER_MOVE_TIME if abs(seq[-1, 5] - seq[0, 5]) > 1e-6 else 1e-3
            return T, seq, cum
        T = 1.875 * cum[-1] / max_joint_speed_deg_s
        if wp.tool_path_len > 0:
            T = max(T, 1.875 * wp.tool_path_len / (V_DESCENT_MAX if wp.meta.get("slow") else V_TOOL_MAX))
        T = max(T, MIN_SEGMENT_TIME)
        return T, seq, cum

    @staticmethod
    def _interp(seq: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
        if cum[-1] < 1e-9:
            return seq[0] + (seq[-1] - seq[0]) * s
        x = s * cum[-1]
        i = int(np.searchsorted(cum, x, side="right") - 1)
        i = min(max(i, 0), len(cum) - 2)
        w = (x - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
        return seq[i] + (seq[i + 1] - seq[i]) * min(1.0, max(0.0, w))

    def action_generator(self, plan: Sequence[Waypoint], control_hz: float = C.CONTROL_HZ,
                         max_joint_speed_deg_s: float = C.JOINT_SPEED_DEG_S, log: Optional[list] = None,
                         abort_on_drop: bool = True, adapt_place: bool = True) -> Generator[np.ndarray, None, None]:
        """Yield one LeRobot action per control step.  The caller must apply each action with
        scene.control_step(action) before asking for the next one (the generator reads the scene state for the
        settling waits).  `log` (optional list) receives (label, sim_t_start, sim_t_end, wall) per waypoint.
        self.state is updated with the current phase/label; self.state['abort'] is set when a carried piece is
        lost or a standing piece knocked over (abort_on_drop) and the generator stops.  adapt_place: re-plan each
        place tail with the measured in-jaw offset (_adapt_place); self.state['plan'] is the waypoint list actually
        executed.  Consecutive actions never differ by more than max_joint_speed_deg_s / control_hz on any arm joint."""
        s = self.scene
        dt = 1.0 / control_hz
        q_from = np.asarray(s.get_joints(), float)
        wps: List[Waypoint] = list(plan)             # working list: place tails may be re-planned on the fly
        expected = dict(getattr(plan, "expected", {}))
        self.state = dict(phase="running", waypoint=-1, label="", abort=None, drop=None, plan=wps, corrections=[])
        lying0 = [n for n in s.piece_names if s.piece_tilt_deg(n) > KNOCKED_TILT_DEG]     # already down before the plan
        wi = -1
        while wi + 1 < len(wps):
            wi += 1
            wp = wps[wi]
            if adapt_place and "adapt" in wp.meta:
                new_tail = self._adapt_place(wp.meta["adapt"], q_from)
                self.state["corrections"].append(self.state.get("place_correction"))
                if new_tail is not None:
                    wps[wi:wi + wp.meta["adapt"]["tail_len"]] = new_tail
                    wp = wps[wi]
            T, seq, cum = self.segment_duration(q_from, wp, max_joint_speed_deg_s)
            t_sim0, wall0 = s.data.time, time.time()
            self.state.update(waypoint=wi, label=wp.label, kind=wp.kind)
            n_steps = max(1, int(math.ceil(T / dt)))
            for i in range(1, n_steps + 1):
                sfrac = minjerk(i / n_steps)
                # gripper-only segments: ramp the gripper, keep the arm target
                a = self._interp(seq, cum, sfrac) if cum[-1] >= 1e-9 else seq[0] + (seq[-1] - seq[0]) * sfrac
                yield np.asarray(a, float)
            # dwell / settle
            target = np.asarray(wp.q6, float)
            if wp.kind == "close":
                t_start = s.data.time
                still = 0
                while True:
                    yield target.copy()
                    v = abs(float(s.data.qvel[s.qvel_adr[5]]))
                    still = still + 1 if v < GRIPPER_SETTLE_VEL else 0
                    el = s.data.time - t_start
                    if (el >= SETTLE["close_min"] and still >= 3) or el >= SETTLE["close_max"]:
                        break
            elif wp.dwell > 0:
                for _ in range(int(math.ceil(wp.dwell / dt))):
                    yield target.copy()
            if log is not None:
                log.append((wp.label, t_sim0, s.data.time, time.time() - wall0))
            # carried-piece check after a lift/transit with a piece
            if abort_on_drop and "carried_offset" in wp.meta and not wp.meta.get("no_check"):
                piece = wp.meta["piece"]
                tool_exp = self._expected_carried(piece, target, wp.meta.get("hang"))
                if tool_exp is not None:
                    dev = float(np.linalg.norm(s.piece_pos(piece) - tool_exp))
                    self.state["drop"] = (piece, dev)
                    if dev > DROP_CHECK_TOL:
                        self.state["abort"] = f"{piece} lost during '{wp.label}' (deviation {dev*1000:.1f} mm)"
                        return
            # a standing / released piece knocked over or pushed away: stop (the board is no longer recoverable
            # by this plan; the game loop must stop or re-sync)
            if abort_on_drop:
                why = self._knocked_over(wps, wi, lying0, expected)
                if why is not None:
                    self.state["abort"] = why
                    return
            q_from = target
        self.state["phase"] = "done"

    def _expected_carried(self, piece: str, q6: np.ndarray, hang: Optional[float] = None) -> Optional[np.ndarray]:
        """Expected base position of a carried piece for an arm target: the stem point minus the grasp height
        (hang; default the type's nominal one) along the jaw axis."""
        pt = self.scene.piece_type[piece]
        p, R, stem = self.kin.fk(C.lerobot_to_qpos(q6)[:5], self.kin.place_tool_site(pt, self._held_theta.get(piece)))
        return stem + R[:, 0] * (self.kin.grasp_height(pt) if hang is None else hang)

    def _knocked_over(self, plan: Sequence[Waypoint], wi: int, ignore: Iterable[str] = (), expected: Optional[dict] = None) -> Optional[str]:
        """Execution abort check after waypoint wi: a standing piece (every piece except the one being carried by
        this step and those in `ignore`, e.g. pieces already lying when the plan started) tilted beyond
        KNOCKED_TILT_DEG, or a piece already released by this plan displaced beyond KNOCKED_DISP from its target.
        Returns the reason or None."""
        s = self.scene
        wp = plan[wi]
        carried = wp.meta.get("piece") if wp.meta.get("carrying") else None
        if expected is None:
            expected = getattr(plan, "expected", {})
        released = set()
        for w in plan[:wi + 1]:
            if w.kind == "open" and w.meta.get("piece"):
                released.add(w.meta["piece"])
        ign = set(ignore)
        for n in s.piece_names:
            if n == carried or n in ign:
                continue
            tilt = s.piece_tilt_deg(n)
            if tilt > KNOCKED_TILT_DEG:
                return f"{n} knocked over during '{wp.label}' (tilt {tilt:.1f} deg)"
            if n in released and n in expected:
                d = float(np.linalg.norm(s.piece_pos(n)[:2] - np.asarray(expected[n])[:2]))
                if d > KNOCKED_DISP:
                    return f"{n} displaced {d*1000:.1f} mm from its target after its release (during '{wp.label}')"
        return None

    def _live_arm_piece_contacts(self) -> List[str]:
        """Pieces currently in contact with the arm in the LIVE simulation (after a step)."""
        s, m, d = self.scene, self.scene.model, self.scene.data
        out = []
        for i in range(d.ncon):
            con = d.contact[i]
            if con.dist >= 0:
                continue
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if b1 in s.arm_body_ids and b2 in self.piece_bodies:
                out.append(m.body(b2).name)
            elif b2 in s.arm_body_ids and b1 in self.piece_bodies:
                out.append(m.body(b1).name)
        return out

    def execute(self, plan: MovePlan, control_hz: float = C.CONTROL_HZ, max_joint_speed_deg_s: float = C.JOINT_SPEED_DEG_S,
                progress=None, abort_on_drop: bool = True, check_watch: bool = True, **kw) -> MoveResult:
        """Run the plan in the physics and evaluate the spec's success definition (plus: no arm-piece contact
        during >= CONTACT_STEPS_FAIL control steps after a release)."""
        s = self.scene
        before = {n: s.piece_pos(n) for n in s.piece_names}
        moved = set(plan.expected.keys())
        wall0, t0 = time.time(), s.data.time
        steps0 = s.control_ticks
        log: list = []
        lift_ok, lift_z_seen = True, []
        tracking = []
        max_step_deg = 0.0
        contact_steps: Dict[str, int] = {}          # piece -> control steps in contact with the arm (any time)
        released_steps: Dict[str, int] = {}         # same, for pieces already released by this plan (not re-grasped)
        released: set = set()
        gen = self.action_generator(plan, control_hz, max_joint_speed_deg_s, log=log, abort_on_drop=abort_on_drop, **kw)
        last_wp = -1
        a_prev = np.asarray(s.get_joints(), float)
        wps: Sequence[Waypoint] = plan
        for a in gen:
            wps = self.state.get("plan", plan)
            max_step_deg = max(max_step_deg, float(np.max(np.abs(a[:5] - a_prev[:5]))))
            a_prev = a
            s.control_step(a)
            tracking.append(float(np.max(np.abs(s.get_joints()[:5] - a[:5]))))
            wi = self.state.get("waypoint", -1)
            if wi != last_wp and last_wp >= 0 and wps[last_wp].meta.get("lift_check"):
                z = s.jaw_tip_positions()[0][2]
                lift_z_seen.append((z, wps[last_wp].meta.get("lift_z", self.lift_z())))
                if z < wps[last_wp].meta.get("lift_z", self.lift_z()) - LIFT_Z_TOL:
                    lift_ok = False
            if wi != last_wp and last_wp >= 0 and wps[last_wp].kind == "open" and wps[last_wp].meta.get("piece"):
                released.add(wps[last_wp].meta["piece"])
            last_wp = wi
            current = wps[wi].meta.get("piece") if 0 <= wi < len(wps) else None
            for n in set(self._live_arm_piece_contacts()):
                if n == current:
                    continue
                contact_steps[n] = contact_steps.get(n, 0) + 1
                if n in released:
                    released_steps[n] = released_steps.get(n, 0) + 1
            if progress is not None:
                progress(self.state)
        if last_wp >= 0 and wps[last_wp].meta.get("lift_check"):
            z = s.jaw_tip_positions()[0][2]
            lift_z_seen.append((z, wps[last_wp].meta.get("lift_z", self.lift_z())))
            lift_ok = lift_ok and z >= wps[last_wp].meta.get("lift_z", self.lift_z()) - LIFT_Z_TOL
        reasons = []
        if self.state.get("abort"):
            reasons.append(self.state["abort"])
        metrics: dict = dict(piece_err_mm={}, piece_tilt_deg={}, disturbed={}, lift_z=lift_z_seen,
                             max_tracking_err_deg=float(max(tracking)) if tracking else 0.0,
                             max_action_step_deg=max_step_deg, contact_steps=contact_steps, contact_steps_after_release=released_steps,
                             plan_seconds=plan.plan_seconds, n_waypoints=len(plan), n_waypoints_executed=len(wps),
                             place_corrections=[(p, float(np.linalg.norm(sh)) * 1000, ok) for (p, sh, ok) in self.state.get("corrections", []) if p])
        for n, tgt in plan.expected.items():
            p = s.piece_pos(n)
            err = float(np.linalg.norm(p[:2] - tgt[:2]))
            tilt = s.piece_tilt_deg(n)
            metrics["piece_err_mm"][n] = err * 1000
            metrics["piece_tilt_deg"][n] = tilt
            if err > C.SUCCESS_POS_TOL:
                reasons.append(f"{n} is {err*1000:.1f} mm from its target")
            if tilt > C.SUCCESS_TILT_DEG:
                reasons.append(f"{n} tilted {tilt:.1f} deg")
        for n, p0 in before.items():
            if n in moved:
                continue
            dmove = float(np.linalg.norm(s.piece_pos(n)[:2] - p0[:2]))
            if dmove > C.DISTURB_TOL:
                metrics["disturbed"][n] = dmove * 1000
                reasons.append(f"{n} displaced {dmove*1000:.1f} mm")
        for n, c in released_steps.items():
            if c >= CONTACT_STEPS_FAIL:
                reasons.append(f"arm touched {n} after its release ({c} control steps)")
        if not lift_ok:
            reasons.append("gripper did not retreat to lift height")
        if check_watch and wps and wps[-1].kind == "watch":
            werr = float(np.max(np.abs(s.get_joints()[:5] - np.asarray(wps[-1].q6)[:5])))
            metrics["watch_err_deg"] = werr
            if werr > 3.0:
                reasons.append(f"not at the watch pose ({werr:.1f} deg)")
        if progress is not None:
            progress(self.state)
        return MoveResult(success=not reasons, reasons=reasons, metrics=metrics, timings=log,
                          sim_seconds=s.data.time - t0, control_steps=s.control_ticks - steps0)


# --------------------------------------------------------------------------- module-level API
_EXPERTS: Dict[int, Expert] = {}


def get_expert(scene) -> Expert:
    e = _EXPERTS.get(id(scene))
    if e is None or e.scene is not scene:
        e = Expert(scene)
        _EXPERTS[id(scene)] = e
    return e


def plan_move(scene, move, board: chess.Board, **kw) -> MovePlan:
    """plan_move(scene, 'e7e5' | chess.Move | ('e7','e5'), board) -> MovePlan (list of Waypoint)."""
    return get_expert(scene).plan_move(move, board, **kw)


def execute(scene, waypoints, control_hz: float = C.CONTROL_HZ, max_joint_speed_deg_s: float = C.JOINT_SPEED_DEG_S, **kw) -> MoveResult:
    plan = waypoints if isinstance(waypoints, MovePlan) else _as_plan(waypoints)
    return get_expert(scene).execute(plan, control_hz, max_joint_speed_deg_s, **kw)


def action_generator(scene, waypoints, control_hz: float = C.CONTROL_HZ, max_joint_speed_deg_s: float = C.JOINT_SPEED_DEG_S, **kw):
    """Generator of LeRobot actions, one per control step (see Expert.action_generator)."""
    plan = waypoints if isinstance(waypoints, MovePlan) else _as_plan(waypoints)
    return get_expert(scene).action_generator(plan, control_hz, max_joint_speed_deg_s, **kw)


def plan_to_watch(scene) -> MovePlan:
    """Plan the return to the watch pose from the scene's current arm pose."""
    return get_expert(scene).plan_to_watch()


def _as_plan(waypoints) -> MovePlan:
    p = MovePlan(None)
    p.extend(waypoints)
    return p


if __name__ == "__main__":
    import sys as _sys
    from chess_sim.scene import ChessScene
    sc = ChessScene()
    sc.reset(seed=0, randomise=False)
    b = chess.Board()
    uci = _sys.argv[1] if len(_sys.argv) > 1 else "e2e4"
    ex = get_expert(sc)
    plan = ex.plan_move(uci, b)
    print(f"plan {uci}: {len(plan)} waypoints, {len(plan.steps)} pick-places, planned in {plan.plan_seconds:.2f}s, notes {plan.notes}")
    for pp in plan.steps:
        print(f"  {pp.piece} -> {pp.dst_name}: grasp lat {pp.grasp_lat_deg:+.0f} tilt {pp.grasp.tilt_deg:+.1f} clr {pp.grasp_clearance*1000:.1f} mm "
              f"open {C.gripper_gap_for_percent(pp.open_pct)*1000:.0f} mm | place lat {pp.place_lat_deg:+.0f} tilt {pp.place.tilt_deg:+.1f} "
              f"clr {pp.place_clearance*1000:.1f} mm | side approach {pp.side_approach} side retreat {pp.side_retreat} slide {pp.slide}")
    res = ex.execute(plan)
    print("success", res.success, res.reasons)
    print("metrics", res.metrics)
    print(f"sim {res.sim_seconds:.1f}s, {res.control_steps} control steps")
    for lab, a, bb, w in res.timings:
        print(f"  {lab:32s} sim {bb - a:5.2f}s  wall {w:5.2f}s")
