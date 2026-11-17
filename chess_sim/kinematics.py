"""Forward / inverse kinematics of the SO-101 gripper for top-down grasps.

Conventions
-----------
* Joint vector q5 = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] in RADIANS (MJCF order,
  config.ARM_JOINTS); LeRobot degrees are converted at the API boundary by the callers (config.deg2rad / rad2deg).
* The gripperframe site (config.ARM_SITE) sits between the fingertips; its +x axis is the jaw (pointing)
  direction and its +z axis is the jaw OPENING axis: the moving jaw swings toward site +z, the fixed jaw is at
  site -z.  The fixed fingertip is at FIXED_TIP_SITE = (6.3, 0, -1.0) mm in the site frame (measured from the
  meshes at import time, see measure_tool_offsets).
* A "tool point" is any point rigidly attached to the gripper body, given in site coordinates (tool_site).  The
  default tool is the fixed fingertip.  For a grasp the planner uses the STEM point
      tool_site = FIXED_TIP_SITE + (0, 0, stem_radius + GRASP_SIDE_CLEARANCE)
  i.e. the point of the jaw frame where the stem axis should be, 2.5 mm from the fixed fingertip point (~1.5 mm
  from its collision box; only the moving jaw swings, so a piece is always referenced to the fixed finger).  A
  HELD piece sits HELD_Z_OFFSET nearer the fixed pad: place_tool_site is used for releases.
* Top-down grasp orientation: site x axis = (0, 0, -1) (jaws vertical) and site z axis = `lat`, a horizontal
  unit vector giving the direction from the fixed jaw to the moving jaw (the jaws close along -lat).
  Wrist-roll convention: with lat = the RADIAL direction (unit vector from the arm base to the target, pointing
  away from the robot) the wrist roll is ~0 (its least twisted state): the moving jaw is on the far side of the
  piece and the jaws close across the radial line.  Other lat directions are obtained by rolling the wrist; the
  planner picks the first collision-free one in the order +radial, -radial, +-tangential, diagonals.
* An optional tilt leans the jaw axis RADIALLY (in the pan plane, the only lean the arm's pitch joints can
  realise): jaw axis = sin(tilt) * radial - cos(tilt) * z.  tilt > 0 points the tips away from the base and
  extends the reach of far targets (the wrist moves inward), tilt < 0 helps targets close to the base.  Grasps
  use |tilt| <= JAW_VERTICAL_TOL_DEG (5 deg), travel at lift height may use up to TRAVEL_TILT_DEG (30 deg): with
  vertical jaws the arm reaches r = 0.104..0.303 m at the king grasp height but only r <= ~0.27 m at the lift
  height (board + 90 mm), so the far ranks need the lean while travelling.

IK: damped least squares on the 6-D error [tool position, orientation (rotation vector)], joint limits enforced
by projection each iteration, Jacobians from mujoco.mj_jac on a private MjData (the live simulation is never
touched).  Results carry the residuals so callers can accept/reject (position < 2 mm, tilt < 5 deg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np

if __package__ in (None, ""):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chess_sim import config as C
else:
    from . import config as C

# fixed fingertip in the gripperframe site frame (m); verified against the mesh at ChessScene/Kinematics creation
FIXED_TIP_SITE = np.array([0.0063, 0.0, -0.0010])
GRASP_SIDE_CLEARANCE = 0.0025       # stem axis to the fixed fingertip point minus the stem radius (m); the finger COLLISION box
                                    # extends ~1 mm beyond the mesh tip, so the real clearance is ~1.5 mm (measured, test_motion c)
# 'cylinders' collision model: grasp height fraction per type (spec: 62 % of the height, config.GRASP_HEIGHT_FRAC).
# The KING is grasped at 50 % (25 mm, still on its stem, which spans 15..50 mm): at 62 % (31 mm) its own square
# e8/d8 is out of reach at board_gap 0.090 (the folded arm needs wrist_flex > 95 deg) and a1/h1 need > 5 deg of
# tilt; no board_gap in 0.080..0.110 satisfies both ends (measured: e8 needs >= 0.0975, a1/h1 <= 5 deg tilt needs
# <= 0.090).  The 'bands' model (mesh pieces) uses config.GRASP_TABLE_BANDS instead (derived from the mesh profile).
GRASP_FRAC_OVERRIDE = {"king": 0.50}
# Where a held piece's BASE sits relative to the grasp tool point, along the opening axis (site z, + = toward the
# moving jaw): the close pushes the stem ~1 mm toward the fixed pad and the piece hangs 2 deg toward the fixed jaw
# (measured in the simulation for every type, probe in test_motion c).  The PLACE pose uses the shifted tool point
# so the base lands on the square centre.
HELD_Z_OFFSET = -0.0011
TRAVEL_TILT_DEG = 30.0              # max radial lean of the jaws while travelling at lift height (far ranks need ~20 deg)
# Max jaw lean accepted for a grasp/release.  The spec asks for vertical jaws +-5 deg (config.JAW_VERTICAL_TOL_DEG);
# the planner always prefers the smallest lean, but the far corners a1/h1 (and b1/g1 for the king) are only
# reachable with ~7.5 deg at board_gap 0.100 (wrist_flex limit), so up to GRASP_MAX_LEAN_DEG is accepted there.
GRASP_MAX_LEAN_DEG = {"default": 8.0}

# Deviations from the spec that the geometry forces (board_gap 0.100, the 0.159 m gripper and the wrist_flex limit):
# they are decisions for the user, not spec checks; test_motion labels the relaxed thresholds with these names.
SPEC_DEVIATIONS = [
    ("king grasped at 50 % of its height (25 mm) instead of 62 % (31 mm)",
     "at 31 mm e8/d8 are out of reach (wrist_flex > 95 deg) and a1/h1 need > 5 deg of lean; no board_gap in 0.080..0.110 fixes both ends"),
    ("jaw lean at grasp/release up to 8 deg (spec +-5) on a1/h1 (all types) and b1/c1/d1/e1 (queen, king); up to 15 deg for pieces that are slid",
     "the far corners are only reachable with ~7.5 deg at board_gap 0.100; a slid piece stays upright on its base so the lean is harmless"),
    ("lift height lowered to the reach ceiling (36..51 mm) on ranks 7-8 instead of board + 90 mm; c8/d8/e8/f8 pieces are dragged, not lifted",
     "the vertical jaws cannot exceed z ~0.046 m within 0.08 m of the pan axis; a side approach / slide is used there (a longer gripper or a "
     "larger board_gap would be needed to meet the spec)"),
    ("execute's 'retreated to lift height' check uses the lowered planned lift height on ranks 7-8", "same cause"),
    ("jaw lean up to 12.5 deg (expert.GRASP_MAX_LEAN_FAR) at a1/h1 when the randomised board is shifted sideways / rotated",
     "a far-reach tier used only when no pose within 8 deg exists: the 'wide' board pose pushes a far corner up to ~11 mm further out"),
]


def grasp_max_lean(piece_type: str) -> float:
    return GRASP_MAX_LEAN_DEG.get(piece_type, GRASP_MAX_LEAN_DEG["default"])


def grasp_height(piece_type: str, frac: Optional[float] = None, collision: Optional[str] = None) -> float:
    """Height of the stem point (= the fingertips) above the piece's base plane for a grasp (m).  'cylinders':
    frac x piece height (default the spec's 62 %, king 50 %); 'bands' (default): the mesh grasp table
    (config.GRASP_TABLE_BANDS, frac ignored).  The expert tries config.grasp_height_candidates in turn when a
    carry corridor is too low."""
    if (collision or C.PIECE_COLLISION_DEFAULT) == "bands":
        return C.grasp_height(piece_type, "bands")
    f = GRASP_FRAC_OVERRIDE.get(piece_type, C.GRASP_HEIGHT_FRAC) if frac is None else float(frac)
    return f * C.PIECE_SPECS[piece_type][0]
LIMIT_MARGIN_RAD = math.radians(0.5)
# seed configurations for the IK (LeRobot degrees of the 5 arm joints): known vertical-jaw poses at different
# radii; the pan is overwritten with atan2(y, x) of the target.
_SEEDS_DEG = [
    [0.0, -31.87, 45.32, 75.81, 0.0],     # tips at r = 0.16, z = 0.035 (test_scene VERTICAL_POSE)
    [0.0, 0.0, 0.0, 90.0, 0.0],           # arm straight out, wrist down: far reach
    [0.0, -60.0, 80.0, 70.0, 0.0],        # close to the base
    [0.0, 20.0, -20.0, 90.0, 0.0],        # far and high
    [0.0, -20.0, 30.0, 80.0, 0.0],
]


@dataclass
class IKResult:
    q5: np.ndarray                      # joint angles (rad), clipped to the limits
    pos_err: float                      # |tool position error| (m)
    tilt_deg: float                     # SIGNED lean of the jaw axis from straight down (deg): + = tips away from the base (radial)
    lat_err_deg: float                  # angle between the jaw opening axis and the requested lat (deg), measured in the horizontal plane
    rot_err: float                      # |rotation-vector error| (rad)
    iterations: int
    converged: bool                     # pos_err < pos_tol and rot_err < rot_tol
    at_limit: List[str] = field(default_factory=list)   # joints clipped at the last iteration
    axis_err_deg: float = 0.0           # angle between the jaw axis and the requested (possibly tilted) jaw axis (deg)

    @property
    def ok(self) -> bool:
        return self.converged

    def lerobot5(self) -> np.ndarray:
        return C.rad2deg(self.q5)


def rotation_vector_error(R: np.ndarray, Rd: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame, |e| = angle in rad) that takes R to Rd: log(Rd R^T) via quaternions.
    NOTE the classic small-angle form 0.5 * sum_i R[:,i] x Rd[:,i] is ZERO for a 180 deg flip about any axis,
    so an IK using it cannot tell the opening direction psi from psi + 180 (the jaws flipped): the exact log map
    is used instead (the DLS step is clipped by step_max, so the large error is harmless)."""
    q = np.zeros(4)
    qd = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    mujoco.mju_mat2Quat(qd, np.ascontiguousarray(Rd, dtype=float).reshape(9))
    qn = np.zeros(4)
    mujoco.mju_negQuat(qn, q)
    qe = np.zeros(4)
    mujoco.mju_mulQuat(qe, qd, qn)            # Rd * R^T
    if qe[0] < 0:
        qe = -qe
    v = qe[1:]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return np.zeros(3)
    ang = 2.0 * math.atan2(nv, float(qe[0]))
    return v / nv * ang


def desired_rotation(lat_world, tilt_deg: float = 0.0, tilt_dir=None) -> np.ndarray:
    """3x3 site rotation of a top-down grasp.  tilt_deg = 0: site x (jaw axis) = straight down, site z (opening
    axis, fixed jaw -> moving jaw) = the horizontal unit vector `lat`.  With tilt_deg != 0 the jaw axis leans by
    tilt_deg from the vertical toward the horizontal unit vector `tilt_dir` (the RADIAL direction u of the arm's
    pitch plane, the only lean it can realise): x = sin(t) u - cos(t) z.  The opening axis keeps its roll angle
    psi relative to the radial direction: with a = cos(t) u + sin(t) z (the in-plane vector perpendicular to x)
    and b = z x u (tangential), site z = cos(psi) a + sin(psi) b where psi = angle from u to lat.  This is
    exactly the family of orientations the 5-DOF arm can produce, so interpolating (tilt, psi) along a path
    always gives reachable orientations."""
    lat = np.asarray(lat_world, float).copy()
    lat[2] = 0.0
    n = np.linalg.norm(lat)
    lat = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else lat / n
    u = np.array([1.0, 0.0, 0.0]) if tilt_dir is None else np.asarray(tilt_dir, float).copy()
    u[2] = 0.0
    u /= np.linalg.norm(u) + 1e-12
    zhat = np.array([0.0, 0.0, 1.0])
    b = np.cross(zhat, u)
    psi = math.atan2(float(lat @ b), float(lat @ u))
    t = math.radians(tilt_deg)
    x = math.sin(t) * u - math.cos(t) * zhat
    a = math.cos(t) * u + math.sin(t) * zhat
    z = math.cos(psi) * a + math.sin(psi) * b
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def frame_params(R: np.ndarray, radial_dir) -> Tuple[float, float]:
    """Inverse of desired_rotation: (tilt_deg, psi_deg) of a site rotation given the radial direction u.
    psi is the roll of the opening axis from u (0 = opening axis radial outward)."""
    u = np.asarray(radial_dir, float).copy()
    u[2] = 0.0
    u /= np.linalg.norm(u) + 1e-12
    zhat = np.array([0.0, 0.0, 1.0])
    x = R[:, 0]
    tilt = math.degrees(math.atan2(float(x @ u), float(-x[2])))
    t = math.radians(tilt)
    a = math.cos(t) * u + math.sin(t) * zhat
    b = np.cross(zhat, u)
    psi = math.degrees(math.atan2(float(R[:, 2] @ b), float(R[:, 2] @ a)))
    return tilt, psi


PAN_AXIS_XY = np.array([0.0388353, 0.0])     # shoulder_pan axis in the world (from the MJCF); Kinematics reads it from the model


def radial_lat(target_xyz, sign: float = 1.0, origin_xy=None) -> np.ndarray:
    """Horizontal unit vector from the shoulder-pan axis (default PAN_AXIS_XY, NOT the base centre: the pan axis
    is 39 mm in front of it) toward the target (sign=+1) or back (sign=-1).  This is the arm's pitch-plane
    direction, the only direction the jaw axis can lean in; it is also the wrist-roll = ~0 opening direction."""
    o = PAN_AXIS_XY if origin_xy is None else np.asarray(origin_xy, float)
    v = np.array([target_xyz[0] - o[0], target_xyz[1] - o[1], 0.0])
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([sign, 0.0, 0.0])
    return sign * v / n


def rotate_lat(lat, deg: float) -> np.ndarray:
    """Rotate a horizontal direction about +z by deg (positive = toward the robot's left / +y side for +x)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * lat[0] - s * lat[1], s * lat[0] + c * lat[1], 0.0])


class Kinematics:
    """FK/IK for one MjModel.  Owns a scratch MjData: nothing here modifies the caller's MjData."""

    def __init__(self, model: mujoco.MjModel, collision: Optional[str] = None):
        """collision: the piece collision model the grasp tool points are computed for ('bands' | 'cylinders',
        default config.PIECE_COLLISION_DEFAULT; the Expert passes its scene's cfg.piece_collision)."""
        self.model = model
        self.data = mujoco.MjData(model)
        self.collision = collision or C.PIECE_COLLISION_DEFAULT
        m = model
        self.joint_ids = [m.joint(n).id for n in C.JOINT_NAMES]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in self.joint_ids])
        self.arm_dofs = self.dof_adr[:5]
        self.lo = np.array([m.jnt_range[j][0] for j in self.joint_ids[:5]]) + LIMIT_MARGIN_RAD
        self.hi = np.array([m.jnt_range[j][1] for j in self.joint_ids[:5]]) - LIMIT_MARGIN_RAD
        self.site_id = m.site(C.ARM_SITE).id
        self.body_id = m.site_bodyid[self.site_id]
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self.fixed_tip_site = FIXED_TIP_SITE.copy()
        self.seeds = [np.concatenate([C.deg2rad(s)]) for s in _SEEDS_DEG]
        # the shoulder-pan axis position (radial directions / tilt planes are measured from it)
        self._set(np.zeros(5))
        self.pan_axis_xy = self.data.xanchor[m.joint("shoulder_pan").id][:2].copy()
        global PAN_AXIS_XY
        PAN_AXIS_XY = self.pan_axis_xy.copy()
        # the wrist-roll axis (parallel to the jaw axis) passes 7.5 mm from the site: its offset in site coords
        # (the radial direction of a pose is measured from the pan axis to a point ON the roll axis)
        jr = m.joint("wrist_roll").id
        anchor, axis = self.data.xanchor[jr].copy(), self.data.xaxis[jr].copy()
        p = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        foot = anchor + ((p - anchor) @ axis) * axis
        self.roll_axis_site = R.T @ (foot - p)          # ~ (0, 0, +0.0075)
        self.roll_axis_site[0] = 0.0
        self._last_q5: Optional[np.ndarray] = None
        # wrist roll = psi + roll_offset_deg (psi = opening direction relative to the radial direction, see
        # frame_params); measured at the vertical seed pose (the offset varies by +-3 deg with the roll because the
        # roll axis is 7.5 mm off the tool point; the IK corrects the rest)
        p0, R0, t0 = self.fk(self.seeds[0])
        self.roll_offset_deg = math.degrees(self.seeds[0][4]) - frame_params(R0, radial_lat(t0, 1.0, self.pan_axis_xy))[1]

    # ------------------------------------------------------------------ FK
    def _set(self, q5, gripper_q: Optional[float] = None) -> None:
        d = self.data
        d.qpos[self.qpos_adr[:5]] = np.asarray(q5, float)
        if gripper_q is not None:
            d.qpos[self.qpos_adr[5]] = gripper_q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def fk(self, q5, tool_site=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(site position, site rotation 3x3, tool point) for joint angles q5 (rad).  tool_site defaults to the
        fixed fingertip."""
        self._set(q5)
        p = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        t = self.fixed_tip_site if tool_site is None else np.asarray(tool_site, float)
        return p, R, p + R @ t

    def tool_pos(self, q5, tool_site=None) -> np.ndarray:
        return self.fk(q5, tool_site)[2]

    def jaw_axis(self, q5) -> np.ndarray:
        """Unit jaw (pointing) direction = site x axis."""
        return self.fk(q5)[1][:, 0]

    def tilt_deg(self, q5) -> float:
        """Angle of the jaw axis from straight down (deg)."""
        x = self.jaw_axis(q5)
        return math.degrees(math.acos(max(-1.0, min(1.0, -x[2]))))

    def lat_axis(self, q5) -> np.ndarray:
        """Unit opening axis (fixed jaw -> moving jaw) = site z axis."""
        return self.fk(q5)[1][:, 2]

    # ------------------------------------------------------------------ IK
    def _jacobian(self, point_world) -> np.ndarray:
        mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, np.asarray(point_world, float), self.body_id)
        return np.vstack([self._jacp[:, self.arm_dofs], self._jacr[:, self.arm_dofs]])

    def solve(self, target_xyz, lat_world, tool_site=None, tilt_deg: float = 0.0, q_init=None,
              max_iter: int = 120, damping: float = 0.02, w_rot: float = 0.1, step_max: float = 0.35,
              pos_tol: float = 1e-4, rot_tol: float = 1e-3, seeds: bool = True, tilt_dir=None) -> IKResult:
        """Damped-least-squares IK: put the tool point at target_xyz with the top-down orientation
        desired_rotation(lat_world, tilt_deg).  Tries q_init (default: the last solution) and, if that does not
        converge, the canonical seeds with the pan set toward the target.  Returns the best result (lowest
        weighted residual) with converged=True when pos_err < pos_tol and rot_err < rot_tol."""
        target = np.asarray(target_xyz, float)
        tool = self.fixed_tip_site if tool_site is None else np.asarray(tool_site, float)
        if tilt_dir is None:
            tilt_dir = self.radial_dir(target, lat_world, tool, tilt_deg)
        Rd = desired_rotation(lat_world, tilt_deg, tilt_dir)
        pan = math.atan2(target[1] - self.pan_axis_xy[1], target[0] - self.pan_axis_xy[0])
        roll_seed = self.roll_for_psi(frame_params(Rd, tilt_dir)[1])
        starts = []
        if q_init is not None:
            starts.append(np.asarray(q_init, float).copy())
        elif self._last_q5 is not None:
            starts.append(self._last_q5.copy())
        if seeds:
            for s in self.seeds:
                q = s.copy()
                q[0] = pan
                if roll_seed is not None:
                    q[4] = roll_seed
                starts.append(q)
        if roll_seed is None and not starts:
            starts.append(self.seeds[0].copy())
        best: Optional[IKResult] = None
        for q0 in starts:
            r = self._solve_from(q0, target, Rd, tool, max_iter, damping, w_rot, step_max, pos_tol, rot_tol, tilt_dir)
            if best is None or (r.pos_err + 0.05 * r.rot_err) < (best.pos_err + 0.05 * best.rot_err):
                best = r
            if r.converged:
                break
        assert best is not None
        if best.converged:
            self._last_q5 = best.q5.copy()
        return best

    def roll_for_psi(self, psi_deg: float) -> Optional[float]:
        """Wrist-roll angle (rad) inside the joint range that realises the opening direction psi (deg, relative
        to the radial direction; roll ~= psi + ROLL_OFFSET_DEG), or None when psi falls in the roll dead zone
        (the range -157..+162 deg leaves ~40 deg around psi = 180 unreachable)."""
        base = math.radians(psi_deg + self.roll_offset_deg)
        cands = [base + 2 * math.pi * j for j in (-1, 0, 1)]
        inside = [c for c in cands if self.lo[4] - 1e-9 <= c <= self.hi[4] + 1e-9]
        if not inside:
            return None
        return min(inside, key=lambda c: abs(c))

    def psi_reachable(self, psi_deg: float) -> bool:
        return self.roll_for_psi(psi_deg) is not None

    def radial_dir(self, target, lat_world, tool_site, tilt_deg: float = 0.0) -> np.ndarray:
        """Radial (pitch-plane) direction of a pose: from the pan axis to the point of the wrist-roll axis that
        corresponds to the tool point (one fixed-point refinement of the lateral tool offset)."""
        target = np.asarray(target, float)
        u = radial_lat(target, 1.0, self.pan_axis_xy)
        off = np.asarray(tool_site, float) - self.roll_axis_site     # lateral offset of the tool from the roll axis
        for _ in range(2):
            R = desired_rotation(lat_world, tilt_deg, u)
            axis_pt = target - R[:, 1] * off[1] - R[:, 2] * off[2]
            u = radial_lat(axis_pt, 1.0, self.pan_axis_xy)
        return u

    def _solve_from(self, q0, target, Rd, tool, max_iter, damping, w_rot, step_max, pos_tol, rot_tol, tilt_dir=None) -> IKResult:
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        u_tilt = radial_lat(target, 1.0, self.pan_axis_xy) if tilt_dir is None else np.asarray(tilt_dir, float)
        lam2 = damping ** 2
        W = np.diag([1, 1, 1, w_rot, w_rot, w_rot])
        it = 0
        pos_err = rot_err = np.inf
        hist: List[float] = []
        for it in range(1, max_iter + 1):
            self._set(q)
            p = self.data.site_xpos[self.site_id]
            R = self.data.site_xmat[self.site_id].reshape(3, 3)
            tp = p + R @ tool
            e_p = target - tp
            e_r = rotation_vector_error(R, Rd)
            pos_err, rot_err = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if pos_err < pos_tol and rot_err < rot_tol:
                break
            hist.append(pos_err + w_rot * rot_err)
            if len(hist) > 12 and hist[-1] > 0.995 * hist[-12]:      # stalled (unreachable / at a limit): give up early
                break
            J = self._jacobian(tp)
            e = np.concatenate([e_p, e_r])
            Jw = W @ J
            ew = W @ e
            # dq = Jw^T (Jw Jw^T + lam2 I)^-1 ew
            A = Jw @ Jw.T + lam2 * np.eye(6)
            dq = Jw.T @ np.linalg.solve(A, ew)
            n = np.linalg.norm(dq)
            if n > step_max:
                dq *= step_max / n
            q = np.clip(q + dq, self.lo, self.hi)
        self._set(q)
        R = self.data.site_xmat[self.site_id].reshape(3, 3)
        x = R[:, 0]
        axis_err = math.degrees(math.acos(max(-1.0, min(1.0, float(x @ Rd[:, 0])))))
        # signed lean in the radial (pitch) plane: + = tips away from the base, - = toward it (see desired_rotation)
        tilt = math.degrees(math.atan2(float(x[0] * u_tilt[0] + x[1] * u_tilt[1]), float(-x[2])))
        lat_cur = R[:, 2].copy()
        lat_cur[2] = 0.0
        lat_d = Rd[:, 2].copy()
        lat_d[2] = 0.0
        if np.linalg.norm(lat_cur) > 1e-9 and np.linalg.norm(lat_d) > 1e-9:
            lat_err = math.degrees(math.acos(max(-1.0, min(1.0, float(lat_cur @ lat_d) / (np.linalg.norm(lat_cur) * np.linalg.norm(lat_d))))))
        else:
            lat_err = float("nan")
        at_limit = [C.ARM_JOINTS[i] for i in range(5) if q[i] <= self.lo[i] + 1e-9 or q[i] >= self.hi[i] - 1e-9]
        return IKResult(q5=q, pos_err=pos_err, tilt_deg=tilt, lat_err_deg=lat_err, rot_err=rot_err, iterations=it,
                        converged=bool(pos_err < pos_tol and rot_err < rot_tol), at_limit=at_limit, axis_err_deg=axis_err)

    def solve_R(self, target_xyz, R_des, tool_site=None, q_init=None, max_iter: int = 120, damping: float = 0.02,
                w_rot: float = 0.1, step_max: float = 0.35, pos_tol: float = 1e-4, rot_tol: float = 1e-3) -> IKResult:
        """IK for an explicit site rotation (used to reproduce a known pose such as the watch pose)."""
        target = np.asarray(target_xyz, float)
        tool = self.fixed_tip_site if tool_site is None else np.asarray(tool_site, float)
        q0 = self._last_q5 if q_init is None else q_init
        if q0 is None:
            q0 = self.seeds[0].copy()
            q0[0] = math.atan2(target[1] - self.pan_axis_xy[1], target[0] - self.pan_axis_xy[0])
        r = self._solve_from(q0, target, np.asarray(R_des, float), tool, max_iter, damping, w_rot, step_max, pos_tol, rot_tol)
        if r.converged:
            self._last_q5 = r.q5.copy()
        return r

    def solve_reach(self, target_xyz, lat_world, tool_site=None, max_tilt_deg: float = 0.0, tilt_step: float = 2.5,
                    q_init=None, prefer_sign: float = 0.0, tilt_hint: Optional[float] = None, **kw) -> IKResult:
        """IK that, when the exact vertical pose is unreachable, retries with the jaw axis leaning radially on the
        grid 0, +-tilt_step, +-2 tilt_step, ... up to +-max_tilt_deg (positive = tips leaning away from the base,
        which extends the reach of far targets; negative = toward the base, for targets close to the base).  The
        candidates are tried in order of |tilt - tilt_hint| (hint 0 = smallest lean first; a path planner passes
        the previous knot's tilt for continuity), ties broken by prefer_sign.  Returns the first converged result
        or the best attempt."""
        hint = 0.0 if tilt_hint is None else float(tilt_hint)
        grid = [0.0]
        k = 1
        while k * tilt_step < max_tilt_deg - 1e-9:
            grid += [k * tilt_step, -k * tilt_step]
            k += 1
        if max_tilt_deg > 1e-9:
            grid += [max_tilt_deg, -max_tilt_deg]
        grid.sort(key=lambda t: (round(abs(t - hint), 6), -prefer_sign * t))
        best = None
        for tilt in grid:
            r = self.solve(target_xyz, lat_world, tool_site, tilt_deg=tilt, q_init=q_init, **kw)
            if best is None or (r.pos_err + 0.05 * r.rot_err) < (best.pos_err + 0.05 * best.rot_err):
                best = r
            if r.converged:
                return r
        return best

    def solve_batch(self, targets, lats, tool_sites=None, tilt_deg: float = 0.0, **kw) -> List[IKResult]:
        """IK for many targets (each warm-started from the previous solution, with the seed fallback)."""
        out = []
        n = len(targets)
        lats = list(lats) if not isinstance(lats, np.ndarray) or lats.ndim > 1 else [lats] * n
        if len(lats) == 1 and n > 1:
            lats = lats * n
        for i in range(n):
            ts = None if tool_sites is None else tool_sites[i]
            out.append(self.solve(targets[i], lats[i], ts, tilt_deg=tilt_deg, **kw))
        return out

    # ------------------------------------------------------------------ helpers
    def grasp_tool_site(self, piece_type: str, theta_fixed_deg: Optional[float] = None) -> np.ndarray:
        """Tool point (site frame) that must coincide with the piece axis at the grasp: the fixed fingertip plus
        (the piece's extent toward the fixed finger at or above the grasp height + GRASP_SIDE_CLEARANCE) along the
        opening axis (site +z), so the open fixed finger descends beside the widest part above the tips (the
        collar / head / crown; equal to the stem radius for 'cylinders').  theta_fixed_deg: the direction from the
        piece axis to the fixed finger in the piece frame (only matters for the knight: muzzle 10.5 mm, mane 9.2,
        across 5.2)."""
        r = C.piece_side_extent(piece_type, theta_fixed_deg, above=True, collision=self.collision)
        return self.fixed_tip_site + np.array([0.0, 0.0, r + GRASP_SIDE_CLEARANCE])

    def place_tool_site(self, piece_type: str, theta_fixed_deg: Optional[float] = None) -> np.ndarray:
        """Tool point (site frame) that coincides with the axis of a HELD piece of this type: the close pushes the
        piece against the fixed pad, so its axis sits (its extent toward the fixed finger inside the pad window +
        GRASP_SIDE_CLEARANCE + HELD_Z_OFFSET) from the fingertip: the grasp tool point shifted by HELD_Z_OFFSET for
        'cylinders', further by the envelope-minus-stall difference for 'bands' (0.1..1.5 mm per mesh type)."""
        r = C.piece_side_extent(piece_type, theta_fixed_deg, above=False, collision=self.collision)
        return self.fixed_tip_site + np.array([0.0, 0.0, r + GRASP_SIDE_CLEARANCE + HELD_Z_OFFSET])

    def grasp_height(self, piece_type: str, frac: Optional[float] = None) -> float:
        return grasp_height(piece_type, frac, self.collision)

    def tips_world(self, q5, gripper_q: float):
        """Approximate (fixed tip, moving tip) world positions for a gripper angle, from the jaw model: the moving
        tip is placed at the measured gap along the opening axis (the true tip also rises slightly)."""
        p, R, tf = self.fk(q5)
        gap = C.gripper_gap(gripper_q)
        return tf, tf + R[:, 2] * gap

    def q5_lerobot(self, q5) -> np.ndarray:
        return C.rad2deg(q5)

    def joint_speed_ok(self, q_a, q_b, dt: float, max_deg_s: float = C.JOINT_SPEED_DEG_S) -> bool:
        return bool(np.max(np.abs(np.asarray(q_b) - np.asarray(q_a))) <= math.radians(max_deg_s) * dt + 1e-9)


def measure_tool_offsets(scene) -> dict:
    """Measure the fixed fingertip offset (site frame) from the jaw meshes of a ChessScene and compare it with
    FIXED_TIP_SITE.  Returns dict(fixed_tip_site, error_m)."""
    d = scene.data
    q_saved = d.qpos.copy()
    scene.set_joints(np.zeros(6))
    tf, tm = scene.jaw_tip_positions()
    R, p = scene.gripper_mat(), scene.gripper_pos()
    off = R.T @ (tf - p)
    d.qpos[:] = q_saved
    mujoco.mj_forward(scene.model, d)
    return dict(fixed_tip_site=off, error_m=float(np.linalg.norm(off - FIXED_TIP_SITE)))


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chess_sim.scene import ChessScene
    s = ChessScene()
    k = Kinematics(s.model)
    print("tool offset check:", measure_tool_offsets(s))
    for sq in ("e8", "a8", "h8", "e1", "a1", "h1", "d4"):
        t = s.square_xyz(sq, C.grasp_height("king"))
        r = k.solve(t, radial_lat(t), k.grasp_tool_site("king"))
        print(sq, np.round(t, 3), "pos_err mm %.2f tilt %.2f lat_err %.2f it %d conv %s limits %s" %
              (r.pos_err * 1000, r.tilt_deg, r.lat_err_deg, r.iterations, r.converged, r.at_limit), np.round(r.lerobot5(), 1))
