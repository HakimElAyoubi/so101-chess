"""ChessScene: MuJoCo model of the SO-101 + chess board, with helpers for squares, pieces,
FEN positions, rendering, projection, randomised resets and the wrist-camera watch pose.

Joint convention (LeRobot): set_joints / get_joints use a 6-vector
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] in degrees + gripper in percent.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import mujoco
import numpy as np

if __package__ in (None, ""):          # run as a plain script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chess_sim import config as C
    from chess_sim.progress import ProgressBar
    from chess_sim.build_scene import (write_scene, embedded_config, embedded_jaw_geometry, jaw_geometry_agrees,
                                       jaw_tip_vertices, jaw_gap_from_mesh, jaw_clearance_from_mesh, _mesh_world_vertices,
                                       finger_geom_ids, jaw_clearance_from_boxes)
else:
    from . import config as C
    from .progress import ProgressBar
    from .build_scene import (write_scene, embedded_config, embedded_jaw_geometry, jaw_geometry_agrees,
                              jaw_tip_vertices, jaw_gap_from_mesh, jaw_clearance_from_mesh, _mesh_world_vertices,
                                       finger_geom_ids, jaw_clearance_from_boxes)

START_FEN = chess.STARTING_FEN


class SceneConfigMismatch(ValueError):
    """The scene XML was generated from a different ChessConfig than the one passed to ChessScene."""


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return np.concatenate([[math.cos(angle / 2)], axis * math.sin(angle / 2)])


def _quat_yaw(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class ChessScene:
    """Load (or build) the chess scene and expose square/piece/camera helpers."""

    def __init__(self, cfg: Optional[C.ChessConfig] = None, xml_path: Optional[str] = None, rebuild=False):
        """cfg: ChessConfig (default DEFAULT_CONFIG).  xml_path: scene XML (default cfg.scene_xml_path(), i.e. a
        per-config file; generated when missing).  rebuild: True regenerates the XML, 'auto' regenerates it only
        when its embedded config record does not match cfg, False raises SceneConfigMismatch on a mismatch."""
        self.cfg = cfg or C.DEFAULT_CONFIG
        xml_path = xml_path or self.cfg.scene_xml_path()
        self.xml_path = xml_path
        if rebuild is True or not os.path.exists(xml_path):
            write_scene(self.cfg, xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        if not self._config_matches(self.model):
            if rebuild == "auto":
                write_scene(self.cfg, xml_path)
                self.model = mujoco.MjModel.from_xml_path(xml_path)
            else:
                raise SceneConfigMismatch(
                    f"{xml_path} was built from config {embedded_config(self.model)} but ChessScene got "
                    f"{self.cfg.scene_key()} (or a different drop-zone layout DROP_GAP / DROP_X_SHIFT). Pass rebuild=True (or 'auto') to regenerate it, or omit xml_path so the "
                    f"per-config file {self.cfg.scene_xml_path()} is used.")
        self._check_jaw_model(self.model, xml_path)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._jaw_tips: Optional[dict] = None          # lazily built by jaw_gap()/jaw_clearance_mesh()
        self._finger_ids: Optional[Tuple[list, list]] = None   # finger box geoms, built by jaw_clearance()
        self._ctrl_ticks = 0                           # control ticks since the grid anchor (exact 30 Hz grid)
        self._grid_t0 = 0.0                            # sim time the 30 Hz control grid is anchored at (last reset / settle)
        # ids
        self.joint_ids = [m.joint(n).id for n in C.JOINT_NAMES]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_adr = np.array([m.jnt_dofadr[j] for j in self.joint_ids])
        self.act_ids = [m.actuator(n).id for n in C.JOINT_NAMES]
        self.joint_range = np.array([m.jnt_range[j] for j in self.joint_ids])
        self.site_id = m.site(C.ARM_SITE).id
        self.board_id = m.body("board").id
        self.drop_zone_id = m.body("drop_zone").id
        self.spare_zone_id = m.body("spare_zone").id
        self.table_geom_id = m.geom("table").id
        self.arm_body_ids = {m.body(n).id for n in C.ARM_BODIES}
        self.cam_ids = {n: m.camera(n).id for n in C.CAMERA_NAMES}
        self._cam_body = m.body("wrist_camera").id
        self.occlusion_grids = (5, 17)      # ray grids used by the watch-pose occlusion test
        self.piece_names: List[str] = C.piece_body_names()
        self.piece_body_id: Dict[str, int] = {n: m.body(n).id for n in self.piece_names}
        self.piece_qadr: Dict[str, int] = {n: m.jnt_qposadr[m.joint(f"{n}_joint").id] for n in self.piece_names}
        self.piece_vadr: Dict[str, int] = {n: m.jnt_dofadr[m.joint(f"{n}_joint").id] for n in self.piece_names}
        self.piece_type: Dict[str, str] = {n: n.split("_")[1] for n in self.piece_names}
        self.piece_colour: Dict[str, str] = {n: n.split("_")[0] for n in self.piece_names}
        self.corner_sites = {n: m.site(n).id for n in ("corner_a8", "corner_h8", "corner_a1", "corner_h1")}
        # nominal (un-randomised) model parameters
        self._nominal = dict(
            body_pos=m.body_pos.copy(), body_quat=m.body_quat.copy(), light_pos=m.light_pos.copy(),
            light_dir=m.light_dir.copy(), light_diffuse=m.light_diffuse.copy(),
            geom_rgba=m.geom_rgba.copy(), cam_pos=m.cam_pos.copy(), cam_quat=m.cam_quat.copy(),
            head_ambient=np.array(m.vis.headlight.ambient, float), head_diffuse=np.array(m.vis.headlight.diffuse, float),
        )
        # minimum centre distance between two pieces standing on the board so that their mesh envelopes (the
        # real piece footprints at every height, config.PIECE_ENVELOPE) keep randomisation.piece_min_gap_m apart
        self._pair_min_dist = {(a, b): self._envelope_sum(a, b) for a in C.PIECE_TYPES for b in C.PIECE_TYPES}
        self._renderers: Dict[Tuple[int, int], mujoco.Renderer] = {}
        self._watch: Optional[dict] = None
        self._board_range_override = None              # scan_poses: cfg.scan_board_range instead of watch_board_range
        self.watch_region = tuple(self.cfg.watch_region)   # board region the watch metrics look at (scan views: others)
        self.last_cam_redraws = 0                      # camera re-draws of the last randomised reset (keep_board_visible)
        self.rng = np.random.default_rng(0)
        mujoco.mj_forward(m, self.data)
        # arm contact pairs present at the zero pose with the gripper closed (ignored by the collision check)
        self.set_joints(np.zeros(6))
        self._baseline_pairs = self._arm_contact_pairs()
        self._arm_verts = None          # lazily built by _arm_silhouette_vertices()

    @staticmethod
    def _check_jaw_model(model: mujoco.MjModel, xml_path: str) -> None:
        """The jaw kinematics measured from the meshes at build time (embedded in the XML) must agree with the
        constants in config (JAW_MODEL / JAW_CLEARANCE) that the gripper helpers use; otherwise the arm model
        changed and config.py is stale (regenerate the constants with python -m chess_sim.build_scene --print-jaw)."""
        rec = embedded_jaw_geometry(model)
        if rec is None:
            raise SceneConfigMismatch(f"{xml_path} has no jaw-geometry record (old build): rebuild it with "
                                      f"python -m chess_sim.build_scene (or ChessScene(rebuild=True)).")
        cfg_rec = dict(C.JAW_MODEL, table_q=list(C.JAW_CLEARANCE_Q), table_clearance=list(C.JAW_CLEARANCE))
        if not jaw_geometry_agrees(rec, cfg_rec):
            raise SceneConfigMismatch(
                f"jaw geometry embedded in {xml_path} (measured from the arm meshes) differs from config.JAW_MODEL / "
                f"JAW_CLEARANCE by more than 0.2 mm: the arm model changed; update config.py from "
                f"'python -m chess_sim.build_scene --print-jaw'.")

    def _config_matches(self, model: mujoco.MjModel) -> bool:
        """True if the XML's embedded config record equals cfg.scene_key() AND the compiled geometry agrees
        (board body position, board thickness, wrist fovy): an XML without a record never matches."""
        rec = embedded_config(model)
        if rec != self.cfg.scene_key():
            return False
        bid = model.body("board").id
        box = model.geom("board_box")
        cam = model.camera(C.WRIST_CAM)
        dz = model.geom("drop_cell_00").pos
        return (np.allclose(model.body_pos[bid], self.cfg.board_centre, atol=1e-6)
                and np.allclose(dz[:2], self.cfg.drop_cell_xy(0, 0), atol=1e-6)      # drop-zone layout (DROP_GAP / DROP_X_SHIFT)
                and abs(2 * box.size[2] - self.cfg.board_thickness) < 1e-6
                and abs(model.cam_fovy[cam.id] - self.cfg.wrist_fovy) < 1e-6)

    # ------------------------------------------------------------------ geometry
    # The board pose is a planar rigid transform of the board body: translation board_offset (world) + yaw
    # board_yaw (rad, about the vertical axis through the board centre).  "Board coordinates" below are the
    # NOMINAL world coordinates of config.square_xy / board_centre (x forward, y left, the un-randomised layout);
    # board_to_world / world_to_board convert, and every square <-> world conversion goes through them.
    @property
    def board_offset(self) -> np.ndarray:
        """Current board displacement (randomisation) from its nominal position (world xyz, m)."""
        return self.model.body_pos[self.board_id] - self._nominal["body_pos"][self.board_id]

    @property
    def board_yaw(self) -> float:
        """Current board yaw (rad) relative to the nominal layout, about the vertical axis through its centre."""
        q = self.model.body_quat[self.board_id]
        return float(2.0 * math.atan2(q[3], q[0]))

    @property
    def board_yaw_deg(self) -> float:
        return math.degrees(self.board_yaw)

    def board_pose(self) -> Tuple[float, float, float]:
        """(dx m, dy m, yaw deg) of the board relative to its nominal pose."""
        o = self.board_offset
        return float(o[0]), float(o[1]), self.board_yaw_deg

    def set_board_pose(self, dx: float = 0.0, dy: float = 0.0, yaw_deg: float = 0.0, forward: bool = True) -> None:
        """Place the board body at (nominal centre + (dx, dy)) with yaw_deg about its centre.  Pieces are NOT moved
        (call set_position / reset afterwards); the textured top, the corner sites and the collision box follow."""
        m = self.model
        m.body_pos[self.board_id] = self._nominal["body_pos"][self.board_id] + np.array([dx, dy, 0.0])
        m.body_quat[self.board_id] = _quat_yaw(math.radians(yaw_deg))
        if forward:
            mujoco.mj_forward(m, self.data)

    def _board_rot(self) -> np.ndarray:
        c, s_ = math.cos(self.board_yaw), math.sin(self.board_yaw)
        return np.array([[c, -s_], [s_, c]])

    def board_to_world(self, xy) -> np.ndarray:
        """Nominal-layout (board) xy -> world xy under the current board pose."""
        c_nom = self.cfg.board_centre[:2]
        return c_nom + self.board_offset[:2] + self._board_rot() @ (np.asarray(xy, float)[:2] - c_nom)

    def world_to_board(self, xy) -> np.ndarray:
        """World xy -> nominal-layout (board) xy under the current board pose (inverse of board_to_world)."""
        c_nom = self.cfg.board_centre[:2]
        return c_nom + self._board_rot().T @ (np.asarray(xy, float)[:2] - c_nom - self.board_offset[:2])

    @property
    def board_top_z(self) -> float:
        return float(self.cfg.board_thickness + self.board_offset[2])

    def square_xyz(self, square: str, z_offset: float = 0.0) -> np.ndarray:
        """World xyz of a square centre on the playing surface (+ z_offset), under the current board pose."""
        f, r = C.square_index(square)
        xy = self.board_to_world(self.cfg.square_xy(f, r))
        return np.array([xy[0], xy[1], self.board_top_z + z_offset])

    @property
    def drop_zone_offset(self) -> np.ndarray:
        """Jitter of the drop-zone pad (world xyz): it stays on the table, independent of the board pose."""
        return self.model.body_pos[self.drop_zone_id] - self._nominal["body_pos"][self.drop_zone_id]

    @property
    def spare_zone_offset(self) -> np.ndarray:
        """Jitter of the spare-queen pad (world xyz): it stays on the table, independent of the board pose."""
        return self.model.body_pos[self.spare_zone_id] - self._nominal["body_pos"][self.spare_zone_id]

    def drop_cell_xyz(self, i: int, j: int, z_offset: float = 0.0) -> np.ndarray:
        x, y = self.cfg.drop_cell_xy(i, j)
        return np.array([x, y, z_offset]) + self.drop_zone_offset

    def spare_xyz(self, colour: str, z_offset: float = 0.0) -> np.ndarray:
        x, y = self.cfg.spare_cell_xy(0 if colour == "w" else 1)
        return np.array([x, y, z_offset]) + self.spare_zone_offset

    def board_corners(self) -> np.ndarray:
        """4x3 world coordinates of the playing-surface corners (a8, h8, a1, h1 order) under the current board
        pose (computed from the pose, so it is valid right after set_board_pose without a forward pass)."""
        bc = self.cfg.board_centre
        h = C.BOARD_SIZE / 2.0
        out = []
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            xy = self.board_to_world((bc[0] + sx * h, bc[1] + sy * h))
            out.append([xy[0], xy[1], self.board_top_z])
        return np.array(out)

    def board_corners_at(self, dx: float, dy: float, yaw_deg: float, region: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Board corners (a8, h8, a1, h1) for a hypothetical board pose (the model is not changed).  region (r0, r1):
        only the band of the board between fractions r0..r1 of its depth (0 = near edge / rank 8, 1 = far edge /
        rank 1), e.g. (0, 0.6) for a scan view of the near ranks; a 4-tuple (r0, r1, f0, f1) also limits the files
        (0 = the a-file edge, -y; 1 = the h-file edge); default self.watch_region (the whole board)."""
        bc = self.cfg.board_centre
        h = C.BOARD_SIZE / 2.0
        r0, r1, f0, f1 = self._region4(region if region is not None else self.watch_region)
        c, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        out = []
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            lx, ly = (-h + 2 * h * (r0 if sx < 0 else r1)), (-h + 2 * h * (f0 if sy < 0 else f1))
            out.append([bc[0] + dx + c * lx - s_ * ly, bc[1] + dy + s_ * lx + c * ly, self.cfg.board_thickness])
        return np.array(out)

    @staticmethod
    def _is_drop(region) -> bool:
        return isinstance(region, str) and region == "drop"

    def target_corners_at(self, dx: float, dy: float, yaw_deg: float, region=None) -> np.ndarray:
        """Corners (4x3, same order as board_corners) of the current watch target: a board region (board pose
        dx, dy, yaw) or, for region 'drop', the drop-zone cells (pad shifted by dx, dy; yaw ignored), top surface."""
        region = self.watch_region if region is None else region
        if not self._is_drop(region):
            return self.board_corners_at(dx, dy, yaw_deg, region)
        x0 = self.cfg.board_gap + C.DROP_X_SHIFT
        y0 = -(C.BOARD_SIZE / 2 + C.DROP_GAP)
        L = C.DROP_CELLS * C.SQUARE
        z = 0.0015
        return np.array([[x0 + dx, y0 + dy, z], [x0 + dx, y0 - L + dy, z], [x0 + L + dx, y0 + dy, z], [x0 + L + dx, y0 - L + dy, z]])

    @staticmethod
    def _region4(region) -> Tuple[float, float, float, float]:
        """(r0, r1) depth band or (r0, r1, f0, f1) depth x file band -> 4-tuple of board fractions."""
        r = tuple(float(v) for v in region)
        return r if len(r) == 4 else (r[0], r[1], 0.0, 1.0)

    def nearest_square(self, xy) -> Tuple[str, float]:
        """(square name, distance) of the nearest square centre to a world xy (board pose aware)."""
        p = self.world_to_board(xy)
        f = int(np.clip(round(p[1] / C.SQUARE + 3.5), 0, 7))
        r = int(np.clip(round(8 - (p[0] - self.cfg.board_gap) / C.SQUARE + 0.5), 1, 8))
        cx, cy = self.cfg.square_xy(f, r)
        return C.square_name(f, r), float(np.hypot(p[0] - cx, p[1] - cy))

    def inside_board(self, xy, margin: float = 0.005) -> bool:
        """True if a world xy lies on the playing surface (+ margin), board pose aware."""
        p = self.world_to_board(xy)
        return bool(self.cfg.board_gap - margin <= p[0] <= self.cfg.board_gap + C.BOARD_SIZE + margin
                    and abs(p[1]) <= C.BOARD_SIZE / 2 + margin)

    @staticmethod
    def _envelope_sum(a: str, b: str) -> float:
        """Largest sum of the envelope radii of piece types a and b over the heights both have (m): two such
        pieces standing on the same plane do not touch iff their axes are further apart than this."""
        H = min(C.PIECE_SPECS[a][0], C.PIECE_SPECS[b][0])
        zs = np.arange(0.0, H + 1e-9, 0.00025)
        return float(max(C.piece_radius_at(a, z, "bands") + C.piece_radius_at(b, z, "bands") for z in zs))

    def piece_min_distance(self, type_a: str, type_b: str, gap: float = 0.0) -> float:
        """Minimum axis distance (m) at which two standing pieces keep `gap` between their real footprints."""
        return self._pair_min_dist[(type_a, type_b)] + gap

    # ------------------------------------------------------------------ pieces
    def piece_pos(self, name: str) -> np.ndarray:
        """Base-centre position (body origin) of a piece."""
        return self.data.xpos[self.piece_body_id[name]].copy()

    def piece_quat(self, name: str) -> np.ndarray:
        return self.data.xquat[self.piece_body_id[name]].copy()

    def piece_tilt_deg(self, name: str) -> float:
        R = self.data.xmat[self.piece_body_id[name]].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def set_piece_pose(self, name: str, xyz, yaw: float = 0.0, quat=None) -> None:
        a = self.piece_qadr[name]
        self.data.qpos[a:a + 3] = np.asarray(xyz, float)
        self.data.qpos[a + 3:a + 7] = _quat_yaw(yaw) if quat is None else np.asarray(quat, float)
        v = self.piece_vadr[name]
        self.data.qvel[v:v + 6] = 0.0

    def pieces_on_board(self) -> List[str]:
        return [n for n in self.piece_names if self._piece_square(n) is not None]

    def _piece_square(self, name: str, tol: float = 0.010, tilt_deg: float = 15.0, z_tol: float = 0.008):
        p = self.piece_pos(name)
        if abs(p[2] - self.board_top_z) > z_tol or self.piece_tilt_deg(name) > tilt_deg:
            return None
        if not self.inside_board(p, 0.005):
            return None
        sq, dist = self.nearest_square(p)
        return sq if dist <= tol else None

    def board_state(self) -> Dict[str, str]:
        """square -> piece body name for every upright piece within 10 mm of a square centre."""
        out = {}
        for n in self.piece_names:
            sq = self._piece_square(n)
            if sq is not None:
                out[sq] = n
        return out

    def board_state_symbols(self) -> Dict[str, str]:
        """square -> python-chess symbol ('P' white pawn, 'k' black king, ...)."""
        out = {}
        for sq, n in self.board_state().items():
            s = C.TYPE_TO_SYMBOL[self.piece_type[n]]
            out[sq] = s.upper() if self.piece_colour[n] == "w" else s
        return out

    def to_chess_board(self, turn: bool = chess.WHITE) -> chess.Board:
        """chess.Board built from the physical state (castling/en-passant rights are NOT recovered)."""
        b = chess.Board(None)
        for sq, sym in self.board_state_symbols().items():
            b.set_piece_at(chess.parse_square(sq), chess.Piece.from_symbol(sym))
        b.turn = turn
        return b

    def set_position(self, position=START_FEN, settle_steps: int = 0) -> Dict[str, str]:
        """Place the pieces for a FEN / chess.Board.  Exactly one body per piece instance:
        bodies of the right colour/type are assigned in square order; the spare queen is used for a
        promoted queen; unused bodies go to the parking area (spare queens to their cells).
        Returns the square -> body mapping."""
        board = chess.Board(position) if isinstance(position, str) else position
        wanted: Dict[Tuple[str, str], List[str]] = {}
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc is None:
                continue
            key = ("w" if pc.color == chess.WHITE else "b", C.SYMBOL_TO_TYPE[pc.symbol().lower()])
            wanted.setdefault(key, []).append(chess.square_name(sq))
        used = set()
        mapping: Dict[str, str] = {}
        for (colour, ptype), squares in wanted.items():
            pool = [f"{colour}_{ptype}_{i + 1}" for i in range(C.PIECE_COUNTS[ptype])]
            if ptype == "queen":
                pool.append(f"{colour}_queen_spare_1")
            if len(squares) > len(pool):
                raise ValueError(f"position needs {len(squares)} {colour} {ptype}s but only {len(pool)} bodies exist")
            for sq, body in zip(squares, pool):
                self.set_piece_pose(body, self.square_xyz(sq, 0.0005), yaw=self.board_yaw)   # rotated with the board
                used.add(body)
                mapping[sq] = body
        k = 0
        for n in self.piece_names:
            if n in used:
                continue
            if n.endswith("queen_spare_1"):
                self.set_piece_pose(n, self.spare_xyz(self.piece_colour[n], 0.0005))
            else:
                x, y = self.cfg.parking_xy(k)
                k += 1
                self.set_piece_pose(n, (x, y, 0.0005))
        mujoco.mj_forward(self.model, self.data)
        if settle_steps:
            self.step(settle_steps)
            self._anchor_grid()
        return mapping

    def drop_zone_cells(self) -> List[Tuple[int, int]]:
        return [(i, j) for j in range(C.DROP_CELLS) for i in range(C.DROP_CELLS)]

    def free_drop_cell(self) -> Optional[Tuple[int, int]]:
        """Next drop-zone cell (row-major in j then i) with no piece within 14 mm."""
        pieces = np.array([self.piece_pos(n)[:2] for n in self.piece_names])
        for i, j in self.drop_zone_cells():
            c = self.drop_cell_xyz(i, j)[:2]
            if np.min(np.linalg.norm(pieces - c, axis=1)) > 0.014:
                return (i, j)
        return None

    # ------------------------------------------------------------------ arm (LeRobot convention)
    def get_qpos6(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def get_joints(self) -> np.ndarray:
        """[5 joints deg, gripper %]."""
        return C.qpos_to_lerobot(self.get_qpos6())

    def set_joints(self, vec6, set_ctrl: bool = True) -> None:
        """Teleport the arm to [5 joints deg, gripper %] (zero velocity) and hold it there."""
        q = np.clip(C.lerobot_to_qpos(vec6), self.joint_range[:, 0], self.joint_range[:, 1])
        self.data.qpos[self.qpos_adr] = q
        self.data.qvel[self.qvel_adr] = 0.0
        if set_ctrl:
            self.data.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.model, self.data)

    def set_ctrl(self, vec6) -> None:
        """Position-actuator targets in LeRobot convention."""
        q = np.clip(C.lerobot_to_qpos(vec6), self.joint_range[:, 0], self.joint_range[:, 1])
        self.data.ctrl[self.act_ids] = q

    def get_ctrl(self) -> np.ndarray:
        return C.qpos_to_lerobot(self.data.ctrl[self.act_ids])

    def gripper_pos(self) -> np.ndarray:
        """World position of the gripperframe site (between the fingertips)."""
        return self.data.site_xpos[self.site_id].copy()

    def gripper_mat(self) -> np.ndarray:
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def gripper_q(self) -> float:
        """Current gripper joint angle (rad)."""
        return float(self.data.qpos[self.qpos_adr[5]])

    def jaw_gap(self) -> float:
        """Fingertip gap (m) MEASURED from the jaw meshes at the current state (ground truth for config.gripper_gap)."""
        if self._jaw_tips is None:
            self._jaw_tips = jaw_tip_vertices(self.model)
        return jaw_gap_from_mesh(self.model, self.data, self._jaw_tips)

    def jaw_clearance(self) -> float:
        """Jaw clearance (m) at the current state: the minimum distance between the fixed and moving finger boxes
        (the jaws' collision geometry), negative when penetrating; ground truth for config.gripper_clearance."""
        if self._finger_ids is None:
            self._finger_ids = finger_geom_ids(self.model)
        return jaw_clearance_from_boxes(self.model, self.data, self._finger_ids)

    def jaw_clearance_mesh(self) -> float:
        """Minimum distance (m) between the two jaw MESHES in the fingertip region (the real geometry's lips)."""
        if self._jaw_tips is None:
            self._jaw_tips = jaw_tip_vertices(self.model)
        return jaw_clearance_from_mesh(self.model, self.data, self._jaw_tips)

    def jaw_tip_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """World positions (fixed-jaw fingertip, moving-jaw fingertip) at the current state.  The FIXED jaw is the
        grasp reference: only the moving jaw swings, so a stem should be placed next to the fixed fingertip (not
        centred on the gripperframe site, which sits between the tips only when the jaws are closed)."""
        if self._jaw_tips is None:
            self._jaw_tips = jaw_tip_vertices(self.model)
        t, m, d = self._jaw_tips, self.model, self.data
        return (_mesh_world_vertices(m, d, t["geom_fixed"])[t["tip_fixed"]].copy(),
                _mesh_world_vertices(m, d, t["geom_moving"])[t["tip_moving"]].copy())

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def control_step(self, vec6=None) -> None:
        """One control period (exactly 1/CONTROL_HZ s on average) of physics, optionally with a new target.
        Physics runs 16 or 17 steps of 2 ms so that data.time stays on the k/CONTROL_HZ grid anchored at the last
        reset / settle (within one timestep, no drift): after n calls since the anchor, data.time == t0 + n / CONTROL_HZ
        +- timestep.  Physics advanced outside the grid (step(), settle(), set_position(settle_steps=..)) re-anchors
        it, so a control step is never shorter than 16 physics steps."""
        if vec6 is not None:
            self.set_ctrl(vec6)
        dt = self.model.opt.timestep
        target = self._grid_t0 + (self._ctrl_ticks + 1) * C.CONTROL_DT
        n = int(round((target - self.data.time) / dt))
        if n < C.SUBSTEPS - 1 or n > C.SUBSTEPS:
            self._anchor_grid()
            n = C.SUBSTEPS
        self._ctrl_ticks += 1
        self.step(n)

    def _anchor_grid(self) -> None:
        """Re-anchor the 30 Hz control grid at the current sim time (tick count restarts at 0)."""
        self._grid_t0 = float(self.data.time)
        self._ctrl_ticks = 0

    @property
    def control_ticks(self) -> int:
        """Number of control_step calls since the last grid anchor (reset / settle)."""
        return self._ctrl_ticks

    @property
    def grid_t0(self) -> float:
        """Sim time the control grid is anchored at (data.time == grid_t0 + control_ticks / CONTROL_HZ +- timestep)."""
        return self._grid_t0

    def settle(self, seconds: float = 0.5) -> None:
        """Run physics with the arm holding its current target; the control grid is re-anchored afterwards."""
        self.step(int(round(seconds / self.model.opt.timestep)))
        self._anchor_grid()

    # ------------------------------------------------------------------ reset / randomisation
    def restore_nominal(self) -> None:
        m = self.model
        m.body_pos[:] = self._nominal["body_pos"]
        m.body_quat[:] = self._nominal["body_quat"]
        m.light_pos[:] = self._nominal["light_pos"]
        m.light_dir[:] = self._nominal["light_dir"]
        m.vis.headlight.ambient[:] = self._nominal["head_ambient"]
        m.vis.headlight.diffuse[:] = self._nominal["head_diffuse"]
        m.light_diffuse[:] = self._nominal["light_diffuse"]
        m.geom_rgba[:] = self._nominal["geom_rgba"]
        m.cam_pos[:] = self._nominal["cam_pos"]
        m.cam_quat[:] = self._nominal["cam_quat"]

    def reset(self, seed: Optional[int] = None, randomise: bool = False, fen: str = START_FEN,
              arm_pose="watch", settle_seconds: float = 0.2) -> Dict[str, str]:
        """Reset physics, (optionally) randomise the world, place the pieces and the arm.
        arm_pose: 'watch' (default), None (zero pose) or a LeRobot 6-vector."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        m = self.model
        mujoco.mj_resetData(m, self.data)
        self._anchor_grid()
        self.restore_nominal()
        R = self.cfg.randomisation
        if randomise:
            if R.board_jitter:
                # board pose: drawn once (the watch pose covers the whole range, no re-draw), then the scene objects
                # that live in board coordinates follow (squares, pieces placed by set_position, corner sites)
                x0, x1, y0, y1, yaw = R.board_range()
                self.set_board_pose(rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(-yaw, yaw), forward=False)
            if R.zone_jitter:
                # the drop zone and the spare-queen pads stay on the table: their own small jitter
                for b in (self.drop_zone_id, self.spare_zone_id):
                    m.body_pos[b, :2] += rng.uniform(-R.zone_jitter_m, R.zone_jitter_m, 2)
            if R.light:
                aim = self.cfg.board_centre + np.array([0.0, 0.0, self.cfg.board_thickness])
                for i in range(m.nlight):
                    m.light_pos[i] += rng.uniform(-R.light_pos_m, R.light_pos_m, 3)
                    m.light_pos[i, 2] = max(m.light_pos[i, 2], 0.35)
                    m.light_diffuse[i] *= 1.0 + rng.uniform(-R.light_intensity, R.light_intensity)
                    if R.light_pos_m > 0.15:
                        # re-aim the (spot) light at the board, so a large displacement never leaves it in the dark
                        v = aim + rng.uniform(-0.05, 0.05, 3) * np.array([1, 1, 0]) - m.light_pos[i]
                        m.light_dir[i] = v / np.linalg.norm(v)
            if R.headlight:
                m.vis.headlight.ambient[:] = self._nominal["head_ambient"] * (1.0 + rng.uniform(-R.headlight_frac, R.headlight_frac))
                m.vis.headlight.diffuse[:] = self._nominal["head_diffuse"] * (1.0 + rng.uniform(-R.headlight_frac, R.headlight_frac))
            if R.colour:
                fw = 1.0 + rng.uniform(-R.colour_frac, R.colour_frac)
                fb = 1.0 + rng.uniform(-R.colour_frac, R.colour_frac)
                fboard = 1.0 + rng.uniform(-R.colour_frac, R.colour_frac)
                ftable = (1.0 + rng.uniform(-R.colour_frac, R.colour_frac)) * (
                    1.0 + (rng.uniform(-R.table_tint_frac, R.table_tint_frac, 3) if R.table_tint else np.zeros(3)))
                for g in range(m.ngeom):
                    name = m.geom(g).name
                    if "_vis" in name:
                        f = fw if name.startswith("w_") else fb
                        m.geom_rgba[g, :3] = np.clip(m.geom_rgba[g, :3] * f, 0, 1)
                    elif name in ("board_top", "board_box"):
                        m.geom_rgba[g, :3] = np.clip(m.geom_rgba[g, :3] * fboard, 0, 1)
                if R.colour_frac > 0.10 or R.table_tint:
                    g = self.table_geom_id
                    m.geom_rgba[g, :3] = np.clip(m.geom_rgba[g, :3] * ftable, 0, 1)
        # arm
        watch = isinstance(arm_pose, str) and arm_pose == "watch"
        if watch:
            self.set_joints(self.watch_pose())
        elif arm_pose is None:
            self.set_joints(np.zeros(6))
        else:
            self.set_joints(arm_pose)
        # wrist-camera mount jitter; re-drawn (shrinking) while a board corner leaves the wrist image
        if randomise and R.wrist_cam:
            c = self.cam_ids[C.WRIST_CAM]
            nom_cpos, nom_cquat = self._nominal["cam_pos"][c].copy(), self._nominal["cam_quat"][c].copy()
            tries = max(1, R.visible_retries) if (R.keep_board_visible and watch) else 1
            if tries > 1:
                # the guard only re-draws the CAMERA jitter: when the board pose itself puts the watch region
                # outside the nominal-camera window (only for poses beyond cfg.watch_board_range) no draw can help
                mujoco.mj_forward(m, self.data)
                if not self.watch_target_visible():
                    tries = 1
            n_full = max(1, tries // 2)          # full-amplitude draws first, then linearly shrinking ones
            self.last_cam_redraws = 0
            for k in range(tries):
                shrink = 1.0 if k < n_full else 1.0 - (k - n_full + 1) / (tries - n_full + 1)
                # displacement: random direction, |d| <= wrist_cam_m (uniform in the ball), except that the
                # lens never moves BACK into the camera module: the module mesh starts ~1 mm behind the lens,
                # so a backward (camera-local +z) displacement beyond wrist_cam_back_m would put the housing's
                # front face in front of the lens and render it (black frames); such draws are mirrored forward
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction) + 1e-12
                d = shrink * R.wrist_cam_m * rng.random() ** (1.0 / 3.0) * direction
                Rn = np.zeros(9)
                mujoco.mju_quat2Mat(Rn, nom_cquat)
                Rn = Rn.reshape(3, 3)
                d_local = Rn.T @ d
                if d_local[2] > R.wrist_cam_back_m:
                    d_local[2] = -d_local[2]
                    d = Rn @ d_local
                m.cam_pos[c] = nom_cpos + d
                axis = rng.normal(size=3)
                ang = np.deg2rad(shrink * rng.uniform(-R.wrist_cam_deg, R.wrist_cam_deg))
                m.cam_quat[c] = _quat_mul(nom_cquat, _quat_from_axis_angle(axis, ang))
                if tries == 1:
                    break
                mujoco.mj_forward(m, self.data)
                if self.watch_target_visible():
                    break
                self.last_cam_redraws = k + 1
        mujoco.mj_forward(m, self.data)
        # pieces (set_position places them on the current board pose, rotated with it)
        mapping = self.set_position(fen)
        if randomise and (R.piece_jitter or R.piece_yaw):
            self._jitter_pieces(mapping, rng)
        mujoco.mj_forward(m, self.data)
        if settle_seconds > 0:
            self.settle(settle_seconds)
        return mapping

    def watch_target_visible(self, margin: float = 0.0, region=None) -> bool:
        """True if the watch region (cfg.watch_region, or `region`) of the live board (its current pose) projects
        inside WRIST_IMAGE_WINDOW (+ margin, fraction of the window) of the live wrist camera."""
        region = tuple(self.cfg.watch_region) if region is None else region
        bp = self.board_pose() if not self._is_drop(region) else tuple(self.drop_zone_offset[:2]) + (0.0,)
        px = self.project(self.target_corners_at(*bp, region=region), C.WRIST_CAM)
        if np.any(np.isnan(px)):
            return False
        w, h = self.cfg.render_w, self.cfg.render_h
        u0, u1, v0, v1 = C.WRIST_IMAGE_WINDOW
        du, dv = margin * (u1 - u0) * w, margin * (v1 - v0) * h
        return bool(np.all(px[:, 0] >= u0 * w + du) and np.all(px[:, 0] <= u1 * w - du)
                    and np.all(px[:, 1] >= v0 * h + dv) and np.all(px[:, 1] <= v1 * h - dv))

    def _jitter_pieces(self, mapping: Dict[str, str], rng) -> None:
        """Piece xy jitter inside the square (+-piece_jitter_m along the BOARD axes) and random yaw.  A draw that
        brings a piece closer to an already placed neighbour than their real footprints allow (mesh envelopes at
        every height + piece_min_gap_m, piece_min_distance) is re-drawn, shrinking after 10 tries."""
        R = self.cfg.randomisation
        rot = self._board_rot()
        placed: List[Tuple[str, np.ndarray]] = []
        for sq, n in mapping.items():
            centre = self.square_xyz(sq, 0.0005)
            t = self.piece_type[n]
            p = centre.copy()
            if R.piece_jitter and R.piece_jitter_m > 0:
                for k in range(40):
                    shrink = 1.0 if k < 10 else max(0.0, 1.0 - (k - 9) / 30.0)
                    d = shrink * rng.uniform(-R.piece_jitter_m, R.piece_jitter_m, 2)
                    p = centre.copy()
                    p[:2] += rot @ d
                    if all(np.linalg.norm(p[:2] - q[:2]) >= self.piece_min_distance(t, tq, R.piece_min_gap_m)
                           for tq, q in placed):
                        break
                else:
                    p = centre.copy()
            placed.append((t, p))
            yaw = rng.uniform(-math.pi, math.pi) if R.piece_yaw else self.board_yaw
            self.set_piece_pose(n, p, yaw)

    # ------------------------------------------------------------------ rendering / projection
    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=h, width=w)
        return self._renderers[key]

    def render(self, camera_name: str = C.WRIST_CAM, w: Optional[int] = None, h: Optional[int] = None) -> np.ndarray:
        """uint8 RGB image (h, w, 3) from a named camera."""
        w = w or self.cfg.render_w
        h = h or self.cfg.render_h
        r = self._renderer(w, h)
        r.update_scene(self.data, camera=camera_name)
        if camera_name == C.WRIST_CAM:
            # The calibrated optical centre sits inside the lens, i.e. inside the camera module's mesh: hide the
            # module and its mount in the wrist view (the real camera does not see its own housing).
            hide = getattr(self, "_wrist_hide", None)
            if hide is None:
                bodies = {self.model.body(n).id for n in ("wrist_camera", "wrist_camera_mount")
                          if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
                hide = self._wrist_hide = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bodies}
            for i in range(r.scene.ngeom):
                g = r.scene.geoms[i]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in hide:
                    g.rgba[3] = 0.0
        return r.render().copy()

    def camera_pose(self, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """(pos, 3x3 rotation) of a camera in the world; MuJoCo cameras look along -z, +y up."""
        c = self.cam_ids[camera_name]
        return self.data.cam_xpos[c].copy(), self.data.cam_xmat[c].reshape(3, 3).copy()

    def project(self, points_xyz, camera_name: str, w: Optional[int] = None, h: Optional[int] = None) -> np.ndarray:
        """World points (N,3) -> pixel coords (N,2) [u right, v down]; points behind the camera get nan."""
        w = w or self.cfg.render_w
        h = h or self.cfg.render_h
        c = self.cam_ids[camera_name]
        pos, R = self.camera_pose(camera_name)
        f = (h / 2.0) / math.tan(math.radians(self.model.cam_fovy[c]) / 2.0)
        P = np.ascontiguousarray(np.atleast_2d(np.asarray(points_xyz, float)))
        v = np.dot(P - pos, R)                 # = R.T @ (p - pos) for each row
        depth = -v[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = w / 2.0 + f * v[:, 0] / depth
            vv = h / 2.0 - f * v[:, 1] / depth
        out = np.stack([u, vv], axis=1)
        out[depth <= 1e-6] = np.nan
        return out

    # ------------------------------------------------------------------ watch pose
    def _config_key(self) -> dict:
        return self.cfg.watch_key()

    def watch_pose(self, recompute: bool = False, path: Optional[str] = None, strict: bool = True,
                   **search_kwargs) -> np.ndarray:
        """LeRobot 6-vector of the watching pose, cached in a JSON keyed by cfg.watch_key()
        (assets/watch_pose.json for the default config, assets/watch_pose_<hash>.json otherwise).
        A pose that does not meet the margin / occlusion / collision requirements raises RuntimeError
        (strict=True) or warns (strict=False); the infeasible result is still cached in self._watch.
        search_kwargs are passed to compute_watch_pose (n_samples, n_refine, max_rounds, time_budget, ...)."""
        path = path or self.cfg.watch_pose_path()
        if self._watch is None and not recompute and os.path.exists(path):
            with open(path) as f:
                w = json.load(f)
            if w.get("config") == self._config_key() and "lerobot" in w:
                self._watch = w
        if self._watch is None or recompute:
            self._watch = self.compute_watch_pose(**search_kwargs)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._watch, f, indent=2)
        if not self._watch.get("feasible", False):
            msg = (f"watch pose is NOT feasible for config {self._config_key()}: margin {self._watch.get('margin_achieved', float('nan'))*100:.1f}% "
                   f"(required {self._watch.get('margin_required', C.WATCH_MARGIN)*100:.0f}%), occluded={self._watch.get('occluded')}, "
                   f"collision={self._watch.get('collision')}. Re-run with a larger budget: "
                   f"python -m chess_sim.scene --recompute-watch --rounds 8 --time-budget 600 (or change wrist_fovy / wrist_tilt_deg).")
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return np.array(self._watch["lerobot"], float)

    def scan_pose_path(self) -> str:
        p = self.cfg.watch_pose_path()
        return p[:-5] + "_scan.json"

    @staticmethod
    def _region_key(reg):
        return reg if isinstance(reg, str) else [round(float(v), 6) for v in reg]

    def scan_poses(self, recompute: bool = False, strict: bool = True, **search_kwargs) -> List[np.ndarray]:
        """LeRobot 6-vectors of the multi-view board SCAN: one pose per region of config.SCAN_REGIONS (board tiles on
        square boundaries + the drop zone), each seeing its region inside WRIST_IMAGE_WINDOW with SCAN_MARGIN,
        unoccluded, for EVERY board pose of cfg.scan_board_range (drop zone: every pad jitter extreme); together
        they see all 64 squares and the drop zone (with the real camera no single pose can, see README).  View 0 is
        the watch pose when cfg.watch_region is SCAN_REGIONS[0].  Cached in assets/watch_pose_scan.json."""
        path = self.scan_pose_path()
        key = dict(self._config_key(), scan_regions=[self._region_key(r) for r in C.SCAN_REGIONS],
                   scan_board_range=[round(float(v), 6) for v in self.cfg.scan_board_range], scan_margin=C.SCAN_MARGIN,
                   drop_layout=[C.DROP_GAP, C.DROP_X_SHIFT])
        rec = None
        if not recompute and os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
            if rec.get("config") != key:
                rec = None
        if rec is None:
            views = []
            self.watch_pose(strict=False)
            warm = [np.array(self._watch["lerobot"])] + list(self._watch_candidates(None))
            same0 = (self._region_key(C.SCAN_REGIONS[0]) == self._region_key(self.cfg.watch_region)
                     and tuple(self.cfg.watch_board_range) == tuple(self.cfg.scan_board_range) and C.SCAN_MARGIN <= C.WATCH_MARGIN)
            self._board_range_override = tuple(self.cfg.scan_board_range)
            pb_path = search_kwargs.pop("progress_path", C.PROGRESS_TXT)
            try:
                for i, reg in enumerate(C.SCAN_REGIONS):
                    if i == 0 and same0:
                        views.append(dict(self._watch))
                        continue
                    kw = dict(C.SCAN_SEARCH, **search_kwargs)
                    kw.setdefault("warm_starts", warm + [np.array(v["lerobot"]) for v in views])
                    kw.setdefault("margin", C.SCAN_MARGIN)
                    v = self.compute_watch_pose(region=reg, progress_path=pb_path, **kw)
                    views.append(v)
            finally:
                self._board_range_override = None
            rec = dict(config=key, views=views, feasible=all(v["feasible"] for v in views))
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
        self._scan = rec
        if not rec["feasible"]:
            msg = f"scan pose(s) NOT feasible: {[(v['region'], round(v['margin_achieved'], 3), v['occluded']) for v in rec['views']]}"
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return [np.array(v["lerobot"], float) for v in rec["views"]]

    def scan_metrics(self, i: int, vec6=None, **kw) -> dict:
        """watch_metrics of scan view i (its SCAN_REGIONS region, every board pose of cfg.scan_board_range or the
        drop-pad jitter extremes, rays at every pose by default)."""
        vec6 = self.scan_poses()[i] if vec6 is None else vec6
        saved = (self.watch_region, self._board_range_override)
        reg = C.SCAN_REGIONS[i]
        self.watch_region = reg if self._is_drop(reg) else tuple(reg)
        self._board_range_override = tuple(self.cfg.scan_board_range)
        try:
            kw.setdefault("ext_limit", np.inf)
            kw.setdefault("rays_all", True)
            kw.setdefault("margin", C.SCAN_MARGIN)
            return self.watch_metrics(vec6, **kw)
        finally:
            self.watch_region, self._board_range_override = saved

    def _arm_contact_pairs(self) -> set:
        """Body pairs in contact that involve the arm (after mj_forward)."""
        m, d = self.model, self.data
        pairs = set()
        for i in range(d.ncon):
            con = d.contact[i]
            if con.dist > 0:
                continue
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if b1 in self.arm_body_ids or b2 in self.arm_body_ids:
                pairs.add((min(b1, b2), max(b1, b2)))
        return pairs

    def _arm_silhouette_vertices(self, per_geom: int = 600, n_extreme: int = 96) -> List[Tuple[int, np.ndarray]]:
        """[(geom id, (K,3) local vertices)] for the arm's visual meshes (except the camera module itself):
        a random subset plus the extreme vertices along random directions (keeps the jaw tips and other
        silhouette points).  Used as a cheap continuous occlusion measure by watch_metrics."""
        if self._arm_verts is not None:
            return self._arm_verts
        m = self.model
        rng = np.random.default_rng(12345)
        dirs = rng.normal(size=(n_extreme, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        out = []
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b not in self.arm_body_ids or b == self._cam_body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if m.geom_group[g] != 2:        # visual meshes only (the collision copies are identical)
                continue
            mid = m.geom_dataid[g]
            v = np.ascontiguousarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], dtype=float)
            idx = set(np.argmax(np.dot(v, dirs.T), axis=0).tolist())     # np.dot: '@' raises spurious FP flags on Accelerate
            if len(v) > per_geom:
                idx |= set(rng.choice(len(v), per_geom, replace=False).tolist())
            else:
                idx |= set(range(len(v)))
            out.append((g, np.ascontiguousarray(v[sorted(idx)])))
        self._arm_verts = out
        return out

    def _silhouette_penetration_px(self, quad_px: np.ndarray, camera_name: str = C.WRIST_CAM) -> float:
        """Largest distance (pixels) by which a projected arm vertex lies inside the board quad (0 = clear).
        Vertices behind the camera are ignored.  Continuous in the joint angles, so a search can use it."""
        m, d = self.model, self.data
        w, h = self.cfg.render_w, self.cfg.render_h
        pts = []
        for g, v in self._arm_silhouette_vertices():
            R = d.geom_xmat[g].reshape(3, 3)
            pts.append(np.dot(v, R.T) + d.geom_xpos[g])
        P = np.ascontiguousarray(np.concatenate(pts, axis=0))
        px = self.project(P, camera_name, w, h)
        ok = ~np.isnan(px[:, 0])
        if not np.any(ok):
            return 0.0
        px = px[ok]
        # signed distance to each edge of the (convex) quad, oriented so inside is positive
        Q = np.asarray(quad_px, float)
        area2 = np.dot(Q[:, 0], np.roll(Q[:, 1], -1)) - np.dot(Q[:, 1], np.roll(Q[:, 0], -1))
        sgn = 1.0 if area2 > 0 else -1.0
        dists = []
        for i in range(4):
            a, b = Q[i], Q[(i + 1) % 4]
            e = b - a
            n = np.linalg.norm(e) + 1e-12
            dists.append(sgn * (e[0] * (px[:, 1] - a[1]) - e[1] * (px[:, 0] - a[0])) / n)   # cross(e, p - a)
        inside = np.min(np.stack(dists, axis=0), axis=0)      # > 0 inside all four half-planes
        return float(max(0.0, inside.max()))

    def watch_board_poses(self) -> List[Tuple[float, float, float]]:
        """Board poses (dx, dy, yaw_deg) the watch pose must cover: the nominal pose first, then the 8 extreme
        combinations of cfg.watch_board_range (duplicates dropped, e.g. a zero range gives the nominal only)."""
        if self._is_drop(self.watch_region):
            j = self.cfg.randomisation.zone_jitter_m if self.cfg.randomisation.zone_jitter else 0.0
            j = max(j, C.Randomisation().zone_jitter_m)          # the 'wide' pad jitter
            return [(0.0, 0.0, 0.0)] + [(sx * j, sy * j, 0.0) for sx in (-1, 1) for sy in (-1, 1)]
        x0, x1, y0, y1, yaw = self._board_range_override or self.cfg.watch_board_range
        out = [(0.0, 0.0, 0.0)]
        for x in (x0, x1):
            for y in (y0, y1):
                for a in (-yaw, yaw):
                    p = (float(x), float(y), float(a))
                    if p not in out:
                        out.append(p)
        return out

    def watch_metrics(self, vec6, margin: float = C.WATCH_MARGIN, check_collision: bool = True,
                      ext_limit: float = 1.5, shaping: bool = True, rays: bool = True,
                      board_poses: Optional[Sequence[Tuple[float, float, float]]] = None, rays_all: bool = False) -> dict:
        """Evaluate a candidate watch pose.  Returns corner pixels, 'extent' (largest normalised distance of a
        corner from the image centre: <= 1 - 2*margin means all corners are inside with that margin),
        'margin_ok', 'area_frac' (projected board area / image area), 'occ_px' (pixels by which the arm's
        projected silhouette enters the board quad; continuous, used by the search), 'occluded' (ray-cast:
        the arm between the camera and the board, 5x5 then 17x17 grid), 'collision' (arm touching anything)
        and 'feasible'.  Occlusion / collision are only evaluated when extent <= ext_limit (the expensive part);
        the ray test only when rays=True and the silhouette test is clean.
        board_poses (default watch_board_poses(): nominal + the 8 extremes of cfg.watch_board_range): the corner
        extent / margin and the silhouette occlusion are the WORST over these board poses (relative to the nominal
        board, whatever the live board pose is), area_frac the SMALLEST; 'per_pose' lists (pose, extent, area).
        The ray test runs on the live board only, or on every pose when rays_all=True (moves the board body
        temporarily: final checks only)."""
        m, d = self.model, self.data
        w, h = self.cfg.render_w, self.cfg.render_h
        q = np.clip(C.lerobot_to_qpos(vec6), self.joint_range[:, 0], self.joint_range[:, 1])
        d.qpos[self.qpos_adr] = q
        d.qvel[self.qvel_adr] = 0.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        c = self.cam_ids[C.WRIST_CAM]
        cam_pos = d.cam_xpos[c].copy()
        poses = list(board_poses) if board_poses is not None else self.watch_board_poses()
        pxs = [self.project(self.target_corners_at(*bp), C.WRIST_CAM, w, h) for bp in poses]
        u0, u1, v0, v1 = C.WRIST_IMAGE_WINDOW
        uc, hu, vc, hv = (u0 + u1) / 2 * w, (u1 - u0) / 2 * w, (v0 + v1) / 2 * h, (v1 - v0) / 2 * h
        px = pxs[0]
        res = dict(corners_px=px.tolist(), extent=np.inf, margin_ok=False, area_frac=0.0, occ_px=None, occluded=None,
                   collision=None, cam_z=float(cam_pos[2]), feasible=False, per_pose=[])
        if any(np.any(np.isnan(p)) for p in pxs) or cam_pos[2] < self.cfg.board_thickness + 0.03:
            return res
        exts, areas, quads = [], [], []
        for p in pxs:
            # extent inside the usable image window (the real jaw covers the right ~third, WRIST_IMAGE_WINDOW)
            exts.append(max(np.max(np.abs(p[:, 0] - uc)) / hu, np.max(np.abs(p[:, 1] - vc)) / hv))
            quad = p[[0, 1, 3, 2]]                      # a8, h8, h1, a1
            quads.append(quad)
            areas.append(0.5 * abs(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))) / (w * h))
        ext = max(exts)
        res["extent"] = float(ext)
        res["extent_nominal"] = float(exts[0])
        res["margin_ok"] = bool(ext <= 1.0 - 2.0 * margin)
        res["area_frac"] = float(min(areas))
        res["area_nominal"] = float(areas[0])
        res["per_pose"] = [(list(bp), float(e), float(a)) for bp, e, a in zip(poses, exts, areas)]
        if ext > ext_limit:
            return res
        if shaping:
            res["occ_px"] = max(self._silhouette_penetration_px(q) for q in quads)
        if check_collision:
            mujoco.mj_forward(m, d)
            res["collision"] = bool(self._arm_contact_pairs() - self._baseline_pairs)
        if rays and (res["occ_px"] is None or res["occ_px"] == 0.0):
            # rays from the camera to a grid of board points must hit the board first
            # (coarse 5x5 grid first, then a fine grid so the jaw tips cannot slip between the rays)
            occluded = False
            if rays_all:
                saved = self.board_pose()
                saved_dz = m.body_pos[self.drop_zone_id].copy()
                drop = self._is_drop(self.watch_region)
                for bp in poses:
                    if drop:
                        m.body_pos[self.drop_zone_id] = self._nominal["body_pos"][self.drop_zone_id] + np.array([bp[0], bp[1], 0.0])
                    else:
                        self.set_board_pose(*bp, forward=False)
                    mujoco.mj_kinematics(m, d)
                    for n in self.occlusion_grids:
                        occluded = self._board_occluded(cam_pos, n)
                        if occluded:
                            break
                    if occluded:
                        break
                self.set_board_pose(*saved, forward=False)
                m.body_pos[self.drop_zone_id] = saved_dz
                mujoco.mj_kinematics(m, d)
            else:
                for n in self.occlusion_grids:
                    occluded = self._board_occluded(cam_pos, n)
                    if occluded:
                        break
            res["occluded"] = occluded
        elif res["occ_px"] is not None and res["occ_px"] > 0.0:
            res["occluded"] = True
        res["feasible"] = bool(res["margin_ok"] and res["occluded"] is False and not res["collision"])
        return res

    def _board_occluded(self, cam_pos: np.ndarray, n: int) -> bool:
        """True if any ray from cam_pos to an n x n grid of points of the watch target (the board region, or the
        drop-zone cells for region 'drop') hits something before the target's body."""
        m, d = self.model, self.data
        geomgroup = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)   # group 0: table/board, group 2: arm meshes
        geomid = np.zeros(1, dtype=np.int32)
        bc = self.cfg.board_centre
        if self._is_drop(self.watch_region):
            q = self.target_corners_at(*self.drop_zone_offset[:2], 0.0)
            pts = [q[0] + a * (q[2] - q[0]) + b * (q[1] - q[0]) for a in np.linspace(0, 1, n) for b in np.linspace(0, 1, n)]
            target_body = self.drop_zone_id
        else:
            r0, r1, f0, f1 = self._region4(self.watch_region)
            pts = []
            for sx in np.linspace(2 * r0 - 1, 2 * r1 - 1, n):
                for sy in np.linspace(2 * f0 - 1, 2 * f1 - 1, n):
                    xy = self.board_to_world((bc[0] + sx * C.BOARD_SIZE / 2, bc[1] + sy * C.BOARD_SIZE / 2))
                    pts.append(np.array([xy[0], xy[1], self.board_top_z]))
            target_body = self.board_id
        for p in pts:
            vec = p - cam_pos
            # start just in front of the lens (the camera-module body is excluded from the ray test)
            start = cam_pos + 0.005 * vec / np.linalg.norm(vec)
            dist = mujoco.mj_ray(m, d, start, vec, geomgroup, 1, self._cam_body, geomid)
            if dist < 0 or m.geom_bodyid[geomid[0]] != target_body:
                return True
        return False

    def _watch_score(self, r: dict, margin: float) -> float:
        """Continuous objective for the search (larger is better).  Infeasible poses score the worst constraint
        slack (maximin, all in normalised image units): corner extent beyond the margin limit, the arm
        silhouette entering the board quad (pixels / half image height), ray-cast occlusion and collision; a
        maximin lets the optimiser trade one constraint against the other through the narrow valley between
        the top margin and the jaw tips.  Feasible poses score 1 + projected board area, so the search then
        maximises the area.  Unevaluated candidates (board not in front of the camera) score -inf."""
        if not np.isfinite(r["extent"]):
            return -np.inf
        lim = 1.0 - 2.0 * margin
        slack_ext = lim - r["extent"]
        if r["occ_px"] is None and r["occluded"] is None:        # beyond ext_limit: unevaluated
            return min(0.0, slack_ext) - 1.0
        slacks = [slack_ext]
        if r["occ_px"] is not None:
            slacks.append(-r["occ_px"] / (self.cfg.render_h / 2.0))
        if r["occluded"]:
            slacks.append(-0.1)
        if r["collision"]:
            slacks.append(-0.2)
        if r["feasible"]:
            return 1.0 + r["area_frac"]
        return min(slacks)

    def _watch_candidates(self, path: Optional[str]) -> List[np.ndarray]:
        """Warm starts: the config prior, any cached pose (even for another config) and the current pose."""
        cands = [np.array(C.WATCH_POSE_PRIOR, float)]
        for p in {path, self.cfg.watch_pose_path(), C.WATCH_POSE_JSON} - {None}:
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        w = json.load(f)
                    if "lerobot" in w:
                        cands.append(np.array(w["lerobot"], float))
                except (OSError, ValueError):
                    pass
        if self._watch is not None and "lerobot" in self._watch:
            cands.append(np.array(self._watch["lerobot"], float))
        return cands

    def compute_watch_pose(self, n_samples: int = C.WATCH_SEARCH["n_samples"], n_keep: int = C.WATCH_SEARCH["n_keep"],
                           n_refine: int = C.WATCH_SEARCH["n_refine"], max_rounds: int = C.WATCH_SEARCH["max_rounds"],
                           time_budget: float = C.WATCH_SEARCH["time_budget"], seed: int = 0,
                           margin: float = C.WATCH_MARGIN, warm_starts: Optional[Sequence] = None,
                           verbose: bool = True, progress_path: str = C.PROGRESS_TXT,
                           region: Optional[Tuple[float, float]] = None) -> dict:
        """Joint-space search for the watching pose: maximise the projected board area subject to all 4 corners
        >= `margin` inside the wrist image, no arm between the camera and the board, no collisions, gripper
        closed.  Each round draws n_samples random joint vectors, keeps the n_keep best (plus the warm starts:
        config prior, cached poses, or `warm_starts`; pass [] for a cold search) and refines each with a
        (1+1)-ES of n_refine steps on the continuous score; rounds repeat with new seeds until a feasible
        pose is found, max_rounds or time_budget (s) is exhausted.  If the margin is unattainable the best
        pose is returned with feasible=False (see margin_achieved / occluded / collision).  The progress bar is
        written to progress_path (default chess_sim/data/progress.txt) while verbose.
        region (r0, r1): search a pose that sees only that depth band of the board (scan views, see scan_poses)."""
        saved_region = self.watch_region
        region = region if region is not None else tuple(self.cfg.watch_region)
        if region is not None:
            self.watch_region = region if self._is_drop(region) else tuple(float(v) for v in region)
        try:
            return self._compute_watch_pose(n_samples, n_keep, n_refine, max_rounds, time_budget, seed, margin, warm_starts,
                                            verbose, progress_path)
        finally:
            self.watch_region = saved_region

    def _compute_watch_pose(self, n_samples, n_keep, n_refine, max_rounds, time_budget, seed, margin, warm_starts,
                            verbose, progress_path) -> dict:
        m, d = self.model, self.data
        saved_q, saved_ctrl = d.qpos.copy(), d.ctrl.copy()
        saved_board = self.board_pose()
        self.set_board_pose(0.0, 0.0, 0.0, forward=False)        # the search is relative to the nominal board
        lo, hi = self.joint_range[:5, 0], self.joint_range[:5, 1]
        lo_d, hi_d = np.degrees(lo), np.degrees(hi)
        pb = ProgressBar(max_rounds * (n_samples + n_keep * n_refine), path=progress_path) if verbose else None
        t0 = time.time()
        warm = list(warm_starts) if warm_starts is not None else self._watch_candidates(None)
        warm = [np.concatenate([np.asarray(v, float)[:5], [0.0]]) for v in warm]
        global_best: Optional[Tuple[float, np.ndarray, dict]] = None
        n_evals = 0
        rounds_done = 0

        def consider(sc, vec, r):
            nonlocal global_best
            if global_best is None or sc > global_best[0]:
                global_best = (sc, vec.copy(), r)

        for rnd in range(max_rounds):
            rng = np.random.default_rng(seed + 1000 * rnd)
            base = rnd * (n_samples + n_keep * n_refine)
            # ---- sampling (cheap: extent only)
            pool: List[Tuple[float, np.ndarray]] = []
            for i in range(n_samples):
                q5 = rng.uniform(lo_d, hi_d)
                if rng.random() < 0.3:      # the wrist often saturates: sample near its limits too
                    q5[3] = (hi_d[3] if rng.random() < 0.5 else lo_d[3]) * rng.uniform(0.9, 1.0)
                vec = np.concatenate([q5, [0.0]])
                r = self.watch_metrics(vec, margin, check_collision=False, ext_limit=-1.0, shaping=False, rays=False)
                n_evals += 1
                if np.isfinite(r["extent"]):
                    pool.append((-abs(r["extent"] - (1.0 - 2.0 * margin)), vec))
                if pb and i % 500 == 0:
                    pb.update(base + i, f"watch-pose round {rnd + 1}/{max_rounds}: sampling")
            pool.sort(key=lambda t: -t[0])
            starts = (warm if rnd == 0 else []) + [v for _, v in pool[:n_keep]]
            if not starts:
                continue
            # ---- refinement: (1+1)-ES with step-size adaptation from every start
            for si, vec in enumerate(starts):
                r = self.watch_metrics(vec, margin)
                sc = self._watch_score(r, margin)
                consider(sc, vec, r)
                sigma = 4.0
                for k in range(n_refine):
                    cand = vec.copy()
                    cand[:5] = np.clip(cand[:5] + rng.normal(0.0, sigma, 5), lo_d, hi_d)
                    r2 = self.watch_metrics(cand, margin)
                    n_evals += 1
                    sc2 = self._watch_score(r2, margin)
                    if sc2 > sc:
                        vec, sc, r = cand, sc2, r2
                        sigma = min(15.0, sigma * 1.4)
                    else:
                        sigma *= 0.9
                        if sigma < 0.05:            # converged: restart the step size to escape the valley floor
                            sigma = 2.0
                    if pb and k % 100 == 0:
                        pb.update(base + n_samples + min(si, n_keep) * n_refine + k,
                                  f"watch-pose round {rnd + 1}/{max_rounds}: refine {si + 1}/{len(starts)}, best "
                                  f"{'FEASIBLE area %.3f' % global_best[2]['area_frac'] if global_best and global_best[2]['feasible'] else 'score %.3f' % (global_best[0] if global_best else float('nan'))}")
                    if time.time() - t0 > time_budget:
                        break
                consider(sc, vec, r)
                if time.time() - t0 > time_budget:
                    break
            rounds_done = rnd + 1
            if global_best is not None and global_best[2]["feasible"]:
                break
            if time.time() - t0 > time_budget:
                break
        d.qpos[:] = saved_q
        d.ctrl[:] = saved_ctrl
        mujoco.mj_forward(m, d)
        if global_best is None:
            self.set_board_pose(*saved_board)
            if pb:
                pb.fail("watch-pose search FAILED: no candidate pose")
            raise RuntimeError("no watch pose candidate found (camera never sees the board)")
        score, vec, r = global_best
        # final, full evaluation (ray-cast gate) of the chosen pose
        r = self.watch_metrics(vec, margin, ext_limit=np.inf, rays_all=True)
        d.qpos[:] = saved_q
        d.ctrl[:] = saved_ctrl
        self.set_board_pose(*saved_board)
        result = dict(lerobot=[float(v) for v in vec], qpos=[float(v) for v in C.lerobot_to_qpos(vec)],
                      feasible=bool(r["feasible"]), occluded=bool(r["occluded"]), collision=bool(r["collision"]),
                      area_frac=r["area_frac"], extent=r["extent"], margin_required=margin,
                      margin_achieved=float((1.0 - r["extent"]) / 2.0), margin_ok=bool(r["margin_ok"]),
                      occ_px=r["occ_px"], corners_px=r["corners_px"], image_size=[self.cfg.render_w, self.cfg.render_h],
                      area_nominal=r.get("area_nominal"), margin_nominal=float((1.0 - r.get("extent_nominal", r["extent"])) / 2.0),
                      board_poses=[list(p) for p in self.watch_board_poses()], per_pose=r.get("per_pose"),
                      region=self._region_key(self.watch_region),
                      config=self._config_key(), seconds=time.time() - t0, n_samples=n_samples, n_keep=n_keep,
                      n_refine=n_refine, rounds=rounds_done, max_rounds=max_rounds, n_evals=n_evals,
                      warm_starts=len(warm), score=float(score))
        if pb:
            pb.done(f"watch pose {'DONE' if r['feasible'] else 'INFEASIBLE'} area {r['area_frac']:.3f} margin "
                    f"{result['margin_achieved']*100:.1f}% in {time.time() - t0:.0f}s ({rounds_done} rounds)")
        return result


# ---------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    """python -m chess_sim.scene [--recompute-watch] [--samples N] [--refine N] [--rounds N] [--time-budget S]
    [--seed S] [--margin M] [--cold] [--board-gap X ...].  Prints the model summary and the watch pose."""
    import argparse
    from chess_sim.build_scene import add_config_args, config_from_args
    ap = argparse.ArgumentParser(description=main.__doc__)
    add_config_args(ap)
    ap.add_argument("--recompute-watch", action="store_true", help="re-run the watch-pose search and overwrite the cache")
    ap.add_argument("--samples", type=int, default=C.WATCH_SEARCH["n_samples"])
    ap.add_argument("--keep", type=int, default=C.WATCH_SEARCH["n_keep"])
    ap.add_argument("--refine", type=int, default=C.WATCH_SEARCH["n_refine"])
    ap.add_argument("--rounds", type=int, default=C.WATCH_SEARCH["max_rounds"])
    ap.add_argument("--time-budget", type=float, default=C.WATCH_SEARCH["time_budget"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=C.WATCH_MARGIN)
    ap.add_argument("--cold", action="store_true", help="no warm starts (ignore the prior and cached poses)")
    ap.add_argument("--rebuild", action="store_true", help="regenerate the scene XML for this config")
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    s = ChessScene(cfg, rebuild=True if a.rebuild else "auto")
    print("model:", s.xml_path, "nbody", s.model.nbody, "ngeom", s.model.ngeom, "config", cfg.scene_key())
    kw = dict(n_samples=a.samples, n_keep=a.keep, n_refine=a.refine, max_rounds=a.rounds, time_budget=a.time_budget,
              seed=a.seed, margin=a.margin)
    if a.cold:
        kw["warm_starts"] = []
    try:
        w = s.watch_pose(recompute=a.recompute_watch, strict=True, **kw)
    except RuntimeError as e:
        print("ERROR:", e)
        w = np.array(s._watch["lerobot"])
    info = s._watch
    print("watch pose (LeRobot deg/%):", np.round(w, 2), "cache:", cfg.watch_pose_path())
    print(f"  feasible {info['feasible']}  margin {info['margin_achieved']*100:.2f}%  area {info['area_frac']:.3f}  "
          f"occluded {info['occluded']}  collision {info['collision']}  search {info.get('seconds', 0):.0f}s "
          f"rounds {info.get('rounds', '?')} evals {info.get('n_evals', '?')}")
    s.reset(seed=0, randomise=False)
    print("board_state:", len(s.board_state()), "pieces")
    return 0 if info["feasible"] else 1


if __name__ == "__main__":
    sys.exit(main())
