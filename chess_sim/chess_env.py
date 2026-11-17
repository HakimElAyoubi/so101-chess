"""gymnasium environment SO101ChessMove-v0: execute ONE chess move with the SO-101 in the MuJoCo chess scene.

Registered as 'gym_so101_chess/SO101ChessMove-v0' (import this module, then gym.make(...)).

observation (LeRobot style):
    'pixels':    {'wrist': (H, W, 3) uint8}   wrist camera (cfg.render_h x cfg.render_w, default 360 x 640, 16:9 like the real camera)
    'agent_pos': float32[6]                    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] deg
                                               + gripper %  (LeRobot convention, see config.py)
    'environment_state': float32[128]          the MOVE to execute, one-hot: [0:64] from-square, [64:128] to-square
                                               (chess square index a1=0 .. h8=63; encode_move / decode_move).  LeRobot's
                                               preprocess_observation maps it to 'observation.environment_state', which
                                               ACT / diffusion consume (ACT's env_state_feature); record_demos stores the
                                               same vector, so an ACT trained on the demos is move-conditioned.
    'task':      str                           'move the piece from e2 to e4' (used by language-conditioned policies:
                                               smolvla / pi0; ACT and diffusion ignore it)
action: float32[6] joint position targets in the same convention.  Each step the commanded arm target may move at
most JOINT_SPEED_DEG_S / CONTROL_HZ (3 deg) per joint away from the previous target (the spec's speed limit); the
gripper percent is not rate limited.  One step = one control period (1 / CONTROL_HZ s = 16-17 physics steps).

reset(seed, options): options = {'fen': ..., 'move': 'e7e5' | SAN, 'randomise': bool, 'plan': bool, 'keep_scene': bool}.
By default a random mid-game position is drawn (random legal play from the start position, 1..29 plies, black =
the robot to move) together with a random legal move that the scripted expert can plan (so expert demos are always
available; options={'plan': False} skips the planning).  The pieces captured during that random game are placed in
the drop zone in capture order (replay_captures; the 17th and later captures are parked behind the robot), as they
would be after a real game; a position given as 'fen' has no history, so its captured pieces are parked.
'keep_scene': True (needs 'fen' + 'move') starts an episode on the scene AS IT IS (no scene reset: pieces keep their
jittered positions, the drop zone its contents, the arm its pose) for a game loop that runs a policy move by move
(play_game.py --policy-path); the physical board must match the fen.

reward: 1.0 on success at termination (sparse), plus an optional dense shaping term (dense_reward=True):
    -0.05 * (distance fixed fingertip -> stem of the piece to grasp  +  xy distance piece -> target)   [m]
terminated: success, or a piece toppled (> 20 deg), a piece knocked > 15 mm away, the carried piece dropped away
from its source and target squares, a piece off the table; truncated at max_steps.
success (the spec's definition): every moved piece's base centre within 8 mm of its target, tilt < 10 deg, no
other piece displaced > 5 mm, the gripper retreated to the lift height (fingertips >= retreat height above the
board with no arm-piece contact, latched once reached after the pieces are in place; the retreat height is the
spec's LIFT_CLEARANCE = 90 mm wherever the arm can reach it above the last placed piece, else that square's reach
ceiling - 5 mm: ranks 7-8, see kinematics.SPEC_DEVIATIONS; retreat_height=<float> fixes it) and, with
require_watch (default), the arm back at the watch pose (<= 3 deg) so an episode ends in the 'base/watching'
state.  step() raises ValueError on a non-finite action; reset(options={'move': ..}) needs 'fen' too.

info: 'fen' (position BEFORE the move), 'move' (uci), 'task', 'phase' (reach/grasp/carry/place/return/done/failed),
'success', 'failure', 'targets' {piece: xyz}, 'piece_err_mm', 'pieces' {name: [x y z qw qx qy qz]}, 'board'
(square -> symbol from the physical state), 'steps', 'sim_time', 'expert' (label of the expert waypoint when the
expert drives).

expert_action(obs) returns the next action of the scripted expert (chess_sim.expert.action_generator) for the
current episode, so a demo is produced by the ordinary env loop; expert_done tells when the plan is finished.
"""
from __future__ import annotations

import math
import os
import string
import sys
from typing import Dict, List, Optional, Tuple

import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chess_sim import config as C
    from chess_sim.scene import ChessScene
    from chess_sim.expert import Expert, get_expert, PlanError, MovePlan, KNOCKED_TILT_DEG, KNOCKED_DISP, LIFT_Z_TOL
else:
    from . import config as C
    from .scene import ChessScene
    from .expert import Expert, get_expert, PlanError, MovePlan, KNOCKED_TILT_DEG, KNOCKED_DISP, LIFT_Z_TOL

ENV_ID = "gym_so101_chess/SO101ChessMove-v0"
TASK_CHARSET = string.ascii_letters + string.digits + " =-_,'"
MAX_PLIES_DEFAULT = 30
STEP_LIMIT_DEG = C.JOINT_SPEED_DEG_S / C.CONTROL_HZ      # 3 deg per control step
RETREAT_HEIGHT = C.LIFT_CLEARANCE   # nominal retreat (lift) height above the board surface for the success check
                                    # (0.090 m, the spec).  Per episode it is capped at the reach ceiling above the
                                    # last placed piece minus LIFT_Z_TOL (ranks 7-8: 25..55 mm, the expert's own
                                    # lowered lift height, kinematics.SPEC_DEVIATIONS); never below RETREAT_MIN.
RETREAT_MIN = 0.020                 # m, floor for the rank-dependent retreat height
WATCH_TOL_DEG = 3.0
LIFTED_DZ = 0.012               # a target piece is 'lifted' when its base is this high above the surface
RESTING_DZ = 0.004
DROP_FAR = 0.015                # dropped: resting > this from both source and target with the jaws away
JAW_AWAY = 0.020
OFF_TABLE_Z = -0.02
LYING_TILT_DEG = 60.0           # a held target piece tilted beyond this is lying in the jaws: counted as toppled
HELD_DIST = 0.030               # a jaw tip within this of the stem point = the piece is held
STATE_LOW = np.array([-110.0, -100.0, -100.0, -95.0, -160.0, 0.0], np.float32)     # generous bounds (deg, %)
STATE_HIGH = np.array([110.0, 100.0, 100.0, 95.0, 165.0, 100.0], np.float32)


MOVE_ENCODING_DIM = 2 * 64
MOVE_FEATURE_NAMES = [f"from_{sq}" for sq in chess.SQUARE_NAMES] + [f"to_{sq}" for sq in chess.SQUARE_NAMES]


def task_string(move: chess.Move) -> str:
    return f"move the piece from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"


def encode_move(move) -> np.ndarray:
    """One-hot float32[128] of a move: from-square at [0:64], to-square at [64:128] (chess.SQUARES order, a1 = 0).
    A promotion is implied (v1 always promotes to a queen); castling is encoded as the king's move."""
    mv = chess.Move.from_uci(move) if isinstance(move, str) else move
    v = np.zeros(MOVE_ENCODING_DIM, np.float32)
    v[mv.from_square] = 1.0
    v[64 + mv.to_square] = 1.0
    return v


def decode_move(vec, board: Optional[chess.Board] = None) -> chess.Move:
    """Inverse of encode_move (argmax per half).  With `board`, a pawn reaching the last rank gets promotion=QUEEN."""
    v = np.asarray(vec, np.float32).reshape(MOVE_ENCODING_DIM)
    mv = chess.Move(int(np.argmax(v[:64])), int(np.argmax(v[64:])))
    if board is not None:
        pc = board.piece_at(mv.from_square)
        if pc is not None and pc.piece_type == chess.PAWN and chess.square_rank(mv.to_square) in (0, 7):
            mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
    return mv


def replay_captures(scene: ChessScene, board: chess.Board) -> List[str]:
    """Place the pieces captured during board.move_stack (a game played from the start position) in the drop
    zone in capture order, as a real game leaves them: scene.set_position parks every unused body, so for each
    capture a parked body of the captured colour/type is moved to the next free drop cell (free_drop_cell; when
    the zone is full the rest stay parked).  Returns the bodies moved."""
    b = chess.Board()
    captured: List[Tuple[str, str]] = []
    for mv in board.move_stack:
        if b.is_capture(mv):
            sq = mv.to_square
            if b.is_en_passant(mv):
                sq = chess.square(chess.square_file(mv.to_square), chess.square_rank(mv.from_square))
            pc = b.piece_at(sq)
            if pc is not None:
                captured.append(("w" if pc.color == chess.WHITE else "b", C.SYMBOL_TO_TYPE[pc.symbol().lower()]))
        b.push(mv)
    import mujoco
    on_board = set(scene.board_state().values())
    moved: List[str] = []
    for colour, ptype in captured:
        pool = [n for n in scene.piece_names if n not in on_board and n not in moved and scene.piece_colour[n] == colour
                and scene.piece_type[n] == ptype and not n.endswith("queen_spare_1")]
        cell = scene.free_drop_cell()           # reads body positions: forward kinematics after every placement
        if not pool or cell is None:
            continue
        scene.set_piece_pose(pool[0], scene.drop_cell_xyz(*cell, 0.0005))
        mujoco.mj_forward(scene.model, scene.data)
        moved.append(pool[0])
    if moved:
        scene.settle(0.1)
    return moved


def parse_move(board: chess.Board, move) -> chess.Move:
    """chess.Move | UCI | SAN -> chess.Move (must be legal)."""
    if isinstance(move, chess.Move):
        mv = move
    else:
        s = str(move).strip()
        try:
            mv = chess.Move.from_uci(s)
        except ValueError:
            mv = board.parse_san(s)
    if mv not in board.legal_moves:
        raise ValueError(f"{mv.uci()} is not legal in {board.fen()}")
    return mv


def position_is_representable(board: chess.Board) -> bool:
    """The scene has 8 pawns, 2 R/N/B, 1 Q (+1 spare queen) and 1 K per colour."""
    for colour in (chess.WHITE, chess.BLACK):
        for pt, n in ((chess.PAWN, 8), (chess.ROOK, 2), (chess.KNIGHT, 2), (chess.BISHOP, 2), (chess.QUEEN, 2), (chess.KING, 1)):
            if len(board.pieces(pt, colour)) > n:
                return False
    return True


def random_position(rng: np.random.Generator, max_plies: int = MAX_PLIES_DEFAULT, colour: bool = chess.BLACK,
                    max_tries: int = 50) -> chess.Board:
    """Random legal play from the start position for an odd (black to move) number of plies in 1..max_plies-1
    (0 plies would be white to move).  The returned position is not over and has `colour` to move."""
    for _ in range(max_tries):
        n_pairs = int(rng.integers(0, max(1, max_plies // 2)))        # 0..14 -> 1..29 plies
        n = 2 * n_pairs + (1 if colour == chess.BLACK else 0)
        board = chess.Board()
        for _ in range(n):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[int(rng.integers(len(legal)))])
        if board.turn == colour and not board.is_game_over() and position_is_representable(board):
            return board
    return chess.Board() if colour == chess.WHITE else chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")


def move_targets(scene: ChessScene, board: chess.Board, move: chess.Move) -> Dict[str, np.ndarray]:
    """piece body -> target base position for a move, in the expert's order (captured piece first, then the mover
    / promotion pawn + spare queen, then the castling rook).  Same drop-cell policy as the expert."""
    state = scene.board_state()
    frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
    if frm not in state:
        raise ValueError(f"no piece standing on {frm} in the scene")
    mover = state[frm]
    targets: Dict[str, np.ndarray] = {}
    taken: List[Tuple[int, int]] = []

    def next_cell():
        pieces = np.array([scene.piece_pos(n)[:2] for n in scene.piece_names])
        for i, j in scene.drop_zone_cells():
            if (i, j) in taken:
                continue
            c = scene.drop_cell_xyz(i, j)[:2]
            if np.min(np.linalg.norm(pieces - c, axis=1)) > 0.014:
                taken.append((i, j))
                return scene.drop_cell_xyz(i, j)
        raise ValueError("drop zone full")

    if board.is_capture(move):
        if board.is_en_passant(move):
            cap_sq = chess.square_name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
        else:
            cap_sq = to
        if cap_sq in state:
            targets[state[cap_sq]] = next_cell()
    if move.promotion:
        colour = scene.piece_colour[mover]
        targets[mover] = next_cell()
        targets[f"{colour}_queen_spare_1"] = scene.square_xyz(to)
    else:
        targets[mover] = scene.square_xyz(to)
    if board.is_castling(move):
        rank = frm[1]
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            r_from, r_to = f"h{rank}", f"f{rank}"
        else:
            r_from, r_to = f"a{rank}", f"d{rank}"
        if r_from in state:
            targets[state[r_from]] = scene.square_xyz(r_to)
    return targets


class SO101ChessMoveEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": C.CONTROL_HZ}

    def __init__(self, cfg: Optional[C.ChessConfig] = None, scene: Optional[ChessScene] = None, randomise: bool = False,
                 max_steps: int = 1500, dense_reward: bool = False, require_watch: bool = True, obs_pixels: bool = True,
                 max_plies: int = MAX_PLIES_DEFAULT, plan_tries: int = 6, render_mode: Optional[str] = "rgb_array",
                 render_camera: str = C.SPECTATOR_CAM, retreat_height: Optional[float] = None):
        super().__init__()
        self.scene = scene or ChessScene(cfg)
        self.cfg = self.scene.cfg
        self.expert: Expert = get_expert(self.scene)
        self.randomise = randomise
        self.max_steps = int(max_steps)
        self.dense_reward = dense_reward
        self.require_watch = require_watch
        self.obs_pixels = obs_pixels
        self.max_plies = max_plies
        self.plan_tries = plan_tries
        self.render_mode = render_mode
        self.render_camera = render_camera
        self.retreat_height = retreat_height        # None = rank dependent (computed at reset), float = fixed
        self._retreat_z = RETREAT_HEIGHT            # the episode's retreat height above the board surface
        self._retreated_flag = False
        h, w = self.cfg.render_h, self.cfg.render_w
        lo = np.concatenate([np.degrees(self.scene.joint_range[:5, 0]), [0.0]]).astype(np.float32)
        hi = np.concatenate([np.degrees(self.scene.joint_range[:5, 1]), [100.0]]).astype(np.float32)
        self.action_space = spaces.Box(low=lo, high=hi, shape=(6,), dtype=np.float32)
        obs = {"agent_pos": spaces.Box(low=STATE_LOW, high=STATE_HIGH, shape=(6,), dtype=np.float32),
               "environment_state": spaces.Box(low=0.0, high=1.0, shape=(MOVE_ENCODING_DIM,), dtype=np.float32),
               "task": spaces.Text(max_length=64, charset=TASK_CHARSET)}
        if obs_pixels:
            obs["pixels"] = spaces.Dict({"wrist": spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)})
        self.observation_space = spaces.Dict(obs)
        # episode state
        self.board: chess.Board = chess.Board()
        self.move: Optional[chess.Move] = None
        self.task: str = ""
        self.move_vec: np.ndarray = np.zeros(MOVE_ENCODING_DIM, np.float32)
        self.plan: Optional[MovePlan] = None
        self.plan_error: Optional[str] = None
        self.targets: Dict[str, np.ndarray] = {}
        self._initial: Dict[str, np.ndarray] = {}
        self._lifted: Dict[str, bool] = {}
        self._watch: Optional[np.ndarray] = None
        self._prev_target = np.zeros(6)
        self._steps = 0
        self._gen = None
        self._expert_done = False
        self._last_action = np.zeros(6, np.float32)
        self._last_info: dict = {}

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = dict(options or {})
        randomise = bool(options.get("randomise", self.randomise))
        want_plan = bool(options.get("plan", True))
        scene_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.plan, self.plan_error = None, None          # never carry a plan over from the previous episode
        fen = options.get("fen")
        move = options.get("move")
        keep_scene = bool(options.get("keep_scene", False))
        if move is not None and fen is None:
            raise ValueError("reset(options={'move': ...}) needs 'fen' as well (the start position has White to move)")
        if keep_scene and (fen is None or move is None):
            raise ValueError("reset(options={'keep_scene': True}) needs 'fen' and 'move' (the episode runs on the scene as it is)")
        if fen is not None:
            board = chess.Board(fen)
            if not position_is_representable(board):
                raise ValueError("position has more pieces of a type than the scene has bodies (max 2 queens per colour)")
            if keep_scene:
                expected = {chess.square_name(sq): pc.symbol() for sq, pc in board.piece_map().items()}
                if self.scene.board_state_symbols() != expected:
                    raise ValueError("keep_scene: the physical board does not match the fen")
            else:
                self.scene.reset(seed=scene_seed, randomise=randomise, fen=board.fen())
            if move is not None:
                mv = parse_move(board, move)
                if want_plan:
                    self._try_plan(board, mv, raise_on_fail=False)
            else:
                mv = self._pick_move(board, want_plan)
        else:
            mv = None
            for _ in range(10):
                board = random_position(self.np_random, self.max_plies, chess.BLACK)
                self.scene.reset(seed=scene_seed, randomise=randomise, fen=board.fen())
                replay_captures(self.scene, board)
                mv = self._pick_move(board, want_plan)
                if mv is not None:
                    break
            if mv is None:
                raise RuntimeError("could not find a plannable random move")
        self.board = board
        self.move = mv
        self.task = task_string(mv)
        self.move_vec = encode_move(mv)
        self.targets = move_targets(self.scene, board, mv)
        if self.plan is not None and self.plan.expected:
            self.targets.update({k: np.asarray(v, float) for k, v in self.plan.expected.items()})
        self._initial = {n: self.scene.piece_pos(n) for n in self.scene.piece_names}
        self._lifted = {n: False for n in self.targets}
        self._retreat_z = self._episode_retreat_height()
        self._retreated_flag = False
        self._watch = np.asarray(self.scene.watch_pose(), float)
        self._prev_target = np.asarray(self.scene.get_joints(), float)
        self._steps = 0
        self._gen = None
        self._expert_done = False
        self._last_action = self._prev_target.astype(np.float32)
        obs = self._obs()
        info = self._info(success=False, failure=None)
        self._last_info = info
        return obs, info

    def _episode_retreat_height(self) -> float:
        """Retreat height above the board for this episode: the fixed retreat_height, or the spec's lift height
        capped at the reach ceiling (expert.lift_z_at) above the LAST placed piece (the mover, or the castling rook),
        minus LIFT_Z_TOL, like the expert's own 'retreated to lift height' check."""
        if self.retreat_height is not None:
            return float(self.retreat_height)
        if not self.targets:
            return RETREAT_HEIGHT
        xy = np.asarray(list(self.targets.values())[-1], float)[:2]
        z_lift = self.expert.lift_z_at(xy) - self.scene.board_top_z        # = LIFT_CLEARANCE where reachable
        if z_lift >= RETREAT_HEIGHT - 1e-6:
            return float(RETREAT_HEIGHT)
        return float(max(RETREAT_MIN, z_lift - LIFT_Z_TOL))

    def _try_plan(self, board: chess.Board, mv: chess.Move, raise_on_fail: bool) -> bool:
        self.plan, self.plan_error = None, None
        try:
            self.plan = self.expert.plan_move(mv, board)
            return True
        except PlanError as e:
            self.plan_error = str(e)
            if raise_on_fail:
                raise
            return False

    def _pick_move(self, board: chess.Board, want_plan: bool) -> Optional[chess.Move]:
        legal = list(board.legal_moves)
        order = self.np_random.permutation(len(legal))
        if not want_plan:
            self.plan, self.plan_error = None, None
            return legal[int(order[0])] if legal else None
        for k in order[:self.plan_tries]:
            mv = legal[int(k)]
            if self._try_plan(board, mv, raise_on_fail=False):
                return mv
        return None

    # ------------------------------------------------------------------ step
    def step(self, action):
        if self.move is None:
            raise RuntimeError("call reset() before step()")
        a = np.asarray(action, dtype=float).reshape(6)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite action {a.tolist()}")
        a = np.clip(a, self.action_space.low, self.action_space.high)
        d = np.clip(a[:5] - self._prev_target[:5], -STEP_LIMIT_DEG, STEP_LIMIT_DEG)
        target = np.concatenate([self._prev_target[:5] + d, [a[5]]])
        self.scene.control_step(target)
        self._prev_target = target
        self._last_action = target.astype(np.float32)
        self._steps += 1
        for n in self.targets:
            if self.scene.piece_pos(n)[2] > self.scene.board_top_z + LIFTED_DZ:
                self._lifted[n] = True
        success, failure = self._evaluate()
        terminated = bool(success or failure is not None)
        truncated = bool(not terminated and self._steps >= self.max_steps)
        reward = 1.0 if success else 0.0
        if self.dense_reward:
            reward += self._dense()
        obs = self._obs()
        info = self._info(success=success, failure=failure)
        self._last_info = info
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ evaluation
    def _move_ok(self) -> Tuple[bool, Dict[str, float]]:
        s = self.scene
        errs = {}
        ok = True
        for n, tgt in self.targets.items():
            p = s.piece_pos(n)
            err = float(np.linalg.norm(p[:2] - tgt[:2]))
            errs[n] = err * 1000
            resting = abs(p[2] - tgt[2]) < RESTING_DZ
            if err > C.SUCCESS_POS_TOL or s.piece_tilt_deg(n) > C.SUCCESS_TILT_DEG or not resting:
                ok = False
        return ok, errs

    def _others_ok(self) -> Optional[str]:
        s = self.scene
        for n, p0 in self._initial.items():
            if n in self.targets:
                continue
            if np.linalg.norm(s.piece_pos(n)[:2] - p0[:2]) > C.DISTURB_TOL:
                return n
        return None

    def _clear_of_pieces(self) -> bool:
        return not self.expert._live_arm_piece_contacts()

    def _above_retreat(self) -> bool:
        s = self.scene
        tip_fixed, tip_moving = s.jaw_tip_positions()
        return bool(min(tip_fixed[2], tip_moving[2]) >= s.board_top_z + self._retreat_z)

    def _retreated(self) -> bool:
        """The gripper rose to the retreat height clear of every piece after the pieces were in place (latched
        until a placed piece leaves its target again), and is clear of the pieces now."""
        if not self._retreated_flag and self._above_retreat() and self._clear_of_pieces():
            self._retreated_flag = True
        return self._retreated_flag and self._clear_of_pieces()

    def _at_watch(self) -> bool:
        q = self.scene.get_joints()
        return bool(np.max(np.abs(q[:5] - self._watch[:5])) <= WATCH_TOL_DEG)

    def _evaluate(self) -> Tuple[bool, Optional[str]]:
        s = self.scene
        tips = s.jaw_tip_positions()
        # failures
        for n in s.piece_names:
            p = s.piece_pos(n)
            if p[2] < OFF_TABLE_Z:
                return False, f"off_table:{n}"
            tilt = s.piece_tilt_deg(n)
            if tilt > KNOCKED_TILT_DEG:
                # a target piece held in the jaws may swing up to ~20-30 deg while carried (the expert's own check
                # excludes the carried piece too); it counts as toppled only when lying (> LYING_TILT_DEG)
                if n in self.targets and p[2] > s.board_top_z + RESTING_DZ:
                    stem = p + np.array([0.0, 0.0, C.grasp_height(s.piece_type[n])])
                    if min(np.linalg.norm(t - stem) for t in tips) < HELD_DIST and tilt <= LYING_TILT_DEG:
                        continue
                return False, f"toppled:{n}"
        for n, p0 in self._initial.items():
            if n in self.targets:
                continue
            if np.linalg.norm(s.piece_pos(n)[:2] - p0[:2]) > KNOCKED_DISP:
                return False, f"knocked:{n}"
        for n, tgt in self.targets.items():
            if not self._lifted[n]:
                continue
            p = s.piece_pos(n)
            if p[2] < s.board_top_z + RESTING_DZ or abs(p[2] - tgt[2]) < RESTING_DZ:
                d_src = np.linalg.norm(p[:2] - self._initial[n][:2])
                d_tgt = np.linalg.norm(p[:2] - tgt[:2])
                jaw_d = min(np.linalg.norm(t[:2] - p[:2]) for t in tips)
                if d_src > DROP_FAR and d_tgt > DROP_FAR and jaw_d > JAW_AWAY:
                    return False, f"dropped:{n}"
        ok, _ = self._move_ok()
        if not ok:
            self._retreated_flag = False            # the retreat counts only once the pieces are in place
            return False, None
        if self._others_ok() is not None or not self._retreated():
            return False, None
        if self.require_watch and not self._at_watch():
            return False, None
        return True, None

    def _phase(self, success: bool, failure: Optional[str]) -> str:
        if failure:
            return "failed"
        if success:
            return "done"
        s = self.scene
        ok, errs = self._move_ok()
        if ok:
            return "return"
        tips = s.jaw_tip_positions()
        for n, tgt in self.targets.items():
            p = s.piece_pos(n)
            if np.linalg.norm(p[:2] - tgt[:2]) <= C.SUCCESS_POS_TOL and abs(p[2] - tgt[2]) < RESTING_DZ:
                continue
            if p[2] > s.board_top_z + LIFTED_DZ:
                return "carry"
            if np.linalg.norm(p[:2] - tgt[:2]) < 0.02:
                return "place"
            if min(np.linalg.norm(t - p) for t in tips) < 0.040:
                return "grasp"
            return "reach"
        return "return"

    def _dense(self) -> float:
        s = self.scene
        tip_fixed, _ = s.jaw_tip_positions()
        for n, tgt in self.targets.items():
            p = s.piece_pos(n)
            d_t = float(np.linalg.norm(p[:2] - tgt[:2]))
            if d_t <= C.SUCCESS_POS_TOL and abs(p[2] - tgt[2]) < RESTING_DZ:
                continue
            stem = p + np.array([0, 0, C.grasp_height(s.piece_type[n])])
            d_g = 0.0 if p[2] > s.board_top_z + LIFTED_DZ else float(np.linalg.norm(tip_fixed - stem))
            return -0.05 * (d_g + d_t)
        return 0.0

    # ------------------------------------------------------------------ observation / info
    def _obs(self) -> dict:
        obs = {"agent_pos": np.asarray(self.scene.get_joints(), np.float32), "environment_state": self.move_vec.copy(),
               "task": self.task}
        if self.obs_pixels:
            obs["pixels"] = {"wrist": self.scene.render(C.WRIST_CAM)}
        return obs

    def _info(self, success: bool, failure: Optional[str]) -> dict:
        s = self.scene
        _, errs = self._move_ok()
        pieces = {n: np.concatenate([s.piece_pos(n), s.piece_quat(n)]).tolist() for n in s.piece_names}
        return dict(fen=self.board.fen(), move=self.move.uci() if self.move else None, task=self.task,
                    phase=self._phase(success, failure), success=bool(success), failure=failure,
                    targets={k: np.asarray(v).tolist() for k, v in self.targets.items()}, piece_err_mm=errs,
                    pieces=pieces, board=s.board_state_symbols(), steps=self._steps, sim_time=float(s.data.time),
                    expert=self.expert.state.get("label", "") if self._gen is not None else "",
                    expert_done=self._expert_done, plan_error=self.plan_error, disturbed=self._others_ok(),
                    at_watch=self._at_watch() if self._watch is not None else False,
                    retreat_height=self._retreat_z, retreated=self._retreated_flag,
                    board_pose=dict(zip(("dx", "dy", "yaw_deg"), s.board_pose())),
                    board_corners=s.board_corners().tolist(),
                    drop_zone_offset=s.drop_zone_offset[:2].tolist(), spare_zone_offset=s.spare_zone_offset[:2].tolist())

    def render(self):
        return self.scene.render(self.render_camera)

    # ------------------------------------------------------------------ expert
    @property
    def expert_done(self) -> bool:
        return self._expert_done

    def expert_action(self, obs=None) -> np.ndarray:
        """Next action of the scripted expert for the current episode (plans the move first when needed).
        After the plan is exhausted the last action (the watch pose) is repeated and expert_done is True."""
        if self._gen is None:
            if self.plan is None:
                self._try_plan(self.board, self.move, raise_on_fail=True)
            self._gen = self.expert.action_generator(self.plan)
            self._expert_done = False
        if not self._expert_done:
            try:
                a = next(self._gen)
                self._last_action = np.asarray(a, np.float32)
            except StopIteration:
                self._expert_done = True
        return self._last_action.copy()

    @property
    def expert_abort(self) -> Optional[str]:
        return self.expert.state.get("abort") if self._gen is not None else None


def register() -> None:
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="chess_sim.chess_env:SO101ChessMoveEnv", max_episode_steps=None)


register()


def run_expert_episode(env: SO101ChessMoveEnv, seed: Optional[int] = None, options: Optional[dict] = None,
                       on_step=None, extra_steps: int = 15) -> dict:
    """reset + expert rollout through env.step; returns a summary dict (success, steps, failure, reasons...)."""
    obs, info = env.reset(seed=seed, options=options)
    frames = 0
    total_r = 0.0
    terminated = truncated = False
    tail = 0
    while not (terminated or truncated):
        a = env.expert_action(obs)
        obs, r, terminated, truncated, info = env.step(a)
        total_r += r
        frames += 1
        if on_step is not None:
            on_step(obs, a, r, info)
        if env.expert_done or env.expert_abort:
            tail += 1
            if tail > extra_steps:
                break
    return dict(success=bool(info["success"]), failure=info["failure"], steps=frames, reward=total_r, fen=info["fen"],
                move=info["move"], task=info["task"], expert_abort=env.expert_abort, piece_err_mm=info["piece_err_mm"],
                phase=info["phase"], at_watch=info["at_watch"], disturbed=info["disturbed"], plan_error=env.plan_error)


if __name__ == "__main__":
    import time
    env = SO101ChessMoveEnv()
    t0 = time.time()
    res = run_expert_episode(env, seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(res, f"{time.time() - t0:.1f}s wall")
