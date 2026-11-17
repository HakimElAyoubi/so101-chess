"""Play a game against Stockfish in the MuJoCo chess scene: the SO-101 executes Black's moves.

Viewer mode (a native MuJoCo window; macOS REQUIRES mjpython):
    mjpython chess_sim/play_game.py [--skill 5] [--time 0.1]
        You play White by typing moves in the terminal (UCI 'e2e4' or SAN 'Nf3'; 'hint', 'fen', 'auto', 'quit').
        The white piece is moved for you (teleported), Stockfish chooses Black's reply, the arm picks and places
        it in the physics (captures go to the drop zone, promotion via the spare queen, castling moves both
        pieces) and returns to the watch pose.
    ... play_game.py --auto [--plies 20]        Stockfish plays both sides (White teleported, Black by the arm)
    ... play_game.py --auto --arm-white         the arm executes both colours
Headless (plain python, no window):
    python chess_sim/play_game.py --headless --plies 6
        reports every move (executor, success, reasons, timings) and the totals; exit code 0 iff every arm move
        succeeded.
    ... play_game.py --no-viewer               the interactive terminal game without a window (plain python;
                                               moves can be piped on stdin, EOF = quit)
    ... play_game.py --headless --plies 6 --policy-path <ckpt>/pretrained_model [--policy-device mps]
        a trained LeRobot policy (run_policy.PolicyExecutor) executes the arm's moves instead of the scripted
        expert: each move is one SO101ChessMove-v0 episode on the live scene (keep_scene), success per the env.
Board reader (--reader privileged|vision, board_reader.py): after every opponent move the arm 'looks at the board'
from the watch pose (the wrist camera is rendered), the reader returns the position and infer_move() recovers the
move that was played from the difference with the logical position; only the privileged reader (simulator ground
truth behind the camera render) exists in the digital twin, --reader vision raises NotImplementedError.
Stockfish: --stockfish-path (default: shutil.which('stockfish'), or $SO101_STOCKFISH), --skill 0..20,
--time seconds per move, --depth, --multipv N (alternative moves tried when the arm cannot plan the best one).
--fen is validated (and must be representable with the scene's bodies) before the engine starts: a bad FEN exits 1.
If the arm fails a move (PlanError for every candidate, piece dropped, misplaced) the failure is reported, the
pieces are re-synchronised to the logical position (teleport) and the game continues.
Progress bar: every mode writes the one-line bar (plies played / planned, current move and expert phase) to
chess_sim/data/progress.txt (--progress-path) about once per second; view it with chess_sim/progress_view.command.
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import sys
import threading
import time
from typing import Dict, List, Optional

import chess
import chess.engine
import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_sim import config as C
from chess_sim.scene import ChessScene
from chess_sim.expert import get_expert, PlanError, MovePlan
from chess_sim.progress import ProgressBar
from chess_sim.board_reader import BoardReader, PrivilegedBoardReader, make_reader, infer_move

DEFAULT_STOCKFISH = os.environ.get("SO101_STOCKFISH", "stockfish")


def find_stockfish(path: Optional[str] = None) -> str:
    """The Stockfish binary: an explicit `path` must exist (no silent fallback), else PATH, else Homebrew's."""
    if path:
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"--stockfish-path {path!r} does not exist")
    for p in (shutil.which("stockfish"), DEFAULT_STOCKFISH):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("stockfish not found: brew install stockfish, or pass --stockfish-path")


def expected_symbols(board: chess.Board) -> Dict[str, str]:
    return {chess.square_name(sq): pc.symbol() for sq, pc in board.piece_map().items()}


def parse_start_fen(fen: str) -> chess.Board:
    """chess.Board for --fen; ValueError (with a clear message) when it is malformed, invalid or needs more
    bodies than the scene has."""
    from chess_sim.chess_env import position_is_representable
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"--fen {fen!r} is not a valid FEN: {e}")
    if not board.is_valid():
        raise ValueError(f"--fen {fen!r} is not a legal position ({board.status()!r})")
    if not position_is_representable(board):
        raise ValueError(f"--fen {fen!r} needs more pieces of a type than the scene has bodies (max 2 queens per colour)")
    return board


class Game:
    """Game state + physical execution.  `board` is the logical position; the scene is kept consistent with it."""

    def __init__(self, scene: ChessScene, engine_path: Optional[str] = None, skill: int = 5, time_limit: float = 0.1,
                 depth: Optional[int] = None, multipv: int = 5, arm_colours=(chess.BLACK,), viewer=None,
                 realtime: bool = False, speed: float = 1.0, verbose: bool = True, seed: int = 0, randomise: bool = False,
                 policy=None, reader: Optional[BoardReader] = None, policy_max_steps: int = 900):
        """policy: a run_policy.PolicyExecutor (or None: the scripted expert executes the arm's moves).  reader: the
        BoardReader that observes the opponent's moves (default PrivilegedBoardReader)."""
        self.scene = scene
        self.expert = get_expert(scene)
        self.policy = policy
        self.policy_max_steps = int(policy_max_steps)
        self._env = None
        self.reader: BoardReader = reader if reader is not None else PrivilegedBoardReader()
        self.read_log: List[dict] = []            # one entry per opponent move: observed symbols -> inferred move
        self.board = chess.Board()
        self.engine_path = find_stockfish(engine_path)
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        try:
            self.engine.configure({"Skill Level": int(skill)})
        except chess.engine.EngineError:
            pass
        self.limit = chess.engine.Limit(time=time_limit) if depth is None else chess.engine.Limit(depth=depth)
        self.multipv = max(1, multipv)
        self.arm_colours = set(arm_colours)
        self.viewer = viewer
        self.realtime = realtime
        self.speed = speed
        self.verbose = verbose
        self.records: List[dict] = []
        self.scene.reset(seed=seed, randomise=randomise)
        self._parking_k = 0
        self.progress: Optional[ProgressBar] = None     # set by run_auto / run_interactive (or the caller)
        self.plies_target: Optional[int] = None
        self._prog_state = "starting"

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------ helpers
    def say(self, *a):
        if self.verbose:
            print(*a)
            sys.stdout.flush()

    def _sync_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    # ------------------------------------------------------------------ progress bar
    def _prog_count(self) -> int:
        return len(self.board.move_stack)

    def note(self, state: Optional[str] = None, force: bool = False) -> None:
        """Write the progress bar (rate limited to once per second by ProgressBar unless force)."""
        if state is not None:
            self._prog_state = state
        if self.progress is not None:
            self.progress.update(self._prog_count(), f"play_game: ply {self._prog_count() + 1} {self._prog_state}", force=force)

    def _tick(self, label: str = "") -> None:
        """Per-control-step bar refresh (cheap: ProgressBar skips writes within its 1 s period)."""
        if self.progress is not None:
            self.progress.update(self._prog_count(), f"play_game: ply {self._prog_count() + 1} {self._prog_state} {label}".rstrip())

    def _step_wall(self, t_next: float) -> float:
        """Real-time pacing of one control step in viewer mode."""
        if self.realtime:
            dt = C.CONTROL_DT / max(1e-3, self.speed)
            now = time.time()
            if t_next > now:
                time.sleep(t_next - now)
            return max(t_next, now) + dt
        return t_next

    def idle(self, seconds: float = 0.0):
        """Keep physics (and the window) alive while waiting."""
        n = max(1, int(round(seconds / C.CONTROL_DT)))
        t_next = time.time()
        for _ in range(n):
            self.scene.control_step()
            self._sync_viewer()
            t_next = self._step_wall(t_next)
            self._tick()

    def _free_parking(self) -> np.ndarray:
        pieces = np.array([self.scene.piece_pos(n)[:2] for n in self.scene.piece_names])
        for k in range(self._parking_k, self._parking_k + 40):
            x, y = self.scene.cfg.parking_xy(k)
            if np.min(np.linalg.norm(pieces - np.array([x, y]), axis=1)) > 0.014:
                self._parking_k = k + 1
                return np.array([x, y, 0.0005])
        return np.array([-0.4, 0.0, 0.0005])

    def physical_matches(self) -> bool:
        return self.scene.board_state_symbols() == expected_symbols(self.board)

    def resync(self, why: str):
        """Teleport every piece to the logical position (captured pieces leave the drop zone for the parking area)."""
        self.say(f"  ! re-synchronising the pieces to the logical position ({why})")
        self.scene.set_position(self.board, settle_steps=50)
        self._sync_viewer()

    # ------------------------------------------------------------------ moves
    def teleport_move(self, move: chess.Move) -> None:
        """Apply `move` physically by teleporting (the human's move); self.board is the position BEFORE it."""
        s, b = self.scene, self.board
        state = s.board_state()
        frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
        if frm not in state:
            self.resync(f"no piece on {frm}")
            state = s.board_state()
        mover = state[frm]
        if b.is_capture(move):
            cap_sq = to
            if b.is_en_passant(move):
                cap_sq = chess.square_name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
            if cap_sq in state:
                s.set_piece_pose(state[cap_sq], self._free_parking())
        if move.promotion:
            colour = s.piece_colour[mover]
            on_board = set(state.values())
            # the spare queen, or any queen body of that colour that is off the board (captured earlier); when
            # every queen body is in use the position is not representable and the caller's physical check fails
            pool = [f"{colour}_queen_spare_1"] + [n for n in s.piece_names
                                                 if s.piece_colour[n] == colour and s.piece_type[n] == "queen"]
            free = [n for n in pool if n not in on_board and n != mover]
            s.set_piece_pose(mover, self._free_parking())
            if free:
                s.set_piece_pose(free[0], s.square_xyz(to, 0.0005))
            else:
                self.say(f"  ! no free {colour} queen body for the promotion on {to}")
        else:
            s.set_piece_pose(mover, s.square_xyz(to, 0.0005))
        if b.is_castling(move):
            rank = frm[1]
            if chess.square_file(move.to_square) > chess.square_file(move.from_square):
                r_from, r_to = f"h{rank}", f"f{rank}"
            else:
                r_from, r_to = f"a{rank}", f"d{rank}"
            if r_from in state:
                s.set_piece_pose(state[r_from], s.square_xyz(r_to, 0.0005))
        mujoco.mj_forward(s.model, s.data)
        self.idle(0.1)

    def observe_opponent_move(self, expected: Optional[chess.Move] = None) -> Optional[chess.Move]:
        """'Arm looks at the board': read the physical position with the board reader (at the watch pose the arm
        returned to after its own move) and infer the opponent's move from the difference with the logical
        position.  Returns the inferred move (None when nothing legal explains the observation); the read is
        recorded in self.read_log.  `expected` (the move that was teleported) is only used for the log."""
        observed = self.reader.read(self.scene)
        mv = infer_move(self.board, observed)
        self.read_log.append(dict(ply=len(self.board.move_stack) + 1, reader=self.reader.name, inferred=mv.uci() if mv else None,
                                  expected=expected.uci() if expected else None, n_pieces=len(observed)))
        return mv

    def engine_candidates(self, n: Optional[int] = None) -> List[chess.Move]:
        n = n or self.multipv
        infos = self.engine.analyse(self.board, self.limit, multipv=n)
        if isinstance(infos, dict):
            infos = [infos]
        moves = [inf["pv"][0] for inf in infos if inf.get("pv")]
        if not moves:
            moves = [self.engine.play(self.board, self.limit).move]
        return moves

    def policy_move(self, move: chess.Move) -> dict:
        """Execute `move` with the learned policy: one SO101ChessMove-v0 episode on the live scene (keep_scene),
        judged by the env's success definition.  On failure the pieces are re-synchronised (teleport)."""
        from chess_sim.chess_env import SO101ChessMoveEnv
        s = self.scene
        if self._env is None:
            self._env = SO101ChessMoveEnv(scene=s, max_steps=self.policy_max_steps, require_watch=True)
        env = self._env
        rec = dict(ply=len(self.board.move_stack) + 1, side="white" if self.board.turn else "black", executor="policy",
                   candidates=[move.uci()], plan_errors=[], success=False, reasons=[], plan_s=0.0, sim_s=0.0, wall_s=0.0,
                   piece_err_mm={}, uci=move.uci(), san=self.board.san(move))
        t0 = time.time()
        t_sim0 = s.data.time
        obs, info = env.reset(options={"fen": self.board.fen(), "move": move.uci(), "plan": False, "keep_scene": True})
        self.policy.reset()
        self.note(f"{move.uci()} policy", force=True)
        t_next = time.time()
        n_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            a = self.policy.act(obs, info["task"])
            obs, r, terminated, truncated, info = env.step(a)
            n_steps += 1
            self._sync_viewer()
            t_next = self._step_wall(t_next)
            self._tick(f"step {n_steps}: {info['phase']}")
        rec["sim_s"] = s.data.time - t_sim0
        rec["steps"] = n_steps
        rec["piece_err_mm"] = dict(info["piece_err_mm"])
        rec["success"] = bool(info["success"])
        if not info["success"]:
            rec["reasons"].append(info["failure"] or f"not completed in {n_steps} steps (phase {info['phase']})")
        self.board.push(move)
        if not self.physical_matches():
            if rec["success"]:
                rec["reasons"].append("physical board differs from the logical position after the move")
            rec["success"] = False
            self.resync("physical/logical mismatch")
        rec["wall_s"] = time.time() - t0
        return rec

    def arm_move(self, candidates: List[chess.Move]) -> dict:
        """Plan the first plannable candidate, execute it with the arm, return a record.  Falls back to a
        teleport of candidates[0] when no candidate can be planned or the execution fails.  With a policy
        executor the engine's first candidate is executed by the policy instead (policy_move)."""
        s = self.scene
        if self.policy is not None:
            return self.policy_move(candidates[0])
        rec = dict(ply=len(self.board.move_stack) + 1, side="white" if self.board.turn else "black", executor="arm",
                   candidates=[m.uci() for m in candidates], plan_errors=[], success=False, reasons=[], plan_s=0.0,
                   sim_s=0.0, wall_s=0.0, piece_err_mm={})
        plan: Optional[MovePlan] = None
        move = candidates[0]
        t0 = time.time()
        for mv in candidates:
            self.note(f"{mv.uci()} planning", force=True)
            try:
                plan = self.expert.plan_move(mv, self.board)
                move = mv
                break
            except PlanError as e:
                rec["plan_errors"].append(f"{mv.uci()}: {str(e)[:160]}")
        rec["plan_s"] = time.time() - t0
        rec["uci"], rec["san"] = move.uci(), self.board.san(move)
        if plan is None:
            rec["executor"] = "teleport"
            rec["reasons"].append("no candidate move could be planned")
            self.say(f"  ! arm cannot plan any of {rec['candidates']}: {rec['plan_errors'][0] if rec['plan_errors'] else ''}")
            self.note(f"{move.uci()} teleport (not plannable)", force=True)
            self.teleport_move(move)
            self.board.push(move)
            if not self.physical_matches():
                rec["reasons"].append("physical board differs from the logical position after the teleport")
                self.resync("physical/logical mismatch after teleport")
                if not self.physical_matches():
                    rec["reasons"].append("position not representable in the scene (resync failed)")
            rec["wall_s"] = time.time() - t0
            return rec
        if move != candidates[0]:
            self.say(f"  (best move {candidates[0].uci()} not plannable, playing {move.uci()} instead)")
        if plan.notes:
            self.say("  plan notes:", "; ".join(plan.notes))
        # execute step by step (viewer sync + real-time pacing)
        before = {n: s.piece_pos(n) for n in s.piece_names}
        gen = self.expert.action_generator(plan)
        t_next = time.time()
        t_sim0 = s.data.time
        n_steps = 0
        self.note(f"{move.uci()} arm")
        for a in gen:
            s.control_step(a)
            n_steps += 1
            self._sync_viewer()
            t_next = self._step_wall(t_next)
            self._tick(f"step {n_steps}: {self.expert.state.get('label', '')}")
        rec["sim_s"] = s.data.time - t_sim0
        rec["steps"] = n_steps
        if self.expert.state.get("abort"):
            rec["reasons"].append(self.expert.state["abort"])
        # success per the spec, evaluated here (same criteria as Expert.execute)
        for n, tgt in plan.expected.items():
            err = float(np.linalg.norm(s.piece_pos(n)[:2] - np.asarray(tgt)[:2]))
            rec["piece_err_mm"][n] = err * 1000
            if err > C.SUCCESS_POS_TOL:
                rec["reasons"].append(f"{n} is {err*1000:.1f} mm from its target")
            if s.piece_tilt_deg(n) > C.SUCCESS_TILT_DEG:
                rec["reasons"].append(f"{n} tilted {s.piece_tilt_deg(n):.1f} deg")
        for n, p0 in before.items():
            if n in plan.expected:
                continue
            d = float(np.linalg.norm(s.piece_pos(n)[:2] - p0[:2]))
            if d > C.DISTURB_TOL:
                rec["reasons"].append(f"{n} displaced {d*1000:.1f} mm")
        if plan.watch_error:
            rec["reasons"].append(f"watch return not planned: {plan.watch_error[:100]}")
        elif plan.ends_at_watch:
            werr = float(np.max(np.abs(s.get_joints()[:5] - np.asarray(plan[-1].q6)[:5])))
            if werr > 3.0:
                rec["reasons"].append(f"not at the watch pose ({werr:.1f} deg)")
        rec["success"] = not rec["reasons"]
        self.board.push(move)
        if not self.physical_matches():
            rec["reasons"].append("physical board differs from the logical position after the move")
            rec["success"] = False
            self.resync("physical/logical mismatch")
        rec["wall_s"] = time.time() - t0
        return rec

    def play_ply(self, human_move: Optional[chess.Move] = None) -> dict:
        """One ply: the side to move plays (human/engine), executed by the arm if that colour is the arm's."""
        colour = self.board.turn
        if colour in self.arm_colours:
            cands = [human_move] if human_move is not None else self.engine_candidates()
            rec = self.arm_move(cands)
        else:
            t0 = time.time()
            move = human_move if human_move is not None else self.engine.play(self.board, self.limit).move
            rec = dict(ply=len(self.board.move_stack) + 1, side="white" if colour else "black", executor="teleport",
                       uci=move.uci(), san=self.board.san(move), success=True, reasons=[], plan_s=0.0, sim_s=0.0, piece_err_mm={})
            self.note(f"{move.uci()} teleport")
            self.teleport_move(move)
            # the arm 'looks at the board' and infers the opponent's move from what it sees
            seen = self.observe_opponent_move(move)
            rec["observed"] = seen.uci() if seen else None
            if seen != move:
                rec["reasons"].append(f"board reader inferred {seen.uci() if seen else None} instead of {move.uci()}")
                rec["success"] = False
            self.board.push(move)
            if not self.physical_matches():
                self.resync("after teleport")
            rec["wall_s"] = time.time() - t0
        self.records.append(rec)
        ok = sum(r["success"] for r in self.records if r.get("candidates") is not None)
        n_arm = sum(1 for r in self.records if r.get("candidates") is not None)
        self.note(f"{rec['san']} {'OK' if rec['success'] else 'FAIL'} (arm {ok}/{n_arm})", force=True)
        self.say(f"{rec['ply']:3d}. {rec['side']:5s} {rec['san']:7s} ({rec['uci']}) {rec['executor']:8s} "
                 f"{'OK' if rec['success'] else 'FAIL'}  {'; '.join(rec['reasons'])[:200]}  "
                 f"[plan {rec['plan_s']:.1f}s sim {rec['sim_s']:.1f}s wall {rec['wall_s']:.1f}s]")
        return rec

    def summary(self) -> dict:
        arm = [r for r in self.records if r.get("candidates") is not None]      # plies the arm was asked to play
        ok = sum(r["success"] for r in arm)
        reads = [r for r in self.records if "observed" in r]
        return dict(plies=len(self.records), arm_moves=len(arm), arm_success=ok, wall_s=sum(r["wall_s"] for r in self.records),
                    sim_s=sum(r["sim_s"] for r in self.records), result=self.board.result(claim_draw=True),
                    game_over=self.board.is_game_over(claim_draw=True), fen=self.board.fen(),
                    reads=len(reads), reads_ok=sum(1 for r in reads if r["observed"] == r["uci"]), reader=self.reader.name,
                    executor="policy" if self.policy is not None else "expert")


# ---------------------------------------------------------------------- modes
DEFAULT_PLIES_ESTIMATE = 80         # bar total for an open-ended game (a game averages ~80 plies)


def start_progress(game: Game, plies: Optional[int], path: Optional[str] = None, mode: str = "auto") -> ProgressBar:
    """Attach the one-line progress bar (chess_sim/data/progress.txt by default) to the game."""
    already = len(game.board.move_stack)
    total = already + (plies if plies is not None else DEFAULT_PLIES_ESTIMATE)
    game.progress = ProgressBar(total, **({"path": path} if path else {}))
    game.plies_target = plies
    game.note(f"{mode} start", force=True)
    return game.progress


def finish_progress(game: Game, summ: dict, ok: bool = True) -> None:
    if game.progress is None:
        return
    state = f"play_game: {summ['plies']} plies, arm {summ['arm_success']}/{summ['arm_moves']} OK, result {summ['result']}"
    if ok:
        # an open-ended game (no --plies) or one that ended early: the final count is the plies actually played,
        # not the estimate the bar was started with (ProgressBar needs total >= 1, so 0 plies reads '1/1')
        game.progress.total = max(1, len(game.board.move_stack))
        game.progress.done(state + " DONE")
    else:
        game.progress.fail(state + " FAILED")


def run_auto(game: Game, plies: Optional[int], progress_path: Optional[str] = None) -> dict:
    if game.progress is None:
        start_progress(game, plies, progress_path, "auto")
    n = 0
    try:
        while not game.board.is_game_over(claim_draw=True) and (plies is None or n < plies):
            game.play_ply()
            n += 1
    except BaseException:
        if game.progress is not None:
            game.progress.fail(f"play_game: aborted at ply {len(game.board.move_stack) + 1} FAILED")
        raise
    summ = game.summary()
    finish_progress(game, summ)
    return summ


def _reader(q: "queue.Queue[str]"):
    while True:
        try:
            line = input()
        except EOFError:
            q.put("quit")
            return
        q.put(line)


def run_interactive(game: Game, plies: Optional[int], progress_path: Optional[str] = None) -> dict:
    if game.progress is None:
        start_progress(game, plies, progress_path, "interactive")
    try:
        summ = _run_interactive(game, plies)
    except BaseException:
        if game.progress is not None:
            game.progress.fail(f"play_game: aborted at ply {len(game.board.move_stack) + 1} FAILED")
        raise
    finish_progress(game, summ)
    return summ


def _run_interactive(game: Game, plies: Optional[int]) -> dict:
    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_reader, args=(q,), daemon=True).start()
    print("You are White. Type a move (UCI e2e4 or SAN Nf3), 'hint', 'auto' (Stockfish plays this white move), 'fen', 'quit'.")
    n = 0
    while not game.board.is_game_over(claim_draw=True) and (plies is None or n < plies):
        if game.viewer is not None and not game.viewer.is_running():
            break
        if game.board.turn in game.arm_colours:
            game.play_ply()
            n += 1
            continue
        print(f"\n{game.board}\nWhite to move> ", end="", flush=True)
        game.note("waiting for White's move (terminal input)", force=True)
        line = None
        while line is None:
            try:
                line = q.get(timeout=0.05)
            except queue.Empty:
                game.idle(C.CONTROL_DT)
                if game.viewer is not None and not game.viewer.is_running():
                    line = "quit"
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        if line.lower() == "fen":
            print(game.board.fen())
            continue
        if line.lower() == "hint":
            print("hint:", game.board.san(game.engine.play(game.board, game.limit).move))
            continue
        if line.lower() == "auto":
            game.play_ply()
            n += 1
            continue
        try:
            try:
                mv = chess.Move.from_uci(line)
            except ValueError:
                mv = game.board.parse_san(line)
            if mv not in game.board.legal_moves:
                raise ValueError("illegal")
        except ValueError:
            print(f"illegal or unparsable move {line!r}; legal: {' '.join(game.board.san(m) for m in game.board.legal_moves)}")
            continue
        game.play_ply(mv)
        n += 1
    return game.summary()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stockfish-path", default=None)
    ap.add_argument("--skill", type=int, default=5, help="Stockfish Skill Level 0..20")
    ap.add_argument("--time", type=float, default=0.1, help="seconds per engine move")
    ap.add_argument("--depth", type=int, default=None, help="fixed search depth instead of --time")
    ap.add_argument("--multipv", type=int, default=5, help="alternative engine moves tried when a move cannot be planned")
    ap.add_argument("--auto", action="store_true", help="Stockfish plays both sides")
    ap.add_argument("--arm-white", action="store_true", help="the arm also executes White's moves")
    ap.add_argument("--headless", action="store_true", help="no window (implies --auto)")
    ap.add_argument("--no-viewer", action="store_true",
                    help="interactive terminal game without a window (plain python; pipe moves on stdin to script it)")
    ap.add_argument("--plies", type=int, default=None, help="stop after this many plies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomise", action="store_true", help="domain randomisation at the start")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed factor (1 = real time)")
    ap.add_argument("--fen", default=None, help="start position (validated before the engine starts)")
    ap.add_argument("--progress-path", default=None, help=f"progress bar file (default {C.PROGRESS_TXT})")
    ap.add_argument("--policy-path", default=None, help="<checkpoint>/pretrained_model: a LeRobot policy executes the arm's moves")
    ap.add_argument("--policy-device", default="cpu", help="cpu or mps")
    ap.add_argument("--policy-max-steps", type=int, default=900, help="env steps allowed per policy move")
    ap.add_argument("--reader", default="privileged", choices=["privileged", "vision"], help="board reader (board_reader.py)")
    a = ap.parse_args(argv)
    start_board = None
    if a.fen:
        try:
            start_board = parse_start_fen(a.fen)
        except ValueError as e:
            print(f"error: {e}")
            return 1
    policy = None
    if a.policy_path:
        from chess_sim.run_policy import PolicyExecutor
        try:
            policy = PolicyExecutor(a.policy_path, a.policy_device)
        except (FileNotFoundError, OSError) as e:
            print(f"error: cannot load the policy: {e}")
            return 1
    scene = ChessScene()
    viewer = None
    if not a.headless and not a.no_viewer:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
        except RuntimeError as e:
            print(f"cannot open the MuJoCo viewer ({e}).\nOn macOS run it with mjpython:\n"
                  f"  mjpython {os.path.abspath(__file__)} ...\n"
                  f"or use --headless.")
            return 2
    arm = [chess.BLACK] + ([chess.WHITE] if a.arm_white else [])
    game = Game(scene, a.stockfish_path, a.skill, a.time, a.depth, a.multipv, arm, viewer, realtime=viewer is not None,
                speed=a.speed, seed=a.seed, randomise=a.randomise, policy=policy, reader=make_reader(a.reader))
    t0 = time.time()
    try:
        if start_board is not None:
            game.board = start_board
            scene.set_position(game.board, settle_steps=50)
        print(f"stockfish: {game.engine_path} (skill {a.skill}, {'depth %d' % a.depth if a.depth else '%.2f s/move' % a.time}); "
              f"arm plays {'both colours' if a.arm_white else 'Black'} by the {'policy ' + a.policy_path if policy else 'scripted expert'}; "
              f"board reader {a.reader}; {'headless' if a.headless else ('terminal, no viewer' if a.no_viewer else 'viewer')}")
        if a.headless or a.auto:
            summ = run_auto(game, a.plies, a.progress_path)
        else:
            summ = run_interactive(game, a.plies, a.progress_path)
    finally:
        game.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
    summ["total_wall_s"] = time.time() - t0
    print(f"\nSUMMARY plies {summ['plies']}  arm moves {summ['arm_success']}/{summ['arm_moves']} successful  "
          f"result {summ['result']}  sim {summ['sim_s']:.1f}s  total wall {summ['total_wall_s']:.1f}s  "
          f"board reads {summ['reads_ok']}/{summ['reads']} ({summ['reader']})  executor {summ['executor']}")
    for r in game.records:
        if r.get("candidates") is not None and not r["success"]:
            print(f"  failed: ply {r['ply']} {r['uci']}: {'; '.join(r['reasons'])}")
    return 0 if summ["arm_success"] == summ["arm_moves"] else 1


if __name__ == "__main__":
    sys.exit(main())
