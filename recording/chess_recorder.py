"""Per-move demonstration recorder for the SO-101 chess project (leader -> follower + wrist camera).

    python recording/chess_recorder.py   (from a terminal that has camera permission)

Stays connected for the whole session and tracks the game on a python-chess board, so every episode gets its own
task text, e.g. "move the black knight from g8 to f6". All episodes go into ONE dataset that grows across sessions
(LeHak0/so101_chess_moves, private), uploaded to the Hub when you quit.

Commands at the prompt:
    g8f6 / Nf6     the ARM makes this move (recorded). A capture is two episodes (taken piece to the drop zone,
                   then the move), castling is two (king, then rook), a promotion is two (pawn to the drop zone,
                   then a spare queen onto the square).
    h e2e4         a move YOU make by hand (not recorded; just updates the board)
    s              ask Stockfish for a move for the side to move
    u              undo the last move on the board (does not delete recorded episodes)
    n              new game (start position)     fen <FEN>   set a position
    b              show the board                 q           save, upload and quit
During an episode:  B = start / stop recording,  R = redo this episode,  Q = abort this move.
In GET READY:       P = save the current pose as the ready pose (every episode should start there); a live line
                    shows how far each joint is from it and turns green when you are within a few degrees.
"""

import argparse
import json
import os
import select
import shutil
import sys
import termios
import time
import tty
from pathlib import Path

import chess
import chess.engine
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root
import so101_config  # noqa: E402  (ports and Stockfish path from config.yaml)

HERE = Path(__file__).resolve().parent
STATE = HERE / "board_state.json"
READY = HERE / "ready_pose.json"          # saved with P during GET READY
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "wrist",
         "wrist_roll": "roll", "gripper": "grip"}
READY_TOL = {"shoulder_pan": 4.0, "shoulder_lift": 4.0, "elbow_flex": 4.0, "wrist_flex": 5.0, "wrist_roll": 8.0, "gripper": 10.0}
FPS = 30
# Follower position gain. LeRobot's default is 16; a test on 2026-10-15 (calibration/shake_test.py) showed P=32 halves
# the tracking error and cuts lag ~40 % with no oscillation, but in real teleop it passed the hand's tremor straight
# through and felt less controllable, so it was reverted after episodes 53-55 (per-episode record in
# recording/episode_meta.jsonl). Use the same value when running a trained policy.
FOLLOWER_P = 24   # 2026-11-02: far-reach gain test (calibration/shake_output/gain_comparison_far_20261102.json): P=24 takes the
                  # full-stretch sag from 0.94 to 0.18 deg and halves elbow/wrist lag; with the 75 Hz teleop thread. History: 16
                  # (LeRobot default) from 2026-10-15, P=32 felt twitchy then (episodes 53-55 of the moves dataset are P=32)
PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
         chess.QUEEN: "queen", chess.KING: "king"}
C_GREEN, C_YELLOW, C_RED, C_BLUE, C_DIM, C_OFF = "\033[1;30;42m", "\033[1;30;43m", "\033[1;37;41m", "\033[1;37;44m", "\033[2m", "\033[0m"


def banner(color, text):
    print(f"\n{color}  {text}  {C_OFF}\n", flush=True)


# ------------------------------------------------------------------ move -> episodes
def colour(b: bool) -> str:
    return "white" if b == chess.WHITE else "black"


def episodes_for(board: chess.Board, move: chess.Move):
    """List of (task text, env_state vector) for the arm to perform this move."""
    sq = chess.square_name
    piece = board.piece_at(move.from_square)
    who, what = colour(piece.color), PIECE[piece.piece_type]
    eps = []
    if board.is_capture(move):
        cap_sq = move.to_square
        if board.is_en_passant(move):
            cap_sq = move.to_square + (-8 if piece.color == chess.WHITE else 8)
        cap = board.piece_at(cap_sq)
        eps.append((f"move the {colour(cap.color)} {PIECE[cap.piece_type]} on {sq(cap_sq)} to the drop zone",
                    env_state(cap_sq, None, to_drop=True)))
    if board.is_castling(move):
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rank = chess.square_rank(move.from_square)
        r_from = chess.square(7 if kingside else 0, rank)
        r_to = chess.square(5 if kingside else 3, rank)
        eps.append((f"move the {who} king from {sq(move.from_square)} to {sq(move.to_square)}",
                    env_state(move.from_square, move.to_square)))
        eps.append((f"move the {who} rook from {sq(r_from)} to {sq(r_to)}", env_state(r_from, r_to)))
        return eps
    if move.promotion:
        eps.append((f"move the {who} pawn on {sq(move.from_square)} to the drop zone",
                    env_state(move.from_square, None, to_drop=True)))
        eps.append((f"place the spare {who} {PIECE[move.promotion]} on {sq(move.to_square)}",
                    env_state(None, move.to_square, from_spare=True)))
        return eps
    eps.append((f"move the {who} {what} from {sq(move.from_square)} to {sq(move.to_square)}",
                env_state(move.from_square, move.to_square)))
    return eps


def env_state(from_sq, to_sq, to_drop=False, from_spare=False):
    """130-d conditioning vector: one-hot from-square (64), one-hot to-square (64), to-drop-zone flag, from-spare flag.
    (The sim's SO101ChessMove-v0 uses the first 128 dims with the same layout.)"""
    v = np.zeros(130, np.float32)
    if from_sq is not None:
        v[from_sq] = 1
    if to_sq is not None:
        v[64 + to_sq] = 1
    v[128], v[129] = float(to_drop), float(from_spare)
    return v


# ------------------------------------------------------------------ keys
class Keys:
    """Single-key reading from the terminal (cbreak mode) while the control loop runs."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.last = 0.0
        return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(self.fd, 32).decode(errors="ignore")
            now = time.monotonic()
            if now - self.last < 0.6:  # one physical press = one action (ignore repeats)
                return None
            for c in ch:
                if c.lower() in "brqp":
                    self.last = now
                    return c.lower()
        return None


# ------------------------------------------------------------------ hardware + dataset
class Rig:
    def __init__(self, repo_id, root, camera_index, width, height):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.processor import make_default_processors
        from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
        from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader
        from lerobot.utils.feature_utils import combine_feature_dicts

        cams = {"wrist": OpenCVCameraConfig(index_or_path=camera_index, width=width, height=height, fps=FPS)}
        self.robot = SOFollower(SO101FollowerConfig(port=so101_config.FOLLOWER_PORT, id=so101_config.FOLLOWER_ID, cameras=cams,
                                                    position_p_coefficient=FOLLOWER_P))
        self.teleop = SOLeader(SO101LeaderConfig(port=so101_config.LEADER_PORT, id=so101_config.LEADER_ID))
        self.t_proc, self.r_proc, self.o_proc = make_default_processors()
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(pipeline=self.t_proc, use_videos=True,
                                                initial_features=create_initial_features(action=self.robot.action_features)),
            aggregate_pipeline_dataset_features(pipeline=self.o_proc, use_videos=True,
                                                initial_features=create_initial_features(observation=self.robot.observation_features)))
        features["observation.environment_state"] = {"dtype": "float32", "shape": (130,), "names": None}
        kw = dict(streaming_encoding=True, encoder_threads=2, image_writer_threads=4)
        if (Path(root) / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset.resume(repo_id, root=root, **kw)
            print(f"continuing dataset {repo_id}: {self.dataset.num_episodes} episodes so far")
        else:
            self.dataset = LeRobotDataset.create(repo_id, FPS, features=features, root=root,
                                                 robot_type=self.robot.name, use_videos=True, **kw)
            print(f"new dataset {repo_id}")
        # features filled from the robot / teleop (the move encoding is added separately in run())
        self.hw_features = {k: v for k, v in self.dataset.features.items() if k != "observation.environment_state"}
        self.ready = json.loads(READY.read_text()) if READY.exists() else None
        self.teleop.connect()
        self.robot.connect()

    def ready_report(self, obs):
        """(all joints within tolerance?, one-line coloured report of the offsets from the saved ready pose)"""
        if self.ready is None:
            return False, f"{C_DIM}  no ready pose saved yet: put the arm where every episode should start and press P{C_OFF}"
        parts, ok = [], True
        for j in JOINT_NAMES:
            d = obs[f"{j}.pos"] - self.ready[j]
            good = abs(d) <= READY_TOL[j]
            ok &= good
            unit = "%" if j == "gripper" else "°"
            parts.append(f"{'\033[32m' if good else '\033[33m'}{SHORT[j]} {d:+5.1f}{unit}\033[0m")
        head = "\033[1;32m  AT READY POSE  \033[0m" if ok else "\033[1;33m  move to ready pose: \033[0m"
        return ok, head + "  ".join(parts)

    def run(self, keys, task=None, env=None, record=False, ready_check=True):
        """Teleoperate at 30 Hz (recording frames if `record`) until a key: returns 'b', 'r' or 'q'.
        When not recording: shows the live offset from the saved ready pose; P saves the current pose as the ready
        pose; B away from the ready pose needs a second B (start anyway)."""
        from lerobot.utils.feature_utils import build_dataset_frame
        period, n, t0 = 1 / FPS, 0, time.perf_counter()
        while True:
            tick = time.perf_counter()
            obs = self.robot.get_observation()
            obs_p = self.o_proc(obs)
            act = self.teleop.get_action()
            act_p = self.t_proc((act, obs))
            self.robot.send_action(self.r_proc((act_p, obs)))
            if record:
                frame = {**build_dataset_frame(self.hw_features, obs_p, prefix="observation"),
                         **build_dataset_frame(self.dataset.features, act_p, prefix="action"),
                         "observation.environment_state": env, "task": task}
                self.dataset.add_frame(frame)
                n += 1
                if n % FPS == 0:
                    print(f"\r{C_DIM}  recording... {n / FPS:4.0f} s{C_OFF}", end="", flush=True)
            k = keys.poll()
            if not record:
                n += 1
                at_ready, line = self.ready_report(obs) if ready_check else (True, "")
                if ready_check and n % 6 == 0:
                    print("\r\033[K" + line, end="", flush=True)
                if k == "p":
                    self.ready = {j: round(float(obs[f"{j}.pos"]), 2) for j in JOINT_NAMES}
                    READY.write_text(json.dumps(self.ready, indent=1))
                    print("\r\033[K\033[1;32m  Ready pose saved.\033[0m", flush=True)
                    continue
                if k == "b" and self.ready is not None and not at_ready and not getattr(self, "_warned", False):
                    self._warned = True
                    print("\r\033[K\033[1;37;41m  Not at the ready pose: press B again to start anyway  \033[0m", flush=True)
                    continue
                if k:
                    self._warned = False
                    print()
                    return k
            elif k:
                print()
                return k
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def close(self, push, private=True):
        for dev in (self.robot, self.teleop):
            try:
                dev.disconnect()
            except Exception:
                pass
        try:
            self.dataset.finalize()
        except Exception as e:
            print("finalize:", e)
        if push and self.dataset.num_episodes > 0:
            banner(C_BLUE, "Uploading the dataset to Hugging Face (private)...")
            self.dataset.push_to_hub(private=private)
            print("uploaded")


def log_episode(index, task, **extra):
    """Side record of recording conditions per episode (follower gain etc.), next to the dataset."""
    with (HERE / "episode_meta.jsonl").open("a") as f:
        ready = json.loads(READY.read_text()) if READY.exists() else None
        f.write(json.dumps({"episode_index": index, "task": task, "follower_p": FOLLOWER_P, "ready_pose": ready,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}) + "\n")


# ------------------------------------------------------------------ session
def load_board():
    if STATE.exists():
        s = json.loads(STATE.read_text())
        b = chess.Board(s["fen0"])
        for m in s["moves"]:
            b.push_uci(m)
        return b, s["fen0"]
    return chess.Board(), chess.STARTING_FEN


def save_board(board, fen0):
    STATE.write_text(json.dumps({"fen0": fen0, "moves": [m.uci() for m in board.move_stack]}, indent=1))


def parse_move(board, text):
    try:
        m = chess.Move.from_uci(text)
        if m in board.legal_moves:
            return m
    except ValueError:
        pass
    try:
        return board.parse_san(text)
    except ValueError:
        return None


def show(board):
    print()
    rows = str(board).split("\n")
    for i, r in enumerate(rows):
        print(f"  {8 - i}  {r}")
    print("     a b c d e f g h")
    print(f"  {colour(board.turn)} to move" + ("  (game over: " + board.result() + ")" if board.is_game_over() else ""))


def record_move(rig, board, move):
    """Record every episode of a move. Returns True if all were saved."""
    eps = episodes_for(board, move)
    with __import__("contextlib").ExitStack() as st:
        keys = st.enter_context(Keys())
        for i, (task, env) in enumerate(eps):
            part = f" (part {i + 1} of {len(eps)})" if len(eps) > 1 else ""
            while True:
                banner(C_YELLOW, f"GET READY{part}: {task}.  Arm in the ready pose, then press B to START")
                k = rig.run(keys)
                if k == "q":
                    banner(C_RED, "Move aborted (nothing saved for this part)")
                    return False
                if k != "b":
                    continue
                banner(C_GREEN, f">>> RECORDING{part}: {task}.  Press B when done (R = redo)")
                k = rig.run(keys, task, env, record=True)
                if k == "b" and rig.dataset.has_pending_frames():
                    rig.dataset.save_episode()
                    log_episode(rig.dataset.num_episodes - 1, task)
                    print(f"  saved episode {rig.dataset.num_episodes - 1}: \"{task}\"")
                    break
                rig.dataset.clear_episode_buffer()
                if k == "q":
                    banner(C_RED, "Move aborted (nothing saved for this part)")
                    return False
                banner(C_RED, "Redoing this episode")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="LeHak0/so101_chess_moves")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    root = Path.home() / ".cache/huggingface/lerobot" / args.repo_id

    board, fen0 = load_board()
    engine = None
    sf = so101_config.STOCKFISH
    rig = Rig(args.repo_id, root, args.camera, args.width, args.height)
    try:
        print(__doc__.split("Commands at the prompt:")[1])
        show(board)
        while True:
            try:
                cmd = input(f"\n{colour(board.turn)} to move > ").strip()
            except EOFError:
                break
            if not cmd:
                continue
            if cmd == "q":
                break
            if cmd == "b":
                show(board); continue
            if cmd == "u":
                if board.move_stack:
                    print("undid", board.pop().uci()); save_board(board, fen0)
                show(board); continue
            if cmd == "n":
                board, fen0 = chess.Board(), chess.STARTING_FEN; save_board(board, fen0); show(board); continue
            if cmd.startswith("fen "):
                try:
                    board = chess.Board(cmd[4:].strip()); fen0 = board.fen(); save_board(board, fen0); show(board)
                except ValueError as e:
                    print("bad FEN:", e)
                continue
            if cmd == "s":
                if engine is None and os.path.exists(sf):
                    engine = chess.engine.SimpleEngine.popen_uci(sf)
                if engine:
                    mv = engine.play(board, chess.engine.Limit(time=0.2)).move
                    print(f"Stockfish suggests {board.san(mv)} ({mv.uci()}): " + "; ".join(t for t, _ in episodes_for(board, mv)))
                else:
                    print("Stockfish not found")
                continue
            by_hand = cmd.startswith("h ")
            mv = parse_move(board, cmd[2:].strip() if by_hand else cmd)
            if mv is None:
                print(f"not a legal move here: {cmd!r}  (legal: {', '.join(board.san(m) for m in list(board.legal_moves)[:12])}...)")
                continue
            if by_hand:
                board.push(mv); save_board(board, fen0); print("board updated"); show(board); continue
            print("the arm will do: " + "; then ".join(t for t, _ in episodes_for(board, mv)))
            if record_move(rig, board, mv):
                board.push(mv); save_board(board, fen0); show(board)
            print(f"{C_DIM}dataset: {rig.dataset.num_episodes} episodes{C_OFF}")
    finally:
        if engine:
            engine.quit()
        # Park before releasing: the follower goes limp on disconnect, so it must be folded (stable) first.
        banner(C_BLUE, "Before saving: bring the LEADER to the folded rest pose (gripper up), then press B")
        try:
            with Keys() as keys:
                while rig.run(keys, ready_check=False) != "b":
                    pass
        except Exception as e:
            print("parking step skipped:", e)
        rig.close(push=not args.no_push)
        save_board(board, fen0)
        print("done")


if __name__ == "__main__":
    main()
