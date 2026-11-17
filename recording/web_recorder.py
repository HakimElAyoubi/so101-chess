"""Web UI for the per-move chess demo recorder (leader -> follower + wrist camera).

    scripts/record_demos.sh   (from a terminal with camera permission; the page opens in your browser)

The app plays a whole game and picks every move itself (Stockfish, with randomness for variety):
    WHITE  -> it tells you which white move to make BY HAND on the real board, you click "Done"
    BLACK  -> it tells you which black move to demonstrate with the leader arm; Start / Stop records one episode
Same dataset and format as chess_recorder.py (LeHak0/so101_chess_moves, task text + 130-d move encoding),
uploaded to the Hub (private) when you click "Finish session".

2026-11-02: the leader -> follower link runs in its own thread at TELEOP_HZ (75 Hz) with one lock per serial bus, the
follower written only in the phases where the leader drives it (checked under the lock); the 30 Hz loop keeps sampling
the camera and writing the dataset frames, their action being the thread's latest leader action. Every other bus access
(auto-move to the ready pose, leader torque on/off, closing) goes under the same locks. Ported from grasp_recorder.py.
The page shows the achieved rate ("leader→follower N Hz, M ms"). --teleop-hz 0 is the old single loop.
--fake: simulated arms and camera (grasp_recorder.FakeRig), the dataset and the state files under --root: a dry run.
"""

import argparse
import json
import queue
import random
import shutil
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.engine
import cv2
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root
import so101_config  # noqa: E402  (ports and Stockfish path from config.yaml)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chess_recorder import (FPS, JOINT_NAMES, PIECE, READY, READY_TOL, SHORT, Rig, colour, env_state,  # noqa: E402
                            episodes_for, load_board, log_episode, save_board)

HERE = Path(__file__).resolve().parent
PAGE = HERE / "web_ui.html"
DRILL_FILE = HERE / "drill_40.json"            # from drill_plan.py: targeted episodes that fill dataset gaps
DRILL_PROGRESS = HERE / "drill_progress.json"  # next drill index + which dataset episode each drill became
TELEOP_HZ = 75.0                               # the leader -> follower link runs in its own thread at this rate (0: inside the 30 Hz loop, the old way)
# the phases in which the leader drives the follower: not while an auto-move runs (home_step moves both arms) and not
# while the buses are being closed (closing / done)
TELEOP_PHASES = ("setup", "white", "ready", "recording", "saving", "parking", "drill_setup")


class Drill:
    """Walks through drill_40.json in order (resumable). The board is tracked from the one-time setup so the page
    can draw it; spare queens are taken off by hand after their episode."""

    def __init__(self):
        d = json.loads(DRILL_FILE.read_text())
        self.eps, self.setup_black, self.setup_white = d["episodes"], d["setup_black"], d["setup_white"]
        prog = json.loads(DRILL_PROGRESS.read_text()) if DRILL_PROGRESS.exists() else {}
        self.idx, self.done = int(prog.get("next", 0)), prog.get("done", {})

    @property
    def n(self):
        return len(self.eps)

    def save(self):
        DRILL_PROGRESS.write_text(json.dumps({"next": self.idx, "done": self.done}, indent=1))

    def board_at(self, k):
        """python-chess board before drill k (start position for black + the listed white pieces)."""
        b = chess.Board(None)
        sym = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}
        for sq, p in self.setup_black.items():
            b.set_piece_at(chess.parse_square(sq), chess.Piece.from_symbol(sym[p]))
        for sq, p in self.setup_white.items():
            b.set_piece_at(chess.parse_square(sq), chess.Piece.from_symbol(sym[p].upper()))
        for e in self.eps[:k]:
            if e["frm"] and e["to"]:
                b.set_piece_at(chess.parse_square(e["to"]), b.remove_piece_at(chess.parse_square(e["frm"])))
            elif e["frm"]:
                b.remove_piece_at(chess.parse_square(e["frm"]))
        return b

    def episode(self):
        e = self.eps[self.idx]
        sq = lambda x: chess.parse_square(x) if x else None
        return [(e["task"], env_state(sq(e["frm"]), sq(e["to"]), to_drop=e["to_drop"], from_spare=e["from_spare"]))]

    def setup_text(self):
        w = ", ".join(f"{p} {sq}" for sq, p in self.setup_white.items())
        return f"Black: the normal starting position (all 16 pieces). White: only {w}. Everything else off the board."


def describe_by_hand(board: chess.Board, move: chess.Move) -> str:
    """Plain-English instruction for a move the user makes by hand."""
    sq = chess.square_name
    p = board.piece_at(move.from_square)
    txt = f"Move the {colour(p.color)} {PIECE[p.piece_type]} from {sq(move.from_square)} to {sq(move.to_square)}"
    if board.is_castling(move):
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        r = chess.square_rank(move.from_square)
        txt += f", and the rook from {sq(chess.square(7 if kingside else 0, r))} to {sq(chess.square(5 if kingside else 3, r))}"
    elif board.is_capture(move):
        cap_sq = move.to_square
        if board.is_en_passant(move):
            cap_sq = move.to_square + (-8 if p.color == chess.WHITE else 8)
        c = board.piece_at(cap_sq)
        txt += f", taking the {colour(c.color)} {PIECE[c.piece_type]} on {sq(cap_sq)} off the board"
    if move.promotion:
        txt += f" and swap it for a {PIECE[move.promotion]}"
    return txt + "."


class Session:
    """Owns the hardware. The 30 Hz control loop (follower + camera + dataset) runs in its own thread and the leader ->
    follower link in another (teleop_loop); the web handlers only queue commands."""

    def __init__(self, rig: Rig, push: bool):
        self.rig, self.push = rig, push
        self.cmds: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()
        self.jpeg = None
        self.board, self.fen0 = load_board()
        sf = so101_config.STOCKFISH
        self.engine = chess.engine.SimpleEngine.popen_uci(sf) if Path(sf).exists() else None
        self.phase = "setup"          # setup | white | ready | recording | saving | parking | closing | done
        self.move = None              # move currently being made (white by hand, or black by the arm)
        self.eps, self.part = [], 0   # episodes of the current black move
        self.rec_start = 0.0
        self.force_until = 0.0        # second Start within this time = start even if not at the ready pose
        self.session_eps = 0
        self.msg, self.msg_kind = "", "info"
        self.joints, self.ready_off, self.at_ready = {}, {}, False
        self.error = ""
        self.stop = False
        self.home = None              # active auto-move of both arms to the ready pose
        # After an auto-move the FOLLOWER keeps holding the exact ready pose and ignores the leader (the leader cannot
        # reach that pose: its handle hits its own arm). Start / Follow then glides the follower onto the leader.
        self.hold = False
        self.engage = None            # {"t0", "T"}: follower blending from the ready pose onto the live leader
        self.lead = {}                # latest leader joint positions
        self.drill = None             # Drill while in drill mode
        _d = Drill() if DRILL_FILE.exists() else None
        self.drill_next, self.drill_n = (_d.idx, _d.n) if _d else (0, 0)
        self.lk_fol, self.lk_lead = threading.Lock(), threading.Lock()     # one serial transaction at a time per bus
        self.teleop_hz, self.obs_latest = TELEOP_HZ, None   # the thread works on the loop's latest observation
        self.last_act, self.last_act_t = None, 0.0          # the thread's latest leader action (what the dataset records) and when
        self.teleop_stats, self.teleop_error = {"hz": 0.0, "ms": 0.0}, ""

    # ---------------------------------------------------------------- moves
    def pick(self):
        """A good-but-varied move for the side to move: random among Stockfish's top moves within ~1 pawn."""
        legal = list(self.board.legal_moves)
        if not self.engine:
            return random.choice(legal)
        info = self.engine.analyse(self.board, chess.engine.Limit(time=0.12), multipv=5)
        scored = [(i["pv"][0], i["score"].pov(self.board.turn).score(mate_score=10000)) for i in info if i.get("pv")]
        best = max(s for _, s in scored)
        return random.choice([m for m, s in scored if s >= best - 100])

    def next_turn(self, note=""):
        if self.board.is_game_over():
            self.board, self.fen0 = chess.Board(), chess.STARTING_FEN
            save_board(self.board, self.fen0)
            self.phase, self.move = "setup", None
            self.say("Game over! Put all the pieces back in the starting position, then click the button.", "good")
            return
        self.move = self.pick()
        if self.board.turn == chess.WHITE:
            self.phase = "white"
        else:
            self.eps, self.part, self.phase = episodes_for(self.board, self.move), 0, "ready"
        self.say(note, "good" if note else "info")

    def next_drill(self, note):
        self.drill_next = self.drill.idx
        if self.drill.idx >= self.drill.n:
            self.drill = None; self.phase, self.move, self.eps, self.part = "setup", None, [], 0
            self.say("Drill complete: all targeted episodes recorded! Set the board to the game position to continue.", "good")
            return
        self.eps, self.part, self.phase = self.drill.episode(), 0, "ready"
        self.say(note, "good")

    def say(self, msg, kind="info"):
        self.msg, self.msg_kind = msg, kind

    # ---------------------------------------------------------------- commands (run in the control thread)
    def handle(self, c):
        ph = self.phase
        if c == "drill_start" and ph in ("setup", "white", "ready") and self.part == 0:
            self.drill = Drill()
            if self.drill.idx >= self.drill.n:
                self.drill = None; self.say("All drill episodes are already recorded.", "good"); return
            self.phase = "drill_setup"
            self.say("" if self.drill.idx == 0 else f"Resuming the drill at {self.drill.idx + 1} of {self.drill.n}: "
                     "set the board as drawn below.")
        elif c == "drill_setup_done" and ph == "drill_setup":
            self.eps, self.part, self.phase = self.drill.episode(), 0, "ready"
            self.say("")
        elif c == "drill_skip" and self.drill and ph == "ready":
            self.drill.idx += 1; self.drill.save(); self.next_drill("Skipped: make that move by hand on the board.")
        elif c == "drill_exit" and self.drill and ph in ("ready", "drill_setup"):
            self.drill_next = self.drill.idx
            self.drill = None; self.phase, self.move, self.eps, self.part = "setup", None, [], 0
            self.say("Left the drill (progress is kept). Set the board back to the game position shown, then start.")
        elif c == "start_game" and ph == "setup":
            self.next_turn()
        elif c == "white_done" and ph == "white":
            self.board.push(self.move); save_board(self.board, self.fen0)
            self.next_turn()
        elif c == "reroll" and not self.drill and (ph == "white" or (ph == "ready" and self.part == 0)):
            old = self.move
            for _ in range(4):
                self.next_turn()
                if self.move != old:
                    break
        elif c == "start" and ph == "ready" and self.hold:
            far = max(abs(self.lead[j] - self.rig.ready[j]) for j in JOINT_NAMES) if self.lead else 0
            if far > self.ENGAGE_MAX and time.monotonic() > self.force_until:
                self.force_until = time.monotonic() + 6
                self.say(f"The leader is {far:.0f}° from the ready pose: the follower would make a big jump at the start. "
                         "Bring the leader closer, or click Start again to record anyway.", "warn")
                return
            self.engage = {"t0": time.monotonic(), "T": self.ENGAGE_T}
            self.phase, self.rec_start, self.force_until = "recording", time.monotonic(), 0.0
            self.say("")
        elif c == "follow" and self.hold and ph in ("setup", "white", "ready", "drill_setup"):
            self.engage = {"t0": time.monotonic(), "T": self.ENGAGE_T}
            self.say("The follower is following the leader again.")
        elif c == "start" and ph == "ready":
            if self.rig.ready is not None and not self.at_ready and time.monotonic() > self.force_until:
                self.force_until = time.monotonic() + 6
                self.say("The arm is not at the ready pose. Click Start again to record anyway.", "warn")
                return
            self.phase, self.rec_start, self.force_until = "recording", time.monotonic(), 0.0
            self.say("")
        elif c == "stop" and ph == "recording":
            if not self.rig.dataset.has_pending_frames():
                self.phase = "ready"; return
            if self.rig.ready is not None and not self.at_ready and time.monotonic() > self.force_until:
                # the policy must learn to come back to the watch pose, so the episode should end there too
                self.force_until = time.monotonic() + 6
                self.say("Not back at the ready pose yet: bring the arm back (numbers green), then Stop. "
                         "Click Stop again to save anyway.", "warn")
                return
            self.force_until = 0.0
            self.phase = "saving"; self.publish()
            self.rig.dataset.save_episode()
            log_episode(self.rig.dataset.num_episodes - 1, self.eps[self.part][0],
                        auto_return=bool(getattr(self, "_auto_return", False)),
                        **({"drill": self.drill.idx} if self.drill else {}))
            self._auto_return = False
            self.session_eps += 1
            self.part += 1
            if self.drill:
                e = self.drill.eps[self.drill.idx]
                self.drill.done[str(self.drill.idx)] = self.rig.dataset.num_episodes - 1
                self.drill.idx += 1; self.drill.save()
                self.next_drill(f"Saved. Take the spare queen off {e['to']} by hand." if e["from_spare"] else "Saved!")
            elif self.part < len(self.eps):
                self.phase = "ready"
                self.say(f"Saved. Part {self.part + 1} of {len(self.eps)} next.", "good")
            else:
                self.board.push(self.move); save_board(self.board, self.fen0)
                self.next_turn("Saved!")
        elif c == "redo" and ph == "recording" and self.drill:
            self.rig.dataset.clear_episode_buffer(); self.eps, self.part, self.phase = self.drill.episode(), 0, "ready"
            self.say("Discarded. Put the piece back where it was, go to the ready pose and start again.", "warn")
        elif c == "redo" and ph == "recording":
            self.rig.dataset.clear_episode_buffer()
            self.phase = "ready"
            self.say("Discarded. Put the piece back where it was, go to the ready pose and start again.", "warn")
        elif c == "go_ready" and ph in ("setup", "white", "ready", "parking", "drill_setup"):
            if self.rig.ready is None:
                self.say("Save a ready pose first.", "warn"); return
            self.start_home(ph)
        elif c == "return_save" and ph == "recording":
            if self.rig.ready is None:
                self.say("Save a ready pose first.", "warn"); return
            self._auto_return = True
            self.start_home("recording", record=True)
        elif c == "cancel_home" and ph == "homing":
            self._auto_return = False
            self.end_home("Stopped. You have control of the leader again.")
        elif c == "save_ready" and ph not in ("recording", "saving", "closing", "done", "homing"):
            j = self.joints
            self.rig.ready = {n: round(float(j[n]), 2) for n in JOINT_NAMES}
            READY.write_text(json.dumps(self.rig.ready, indent=1))
            self.say("Ready pose saved. Every episode will start and end here.", "good")
        elif c == "undo" and not self.drill and ph in ("white", "ready", "setup") and (ph != "ready" or self.part == 0):
            if self.board.move_stack:
                m = self.board.pop(); save_board(self.board, self.fen0)
                self.next_turn(f"Took back {m.uci()}. Put that piece back on the real board.")
        elif c == "new_game" and not self.drill and ph in ("white", "ready", "setup"):
            self.board, self.fen0 = chess.Board(), chess.STARTING_FEN
            save_board(self.board, self.fen0)
            self.phase, self.move = "setup", None
            self.say("")
        elif c.startswith("fake_lead:") and getattr(self.rig, "fake", False):   # dry run: the 'human' moves the leader
            with self.lk_lead:
                self.rig.teleop.set(json.loads(c[10:]))
        elif c == "finish" and ph in ("white", "ready", "setup", "drill_setup"):
            self.before_park = ph
            self.phase = "parking"
            self.say("")
        elif c == "cancel_finish" and ph == "parking":
            self.phase = self.before_park
        elif c == "park_done" and ph == "parking":
            self.phase = "closing"
            self.say("Saving the dataset" + (" and uploading it to Hugging Face..." if self.push else "..."))
            self.publish()
            try:
                if self.engine:
                    self.engine.quit()
                with self.lk_lead, self.lk_fol:         # the thread idles in "closing"; no bus traffic while disconnecting
                    self.rig.close(push=self.push)
                self.say(f"All done: {self.rig.dataset.num_episodes} episodes in the dataset"
                         + (" (uploaded)." if self.push else "."), "good")
            except Exception as e:
                self.say(f"Problem while saving: {e}", "warn")
            self.phase, self.stop = "done", True

    # ---------------------------------------------------------------- auto-move both arms to the ready pose
    HOME_SPEED = 30.0                 # deg/s (gripper: %/s) at the fastest point of the move
    LEADER_WRIST_BACKOFF = 8.0        # deg: the leader's handle hits its own arm at the ready pose's wrist flex
    ENGAGE_T = 0.8                    # s: follower glide from the held ready pose onto the leader
    ENGAGE_MAX = 25.0                 # deg / %: warn before Start if the leader is further than this from ready

    def start_home(self, prev, record=False):
        """Leader motors on (holding where they are), then both arms glide to the ready pose; leader released at the end."""
        lead_bus = self.rig.teleop.bus
        with self.lk_lead:
            here = lead_bus.sync_read("Present_Position")
            lead_bus.sync_write("Goal_Position", here)      # hold the current pose: no jump when torque comes on
            lead_bus.enable_torque()
        with self.lk_fol:
            fol = {j: float(v) for j, v in self.rig.robot.bus.sync_read("Present_Position").items()}
        lead = {j: float(here[j]) for j in JOINT_NAMES}
        R = self.rig.ready
        dist = max(max(abs(R[j] - lead[j]), abs(R[j] - fol[j])) for j in JOINT_NAMES)
        T = max(1.5, 1.875 * dist / self.HOME_SPEED)    # min-jerk peak speed = 1.875 * distance / T
        self.home = {"prev": prev, "t0": time.monotonic(), "T": T, "lead": lead, "fol": fol, "record": record}
        self.phase = "homing"
        self.say(f"Moving both arms to the ready pose ({T:.0f} s)" + (", still recording" if record else "")
                 + ". Hands off the leader.", "info")

    def home_step(self, obs):
        h, R = self.home, self.rig.ready
        t = (time.monotonic() - h["t0"]) / h["T"]
        u = min(1.0, t)
        a = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        RL = dict(R); RL["wrist_flex"] = R["wrist_flex"] - self.LEADER_WRIST_BACKOFF
        with self.lk_lead:
            self.rig.teleop.send_feedback({f"{j}.pos": h["lead"][j] + (RL[j] - h["lead"][j]) * a for j in JOINT_NAMES})
        target = {f"{j}.pos": h["fol"][j] + (R[j] - h["fol"][j]) * a for j in JOINT_NAMES}
        with self.lk_fol:
            self.rig.robot.send_action(target)
        if h["record"]:                                  # the way back is part of the demonstration
            from lerobot.utils.feature_utils import build_dataset_frame
            rig = self.rig
            rig.dataset.add_frame({**build_dataset_frame(rig.hw_features, rig.o_proc(obs), prefix="observation"),
                                   **build_dataset_frame(rig.dataset.features, target, prefix="action"),
                                   "observation.environment_state": self.eps[self.part][1],
                                   "task": self.eps[self.part][0]})
        self.home_progress = u
        if t >= 1.0 + 0.4 / h["T"]:                    # 0.4 s settle, then hand the leader back
            if h["record"]:
                self.end_home("Back at the ready pose.")
                self.force_until = time.monotonic() + 1       # it is at the ready pose: no Stop warning
                self.handle("stop")
            else:
                self.end_home("Both arms are at the ready pose. The follower holds it until you press Start "
                              "(or Follow); the leader is free.")

    def end_home(self, msg):
        try:
            with self.lk_lead:
                self.rig.teleop.bus.disable_torque()
        except Exception as e:
            msg += f" (could not switch the leader motors off: {e})"
        self.phase = self.home["prev"] if self.home else "ready"
        self.home = None
        self.hold, self.engage = True, None          # follower stays exactly at the ready pose
        self.say(msg, "good")

    def held_command(self, act_p):
        """Follower target while holding: the ready pose; while engaging: min-jerk blend ready -> live leader."""
        R = self.rig.ready
        if self.engage:
            u = min(1.0, (time.monotonic() - self.engage["t0"]) / self.engage["T"])
            k = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            out = {f"{j}.pos": R[j] + (float(act_p[f"{j}.pos"]) - R[j]) * k for j in JOINT_NAMES}
            if u >= 1.0:
                self.engage, self.hold = None, False
            return out
        return {f"{j}.pos": float(R[j]) for j in JOINT_NAMES}

    # ---------------------------------------------------------------- leader -> follower link
    def teleop_target(self, act, obs):
        """The leader's action -> (what the dataset records, what the follower is sent): the live leader; or, after an
        auto-move, the held ready pose / the min-jerk blend from it onto the leader (held_command)."""
        act_p = self.rig.t_proc((act, obs))
        if self.hold or self.engage:
            act_p = self.held_command(act_p)          # ready pose, or blending onto the leader
            return act_p, act_p
        return act_p, self.rig.r_proc((act_p, obs))

    def teleop_loop(self):
        """The leader -> follower link in its own thread at TELEOP_HZ (as grasp_recorder.py since 2026-11-02): the follower
        gets a fresh target every ~13 ms instead of once per 30 Hz tick (the tick also reads the follower and the camera
        and writes the dataset). Each bus is used by one thread at a time (lk_lead, lk_fol), and the follower is only
        written in the phases where the leader drives it (TELEOP_PHASES), checked under the lock so a phase change
        (an auto-move starting, the rig closing) never races a stale target."""
        rig, period = self.rig, 1 / self.teleop_hz
        n, t_win, lat, due = 0, time.perf_counter(), [], time.perf_counter()
        while not self.stop:
            tick = time.perf_counter()
            due = max(due + period, tick)      # fixed-rate pacing: a late tick (GIL, a slow read) does not shorten the next
            obs = self.obs_latest              # period, and a very late one is not caught up by flooding the bus
            if self.phase in TELEOP_PHASES and obs is not None:
                try:
                    with self.lk_lead:
                        act = rig.teleop.get_action()
                    self.lead = {j: float(act[f"{j}.pos"]) for j in JOINT_NAMES}
                    act_p, cmd = self.teleop_target(act, obs)
                    with self.lk_fol:
                        if self.phase in TELEOP_PHASES:
                            rig.robot.send_action(cmd)
                    self.last_act, self.last_act_t = act_p, time.perf_counter()
                    lat.append(time.perf_counter() - tick); n += 1
                    self.teleop_error = ""
                except Exception as ex:
                    self.last_act, self.teleop_error = None, f"Teleop error: {ex}"
                    time.sleep(0.2)
            else:
                self.last_act = None
            if time.perf_counter() - t_win >= 1.0:
                self.teleop_stats = {"hz": round(n / (time.perf_counter() - t_win), 1),
                                     "ms": round(1000 * sum(lat) / len(lat), 1) if lat else 0.0}
                n, t_win, lat = 0, time.perf_counter(), []
            time.sleep(max(0.0, due - time.perf_counter()))

    # ---------------------------------------------------------------- control loop
    def loop(self):
        from lerobot.utils.feature_utils import build_dataset_frame
        rig, period, n = self.rig, 1 / FPS, 0
        if self.teleop_hz > 0:
            threading.Thread(target=self.teleop_loop, daemon=True, name="teleop").start()
        while not self.stop:
            tick = time.perf_counter()
            try:
                with self.lk_fol:
                    obs = rig.robot.get_observation()      # the follower's joints (bus) + the latest camera frame (no wait)
                self.obs_latest = obs
                if self.phase == "homing":
                    self.home_step(obs)
                    continue_teleop = False
                else:
                    continue_teleop = True
                    obs_p = rig.o_proc(obs)
                    # the recorded action is the thread's latest leader action (fresh within 0.1 s); the old single-loop
                    # way (--teleop-hz 0, or the thread has not run yet) reads the leader and writes the follower here
                    act_p = self.last_act if time.perf_counter() - self.last_act_t < 0.1 else None
                    if act_p is None:
                        with self.lk_lead:
                            act = rig.teleop.get_action()
                        self.lead = {j: float(act[f"{j}.pos"]) for j in JOINT_NAMES}
                        act_p, cmd = self.teleop_target(act, obs)
                        with self.lk_fol:
                            rig.robot.send_action(cmd)
                if continue_teleop and self.phase == "recording":
                    rig.dataset.add_frame({**build_dataset_frame(rig.hw_features, obs_p, prefix="observation"),
                                           **build_dataset_frame(rig.dataset.features, act_p, prefix="action"),
                                           "observation.environment_state": self.eps[self.part][1],
                                           "task": self.eps[self.part][0]})
                self.joints = {j: float(obs[f"{j}.pos"]) for j in JOINT_NAMES}
                if rig.ready:
                    self.ready_off = {j: self.joints[j] - rig.ready[j] for j in JOINT_NAMES}
                    self.at_ready = all(abs(self.ready_off[j]) <= READY_TOL[j] for j in JOINT_NAMES)
                    if self.lead:
                        self.lead_off = {j: self.lead[j] - rig.ready[j] for j in JOINT_NAMES}
                n += 1
                if n % 2 == 0 and "wrist" in obs:
                    img = cv2.resize(cv2.cvtColor(obs["wrist"], cv2.COLOR_RGB2BGR), (640, 360))
                    self.jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                self.error = ""
            except Exception as e:
                self.error = f"Arm/camera read failed: {e}"
                if self.phase == "homing":
                    self.end_home(f"Auto-move stopped (communication error: {e}).")
                time.sleep(0.2)
            while not self.cmds.empty():
                try:
                    self.handle(self.cmds.get_nowait())
                except Exception as e:
                    traceback.print_exc()
                    self.say(f"Error: {e}", "warn")
            self.publish()
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    # ---------------------------------------------------------------- state for the page
    def publish(self):
        b, m = self.board, self.move
        dr = self.drill
        if dr:
            b = dr.board_at(min(dr.idx, dr.n))
        cur = None
        if (self.phase in ("ready", "recording", "saving") or (self.phase == "homing" and self.home and self.home["record"])) \
                and self.eps and self.part < len(self.eps):
            task, env = self.eps[self.part]
            frm = int(env[:64].argmax()) if env[:64].any() else None
            to = int(env[64:128].argmax()) if env[64:128].any() else None
            cur = {"task": task[0].upper() + task[1:], "from": frm, "to": to,
                   "to_drop": bool(env[128]), "from_spare": bool(env[129])}
        st = {
            "phase": self.phase, "msg": self.msg, "msg_kind": self.msg_kind, "error": self.error or self.teleop_error,
            "fen": b.board_fen(), "turn": colour(b.turn), "move_no": b.fullmove_number,
            "white": ({"text": describe_by_hand(b, m), "from": m.from_square, "to": m.to_square}
                      if self.phase == "white" and m else None),
            "cur": cur, "part": self.part + 1, "parts": len(self.eps),
            "all_tasks": [t[0].upper() + t[1:] for t, _ in self.eps] if self.phase in ("ready", "recording", "saving") else [],
            "rec_s": round(time.monotonic() - self.rec_start, 1) if self.phase == "recording" or (self.home and self.home.get("record")) else 0,
            "ready_saved": self.rig.ready is not None, "at_ready": self.at_ready,
            "ready": [{"name": SHORT[j], "off": round(self.ready_off.get(j, 0), 1), "tol": READY_TOL[j],
                       "unit": "%" if j == "gripper" else "°"} for j in JOINT_NAMES],
            "force_pending": time.monotonic() < self.force_until,
            "home_pct": round(100 * getattr(self, "home_progress", 0)) if self.phase == "homing" else 0,
            "home_rec": bool(self.home and self.home.get("record")),
            "hold": bool(self.hold and not self.engage),
            "lead_ready": [{"name": SHORT[j], "off": round(getattr(self, "lead_off", {}).get(j, 0), 1), "tol": self.ENGAGE_MAX,
                            "unit": "%" if j == "gripper" else "°"} for j in JOINT_NAMES],
            "repo": getattr(self.rig.dataset, "repo_id", ""),
            "episodes": self.rig.dataset.num_episodes, "session_eps": self.session_eps,
            "last": b.peek().uci() if b.move_stack else None,
            "drill": ({"idx": dr.idx + 1, "n": dr.n, "setup": dr.setup_text()} if dr else None),
            "drill_next": (dr.idx if dr else self.drill_next), "drill_n": self.drill_n,
            "teleop": self.teleop_stats, "fake": bool(getattr(self.rig, "fake", False)),
        }
        with self.lock:
            self.state = json.dumps(st)


def serve(session: Session, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/state":
                with session.lock:
                    self.send(session.state.encode(), "application/json")
            elif self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while not session.stop:
                        j = session.jpeg
                        if j is not None and j is not last:
                            last = j
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(1 / 20)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/cmd":
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                session.cmds.put(str(body.get("cmd", "")))
                self.send(b"{}", "application/json")
            else:
                self.send_error(404)

    for p in range(port, port + 50):      # first free port (another app may already use the default)
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), H)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {port}-{port + 49}")
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="LeHak0/so101_chess_moves")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--fake", action="store_true", help="simulated arms and camera: a dry run (dataset and state files under --root, "
                                                          "default ~/.cache/huggingface/lerobot/dry_run/<repo-id>)")
    ap.add_argument("--root", default=None, help="dataset folder (default: the Hugging Face cache)")
    ap.add_argument("--teleop-hz", type=float, default=TELEOP_HZ, help="leader -> follower thread rate; 0 = inside the 30 Hz loop (the old way)")
    args = ap.parse_args()
    cache = Path.home() / ".cache/huggingface/lerobot"
    root = Path(args.root) if args.root else (cache / "dry_run" if args.fake else cache) / args.repo_id

    if args.fake:                       # a dry run keeps its board, ready pose, drill progress and episode log next to its own dataset
        import chess_recorder as CR
        from grasp_recorder import FakeRig
        global READY, DRILL_PROGRESS
        READY = CR.READY = root.parent / "ready_pose.json"; CR.STATE = root.parent / "board_state.json"
        CR.HERE = root.parent             # log_episode -> <root.parent>/episode_meta.jsonl
        DRILL_PROGRESS = root.parent / "drill_progress.json"
        rig = FakeRig(args.repo_id, root)
    else:
        rig = Rig(args.repo_id, root, args.camera, args.width, args.height)
    s = Session(rig, push=not (args.no_push or args.fake)); s.teleop_hz = args.teleop_hz
    s.publish()
    httpd = serve(s, args.port)
    url = f"http://localhost:{httpd.server_address[1]}"
    print(f"\n  Recorder UI: {url}   (keep this Terminal window open)\n", flush=True)
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        s.loop()
    except KeyboardInterrupt:
        print("\nCtrl-C: saving what was recorded (no upload)...")
        s.stop = True                     # the teleop thread ends its loop; no bus traffic while the rig closes
        try:
            with s.lk_lead, s.lk_fol:
                rig.close(push=False)
        except Exception as e:
            print(e)
    time.sleep(3)          # let the page show the final message
    httpd.shutdown()
    print("done")


if __name__ == "__main__":
    main()
