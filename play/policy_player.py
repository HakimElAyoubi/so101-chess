"""Run the fine-tuned SmolVLA on the real follower arm and evaluate it (web UI, like the recorder).

    scripts/play.sh   (from a terminal with camera permission; the page opens in your browser)

Follower only (the leader is not used). For each trial: the arm glides to the ready pose, the policy gets the task text
("move the black knight from g8 to f6") and drives the arm at 30 Hz until it comes back to the ready pose, times out or
you stop it; you then mark Success / Partial / Fail. Results: play/results/trials.jsonl (+ a wrist video per trial).
Safety: follower P=16 as in the demos, every command limited to MAX_STEP_DEG per tick, Space / Stop holds the arm
where it is immediately.

--variant skills: the staged player (plan step 2, play/staged.py). IK takes the gripper above the piece, the skill model
does the grasp, IK carries it above the target square, the skill model does the place, IK returns to the ready pose;
misses are retried. --fake runs it on the twin-simulated follower (play/fake_arm.py, no hardware) and --standin scripted
replaces the skill model by a scripted grasp / place, for a dry run of the whole flow before the model exists.
"""
import argparse
import json
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root
import so101_config  # noqa: E402  (ports and Stockfish path from config.yaml)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "recording"))
from policy_io import CHECKPOINTS, JOINTS  # noqa: E402

PAGE = HERE / "player_ui.html"
EVAL = json.loads((HERE / "eval_20.json").read_text())
RESULTS = HERE / "results"
READY = json.loads((ROOT / "recording/ready_pose.json").read_text())
READY_TOL = {"shoulder_pan": 4.0, "shoulder_lift": 4.0, "elbow_flex": 4.0, "wrist_flex": 5.0, "wrist_roll": 8.0, "gripper": 10.0}
FPS = 30
MAX_STEP_DEG = 4.0          # per tick (120 deg/s): the demos stay well below this
PATH_STEP_DEG = 8.0         # ...for planned paths (path clock, staged.LAG_OK): fully stretched out the shoulder lift needs
                            # 3-3.4 deg of error before it raises the arm at all, so at 4 deg it crawled at 3-6 deg/s
MAX_TRIAL_S = 45.0
LEFT_READY_DEG = 12.0       # the arm has "left" the ready pose once a joint is this far from it
BACK_HOLD_S = 1.0           # ...and the trial ends when it is back within tolerance for this long
HOME_SPEED = 30.0
REFRESH_AT = 25             # async mode: start computing the next chunk when this many actions are left (~0.8 s)
MAX_STAGED_S = 300.0        # skills: a whole staged move (approach, grasp + retries, carry, place, return); the stage timeouts are the real guards
READY6 = np.array([READY[j] for j in JOINTS], float)
DROP_SPOTS = [(0.0, 0.0), (0.0, 0.03), (0.03, 0.0), (0.03, 0.03), (-0.03, 0.0), (0.0, -0.03)]   # captures, around the drop-zone centre (m)
# board calibration tour over the whole board, in an order that never passes low over a tall piece. The rank 7-8 squares
# hold pieces in the eval setup: lifted off for the tour (the player asks first). Twin, eval setup without those four:
# >= 8 mm from every piece on every leg, no contact (d7 / e7 would pass 4 mm from the queen)
POINT_SQUARES = ["a8", "b7", "g7", "h8", "h6", "d6", "a6", "a5", "e4", "h4", "h2", "h1", "e3", "b2", "a1",
                 "b8", "c8", "d8", "e8", "f8", "g8"]   # rank 8, where most grasps start (2026-10-30: b8 hovered ~15 mm off)
POINT_H = 0.012             # ...gripper site (between the closed fingertips) this far above the square's centre (m):
                            # the fingertips ~6 mm above the board; next to the black pieces higher (they are wider at
                            # the foot): the lowest with >= 8 mm to every piece in the twin
# rank 8 at 3.5 cm: with only that square's piece lifted, 2.5 cm passes 5 mm from the rank-7 pawn (twin: >= 8 mm at 3.5)
POINT_H_NEAR = {**{f"{f}8": 0.035 for f in "abcdefgh"}, "b7": 0.025, "g7": 0.025, "a6": 0.021, "d6": 0.018, "h6": 0.018}
POINT_TRAVEL = 0.11         # ...between two squares: up to this height above the board (above the white king; lower only
                            # where the arm can't reach that high, near a1 / e3), across, down


def log_error():
    """The player runs in a Terminal window: keep its tracebacks in a file too (data/logs/player_errors.log)."""
    try:
        (ROOT / "data/logs").mkdir(exist_ok=True)
        with (ROOT / "data/logs/player_errors.log").open("a") as f:
            f.write(f"\n--- {datetime.now().isoformat(timespec='seconds')}\n{traceback.format_exc()}")
    except Exception:
        pass


class ChunkRunner:
    """Asynchronous chunk execution: the arm keeps executing the current chunk while the next one is computed in a
    background thread from a fresh observation; when it arrives, the actions already overtaken by time are dropped.
    No pause at chunk boundaries, and the policy re-plans every ~(50 - REFRESH_AT) ticks instead of every 50."""

    def __init__(self, policy, task, blend=True, same_noise=True):
        self.policy, self.task = policy, task
        self.blend = blend
        self.noise = policy.fixed_noise(int(time.time())) if same_noise else None    # same noise for the whole trial
        self.queue, self.tick = [], 0
        self.busy, self.pending, self.lock = False, None, threading.Lock()
        self.latencies = []

    def _work(self, q, img, tick0):
        t = time.perf_counter()
        c = self.policy.chunk(q, img, self.task, noise=self.noise)
        with self.lock:
            self.pending = (c, tick0)
        self.latencies.append(time.perf_counter() - t)
        self.busy = False

    def step(self, q, img):
        """Called once per control tick with the current observation; returns the action to send (or None = hold)."""
        if not self.queue and not self.busy and self.pending is None:      # very first chunk: compute in place
            self._work(q, img, self.tick)
        with self.lock:
            if self.pending is not None:
                c, tick0 = self.pending; self.pending = None
                late = self.tick - tick0                                    # ticks elapsed since that observation
                if late < len(c):
                    new = list(c[late:])
                    if self.blend and self.queue:
                        # cross-fade from the plan being executed to the new one over their overlap (no abrupt switch)
                        k = min(len(self.queue), len(new))
                        for i in range(k):
                            w = (i + 1) / (k + 1)
                            new[i] = (1 - w) * np.asarray(self.queue[i]) + w * np.asarray(new[i])
                    self.queue = new
        # start the next chunk early enough that it arrives before the queue runs dry (adapts to measured latency)
        lat_ticks = int(np.ceil(np.median(self.latencies[-5:]) * FPS)) + 5 if self.latencies else REFRESH_AT
        if len(self.queue) <= max(REFRESH_AT, lat_ticks) and not self.busy:
            self.busy = True
            threading.Thread(target=self._work, args=(q.copy(), img.copy(), self.tick), daemon=True).start()
        self.tick += 1
        return self.queue.pop(0) if self.queue else None


class SkillDriver:
    """The skill model for the staged player: start(text, which, to_xyz) at each skill stage, step(q6, img) every tick."""

    def __init__(self, player, move_task):
        self.p, self.move = player, move_task
        self.runner, self.text = None, ""

    def start(self, text, which, to_xyz):
        grasp = which == ("from",) and self.p.grasp_policy is not None       # a grasp-only model, if loaded, grasps
        pol = self.pol = self.p.grasp_policy if grasp else self.p.policy
        self.p.active_model = (self.p.grasp_repo if grasp else self.p.model_repo) or ""
        pol.goal = {"move": self.move, "which": which, "to_xyz": to_xyz}     # the markers this stage shows
        pol.staged, pol.player = self.p.staged, self.p                      # the move in progress (play/llm_skill.py uses them)
        pol.reset()
        self.text = text
        self.runner = ChunkRunner(pol, text) if self.p.async_exec else None
        self.p.runner = self.runner

    @property
    def timeout_s(self):
        """A skill may need longer than the player's default stage timeout (Claude: several looks of a few seconds)."""
        return getattr(getattr(self, "pol", None), "timeout_s", None)

    @property
    def gave_up(self):
        """The reason a skill stopped trying this attempt (play/llm_skill.py), for an immediate retry; None otherwise."""
        return getattr(getattr(self, "pol", None), "gave_up", None)

    @property
    def measured_from_xy(self):
        """World xy of the piece from the skill's latest visible look (play/llm_skill.py): the retry re-plans its hover there."""
        return getattr(getattr(self, "pol", None), "measured_from_xy", None)

    @property
    def fatal_error(self):
        return getattr(getattr(self, "pol", None), "fatal_error", None)

    def step(self, q6, img):
        q = np.asarray(q6, np.float32)
        return self.runner.step(q, img) if self.runner else self.pol.act(q, img, self.text)


class Player:
    def __init__(self, robot, async_exec=True, variant="v1", model_repo=None, standin=None, grasp_repo=None, grasp_ckpt="final", scripted_game=False, live_game=False, follower_p=16):
        self.robot, self.cmds = robot, queue.Queue()
        self.async_exec, self.variant = async_exec, variant
        self.model_repo, self.standin = model_repo, standin
        self.grasp_repo, self.grasp_ckpt = grasp_repo, grasp_ckpt    # skills: a grasp-only model for the grasp stage
        self.grasp_policy, self.active_model = None, ""
        self.staged, self._planner, self.drop_k = None, None, 0      # skills variant: the move in progress
        self.show_grid, self._proj = False, None       # display-only overlay of the calibrated board, to align it
        self.calib = None                              # click-the-corners board calibration in progress
        self.lock = threading.Lock()
        self.policy, self.ckpt, self.loading = None, None, ""
        self.phase = "idle"            # idle | homing | running | judging | pointing
        self.point, self.pointed = None, None    # board check: the move in progress / the square the jaws went down to
        self._point_bias = np.zeros(6)           # ...the sag correction holding the jaws there
        self.taught = {}                         # ...squares centred by hand: {square: world xy of its real centre}
        self._lifted = set()                     # ...occupied squares whose piece the user was asked to lift off
        self.taught_z = {}                       # ...touch-offs: {square: world xyz where the fingertips touched the board}
        self._dz = 0.0                           # ...how far (m) the jaws were lowered (-) / raised from the pointing height
        self._point_h = None                     # ...a pointing height chosen by the automatic calibration (m)
        self._point_tilt = 0.0                   # ...and jaw tilt (deg; 0 = vertical)
        self._point_hold = False
        self._point_psi = None                   # ...jaw roll (deg from radial; None = the staged player's PSI_DEG)
        self._point_grip = None                  # ...jaw opening during the pointing (%; None = closed, READY6[5])
        self._grip = None                        # a gripper ramp in progress: target %
        self._last_cmd = None                    # the last joint target sent (held while the gripper ramps)
        self.trial = 0                 # index into EVAL["trials"] (or -1 = custom task)
        self.custom = ""
        self.msg, self.msg_kind, self.error = "Load a checkpoint to start.", "info", ""
        self.joints, self.jpeg, self.stop = {}, None, False
        self.home = None
        self.run = None                # dict for the running trial
        self.last = None               # last finished trial (waiting for a verdict)
        RESULTS.mkdir(exist_ok=True)
        self.results = [json.loads(l) for l in open(RESULTS / "trials.jsonl")] if (RESULTS / "trials.jsonl").exists() else []
        sessions = json.loads((ROOT / "goal/board_poses.json").read_text())["sessions"]
        last = sessions[max((k for k in sessions if k.isdigit()), key=int)]
        self.taught, self.taught_z = dict(last.get("taught") or {}), dict(last.get("taught_z") or {})        # resume a tour
        self.leader, self.rec, self.replay_path, self.rec_blend = None, None, None, None   # play/move_recorder.py
        self.leader_q, self._leader_err = None, ""                 # the leader's live joints once connected (read every 3rd tick)
        self.game = None
        if live_game:
            import staged
            staged.KEEP_GOING = False
            from live_game import LiveGame
            self.game = LiveGame(self, log_root=RESULTS / "live_games")
            self.game.p_gain = follower_p
        elif scripted_game:
            from guided_game import GuidedGame
            self.game = GuidedGame(self)

    # ------------------------------------------------------------------ helpers
    def task(self):
        return self.custom if self.trial < 0 else EVAL["trials"][self.trial]["task"]

    def say(self, m, k="info"):
        self.msg, self.msg_kind = m, k

    def q(self):
        return np.array([self.joints[j] for j in JOINTS])

    def at_ready(self):
        return all(abs(self.joints[j] - READY[j]) <= READY_TOL[j] for j in JOINTS) if self.joints else False

    def send(self, target, max_step=MAX_STEP_DEG):
        """Send a joint target, limited to max_step away from the measured position on every joint."""
        q = self.q()
        t = np.clip(np.asarray(target, float), q - max_step, q + max_step)
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, t)})
        self._last_cmd = t.copy()
        return t

    def hold_here(self):
        self.robot.send_action({f"{j}.pos": float(self.joints[j]) for j in JOINTS})

    # ------------------------------------------------------------------ checkpoint loading (background thread)
    def load(self, ckpt):
        def work():
            try:
                from policy_io import Policy, load_policy
                if self.standin:
                    self.policy, self.ckpt = "scripted stand-in", "standin"
                    self.say("Scripted stand-in skill (no model): dry run of the staged player.", "good"); self.loading = ""
                    return
                self.say(f"Loading checkpoint {ckpt} (downloads it the first time)...")
                p = load_policy(ckpt, self.variant, repo=self.model_repo)
                if getattr(self.game, "require_smolvla", False) and getattr(p, "type", None) != "smolvla":
                    raise ValueError("Live games require a trained SmolVLA placement model.")
                st = np.array([READY[j] for j in JOINTS], np.float32)          # warm-up: first MPS call is slow
                text = self.task()
                if self.variant == "skills":
                    p.goal, text = {"move": self.task(), "which": ("from",), "to_xyz": None}, "pick up the black pawn"
                p.act(st, np.zeros((720, 1280, 3), np.uint8), text); p.reset()
                if self.grasp_repo and self.grasp_policy is None:
                    self.say(f"Loading the grasp model {self.grasp_repo} @ {self.grasp_ckpt}...")
                    g = load_policy(self.grasp_ckpt, "skills", repo=self.grasp_repo)
                    if getattr(self.game, "require_smolvla", False) and getattr(g, "type", None) != "smolvla":
                        raise ValueError("Live games require a trained SmolVLA grasp model.")
                    g.goal = {"move": self.task(), "which": ("from",), "to_xyz": None}
                    g.act(st, np.zeros((720, 1280, 3), np.uint8), "pick up the black pawn"); g.reset()
                    self.grasp_policy = g
                self.policy, self.ckpt = p, ckpt
                if getattr(self.game, "require_smolvla", False):
                    self.game.message = "Selected SmolVLA models are ready. Check calibration and the starting position, then start."
                    self.say(f"SmolVLA checkpoint {ckpt} ready for live chess.", "good")
                else:
                    self.say(f"Checkpoint {ckpt} ready. Put the board in the setup position, then run a trial.", "good")
            except Exception as e:
                traceback.print_exc(); self.say(f"Could not load {ckpt}: {e}", "warn")
                if getattr(self.game, "require_smolvla", False):
                    self.game.message = f"Model loading failed: {e}. Choose an accessible trained checkpoint and load again."
            self.loading = ""
        self.loading = ckpt
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ commands (control thread)
    def handle(self, c):
        if self.game is not None and self.game.handle(c):
            return
        ph = self.phase
        if c == "grid":
            self.show_grid = not self.show_grid
        elif c == "calib_start" and ph in ("idle", "judging"):
            self.projector()
            self.calib = {"order": ["h8", "h1", "a1", "a8"], "i": 0, "clicks": {}, "px": []}
            self.show_grid = False
            self.say("Board calibration: click the corners in the camera image as asked (Skip if a corner is hidden).")
        elif c.startswith("calib_click:") and self.calib:
            fx, fy = (float(v) for v in c.split(":", 1)[1].split(","))
            u, v = fx * 640, fy * 360
            corner = self.calib["order"][self.calib["i"]]
            xy = self._proj.backproject(np.array([self.joints[j] for j in JOINTS]), [[u, v]], 640)[0]
            self.calib["clicks"][corner] = xy; self.calib["px"].append((u, v)); self.calib_next()
        elif c == "calib_skip" and self.calib:
            self.calib_next()
        elif c.startswith("nudge:") and (ph in ("idle", "judging") or (ph == "pointing" and self.point["sq"])):
            self.nudge(c.split(":", 1)[1])
            if self.pointed and c not in ("nudge:save", "nudge:reset") and ph != "judging":
                self.start_point(self.pointed)       # board check: the jaws follow the nudged square centre
        elif c == "measure_board" and ph in ("idle", "judging") and not getattr(self, "_measuring", False):
            if not self.at_ready():
                self.say("Go to the ready pose first (H): the automatic measurement compares the view from there.", "warn"); return
            self._measuring = True; self.say("Measuring the board from the camera (about a minute)...")
            threading.Thread(target=self.measure_board, args=(self._last_img.copy(), dict(self.joints)), daemon=True).start()
        elif c == "calib_cancel" and self.calib:
            self.calib = None; self.say("Board calibration cancelled.")
        elif c.startswith("load:") and ph in ("idle", "judging") and not self.loading:
            self.load(c[5:])
        elif c.startswith("select:") and ph in ("idle", "judging"):
            self.trial, self.last = int(c[7:]), None
            if ph == "judging":
                self.phase = "idle"
        elif c.startswith("custom:") and ph in ("idle", "judging"):
            self.custom, self.trial, self.last, self.phase = c[7:].strip(), -1, None, "idle"
        elif c.startswith("point:") and ph == "idle":
            sq, *rest = c[6:].split(":")              # point:<square>[:<site height, mm>[:<jaw tilt, deg>[:flags]]]
            h = float(rest[0]) / 1000 if rest else None   # flags (comma-separated): hold = integrate the sag on this move even
            self._point_tilt = float(rest[1]) if len(rest) > 1 else 0.0   # if it is short (stiction); psi=<deg> = jaw roll
            flags = rest[2].split(",") if len(rest) > 2 else []
            self._point_hold = "hold" in flags
            self._point_psi = next((float(f[4:]) for f in flags if f.startswith("psi=")), None)
            self._point_grip = next((float(f[5:]) for f in flags if f.startswith("grip=")), None)   # jaw opening to keep (%)
            piece = {**dict(EVAL["setup_black"]), **dict(EVAL["setup_white"])}.get(sq)
            if piece and sq not in self._lifted and h is None:   # occupied in the eval setup: ask first, go on the second press
                self._lifted.add(sq)
                self.say(f"{sq} holds a {piece} in the eval setup: lift it off (put it back after the tour), then press "
                         "again.", "warn"); return
            self.start_point(sq, h)
        elif c.split(":")[0] == "centred" and ph == "idle" and self.pointed:
            # centred:<x>,<y>,<z>: the square's real centre and board top (FK world, m) measured by the camera
            self.centred(*(np.array(c.split(":", 1)[1].split(","), float) if ":" in c else ()))
        elif c.startswith("note:"):
            self.say(c[5:])
        elif c.startswith("grip:") and ph == "idle":          # grip:<%>: ramp the jaws, the arm holding where it is
            self._grip = float(c[5:])
        elif c == "calib_clear" and ph in ("idle", "judging"):
            self.calib_clear()
        elif c.split(":")[0] in ("lower", "raise") and self.pointed and (ph == "idle" or (ph == "pointing" and self.point["sq"])):
            # touch-off: 1 mm per press (lower:5 = 5 mm), no deeper than 8 mm below the model's board (the jaws only press
            # that hard). Overshooting is harmless: the touch height comes from the measured joints, which stop at the board
            floor = -(POINT_H_NEAR.get(self.pointed, POINT_H) - 0.006) - 0.008
            step = 0.001 * float(c.split(":")[1]) if ":" in c else 0.001
            self._dz = float(np.clip(self._dz + (step if c.startswith("raise") else -step), floor, 0.02))
            self.start_point(self.pointed)
            self.say(f"Jaws {abs(1000 * self._dz):.0f} mm {'below' if self._dz < 0 else 'above'} the pointing height. Lower (S) until "
                     "the fingertips just touch the board, then C.")
        elif c.startswith("rec_"):
            self.handle_rec(c, ph)
        elif c == "go_ready" and ph in ("idle", "judging"):
            if self.pointed or self.low():             # near the board: straight up first, then to the ready pose
                self.start_point(None); return
            q0 = self.q(); q1 = np.array([READY[j] for j in JOINTS])
            T = max(1.5, 1.875 * np.abs(q1 - q0)[:5].max() / HOME_SPEED)
            lead = self.leader_home_start(q1)                 # the leader comes along when it is connected
            if lead is not None:
                T = max(T, 1.5, 1.875 * float(np.abs(lead["q1"] - lead["q0"])[:5].max()) / HOME_SPEED)
            self.home, self.phase = {"t0": time.monotonic(), "T": T, "q0": q0, "q1": q1, "prev": ph, "lead": lead}, "homing"
            if lead is not None:
                self.say(f"Moving both arms to the ready pose ({T:.0f} s). Hands off the leader.", "info")
        elif c == "run" and ph == "idle":
            if self.policy is None:
                self.say("Load a checkpoint first.", "warn"); return
            if not self.at_ready():
                self.say("Not at the ready pose: press 'Arm to ready pose' (H) first.", "warn"); return
            self.start_trial()
        elif c == "stop" and ph in ("running", "homing", "pointing"):
            self.hold_here()
            if ph == "running":
                self.end_trial("stopped by you")
            elif ph == "pointing":
                self.phase, self.point = "idle", None; self.say("Stopped. H goes straight up, then to the ready pose.")
            else:
                self.phase = self.home["prev"]; self.home = None; self.say("Auto-move stopped.")
        elif c.startswith("verdict:") and ph == "judging" and self.last:
            v, note = (c[8:].split("|", 1) + [""])[:2]
            rec = {**self.last, "outcome": v, "note": note.strip()}
            with (RESULTS / "trials.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            self.results.append(rec)
            self.last = None
            if self.trial >= 0 and self.trial + 1 < len(EVAL["trials"]):
                self.trial += 1
            self.phase = "idle"
            self.say(f"Saved ({v}). Put the board back in the setup position; the arm is going to the ready pose.", "good")
            self.handle("go_ready")

    def handle_rec(self, c, ph):
        """Demonstration recording with the leader arm and its replay (play/move_recorder.py).
        rec_start:<ply>:<uci> (idle) -> phase 'teleop'; rec_stop saves; rec_discard; rec_play:<ply>:<uci> replays as a trial;
        rec_delete:<ply>:<uci>."""
        import move_recorder as MR
        script = self.game.script if self.game is not None else "adhoc"
        parts = c.split(":")
        if parts[0] == "rec_connect":
            self.connect_leader()
        elif parts[0] == "rec_home" and ph in ("idle", "judging"):
            self.connect_leader(); self.handle("go_ready")
        elif parts[0] == "rec_start" and ph in ("idle", "judging") and len(parts) == 3:
            if not self.connect_leader():
                return
            # Like the old recorder: both arms start from the ready pose, so the recording begins with no jump and a replay
            # from ready reproduces it exactly.
            if not self.at_ready():
                self.say("Bring the follower to ready (H) first.", "warn"); return
            off = self.leader_offsets()
            if off is None or not self.leader_ready():
                bad = ", ".join(f"{j} {v:+.0f}" for j, v in (off or {}).items() if not self.leader_joint_ok(j, v)) or "no reading yet"
                self.say(f"Move the leader nearer the ready pose first ({bad}; within {self.LEADER_NEAR_DEG:.0f} deg).", "warn"); return
            now = time.monotonic()
            self.rec = MR.MoveRecording(MR.recording_path(script, int(parts[1]), parts[2]), int(parts[1]), parts[2], now)
            self.rec_blend = dict(t0=now, q0=self.q(), T=0.8)        # ENGAGE_T: the follower blends from ready onto the live leader
            self.phase = "teleop"
            self.say(f"Recording {parts[2]}: the follower copies the leader. Make the move, then Stop & save.", "good")
        elif parts[0] == "rec_stop" and ph == "teleop":
            rec, self.rec = self.rec, None
            self.hold_here(); self.phase = "idle"
            if rec is None or len(rec.rows) < 10:
                self.say("Nothing recorded (too short).", "warn"); return
            path = rec.save()
            self.say(f"Saved {path.name}: {rec.seconds:.1f} s.", "good")
            if self.game is not None:
                self.game.recorded(rec.ply, path, rec.seconds)
        elif parts[0] == "rec_discard" and ph == "teleop":
            self.rec = None; self.hold_here(); self.phase = "idle"; self.say("Recording discarded.", "warn")
        elif parts[0] == "rec_play" and ph in ("idle", "judging") and len(parts) == 3:
            path = MR.recording_path(script, int(parts[1]), parts[2])
            if not path.exists():
                self.say(f"No recording for {parts[2]} yet.", "warn"); return
            if not self.at_ready():
                self.say("Bring the arm to ready (H) before a play-back.", "warn"); return
            self.replay_path, self.trial, self.custom, self.last, self.phase = path, -1, f"replay {parts[2]}", None, "idle"
            self.start_trial()
        elif parts[0] == "rec_delete" and len(parts) == 3:
            path = MR.recording_path(script, int(parts[1]), parts[2])
            if path.exists():
                path.unlink(); self.say(f"Deleted {path.name}.", "warn")

    def connect_leader(self):
        import move_recorder as MR
        if self.leader is not None:
            return True
        try:
            self.leader = MR.LeaderTeleop().connect(); self._leader_err = ""
            self.say("Leader arm connected. Hold it in the ready pose to record.", "good"); return True
        except Exception as e:
            self.leader = None; self._leader_err = f"{type(e).__name__}: {str(e)[:90]}"
            self.say(f"Leader arm not available ({self._leader_err}). Plug it in on the 5 V supply and try again.", "warn"); return False

    LEADER_WRIST_BACKOFF = 8.0     # deg: the leader's handle hits its own arm at the ready pose's wrist flex (web_recorder)

    def leader_home_start(self, q1):
        """Leader motors on, holding where they are (no jump), ready to glide to the ready pose with the follower.
        Returns {q0, q1} in JOINTS order or None when no leader is connected."""
        if self.leader is None:
            return None
        try:
            bus = self.leader.teleop.bus
            here = bus.sync_read("Present_Position")
            bus.sync_write("Goal_Position", here)
            bus.enable_torque()
            q0 = np.array([float(here[j]) for j in JOINTS])
            q1 = np.asarray(q1, float).copy(); q1[3] -= self.LEADER_WRIST_BACKOFF
            return {"q0": q0, "q1": q1}
        except Exception as e:
            self._leader_err = f"{type(e).__name__}: {str(e)[:60]}"; return None

    def leader_home_step(self, lead, k):
        try:
            t = lead["q0"] + (lead["q1"] - lead["q0"]) * k
            self.leader.teleop.send_feedback({f"{j}.pos": float(v) for j, v in zip(JOINTS, t)})
            self.leader_q = t
        except Exception as e:
            self._leader_err = f"{type(e).__name__}: {str(e)[:60]}"

    def leader_release(self):
        """Leader motors off again: it is the user's to move."""
        try:
            self.leader.teleop.bus.disable_torque()
        except Exception as e:
            self._leader_err = f"{type(e).__name__}: {str(e)[:60]}"

    def leader_offsets(self):
        """{joint: leader - ready} in degrees (gripper %), or None without a reading."""
        if self.leader_q is None:
            return None
        return {j: float(self.leader_q[i] - READY[j]) for i, j in enumerate(JOINTS)}

    LEADER_NEAR_DEG = 25.0         # the old recorder's ENGAGE_MAX: the leader may be this far from ready at Start, the blend absorbs it

    def leader_joint_ok(self, j, off):
        """Near the ready pose (within LEADER_NEAR_DEG, as the old recorder allowed at Start: the follower blends from ready
        onto the live leader); the wrist flex may also sit the LEADER_WRIST_BACKOFF lower (its handle hits its own arm at
        the full ready flex, so the auto-move parks it there)."""
        tol = max(READY_TOL[j], self.LEADER_NEAR_DEG)
        lo = -tol - (self.LEADER_WRIST_BACKOFF if j == "wrist_flex" else 0.0)
        return lo <= off <= tol

    def leader_ok(self):
        off = self.leader_offsets()
        return None if off is None else {j: self.leader_joint_ok(j, v) for j, v in off.items()}

    def leader_ready(self):
        ok = self.leader_ok()
        return ok is not None and all(ok.values())

    def start_trial(self):
        if self.variant == "skills":
            if not self.start_staged():
                return
        else:
            self.policy.reset()
            self.runner = ChunkRunner(self.policy, self.task()) if self.async_exec else None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vid = RESULTS / "videos" / f"{stamp}_{self.ckpt}_{self.trial + 1 if self.trial >= 0 else 'custom'}.mp4"
        vid.parent.mkdir(exist_ok=True)
        self.run = {"t0": time.monotonic(), "left": False, "back_since": None, "stamp": stamp, "video": vid,
                    "writer": cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), FPS / 2, (640, 360)),
                    "log": [], "n": 0, "slow_ticks": 0}
        self.phase = "running"
        self.say("")

    def start_staged(self):
        """Plan the move (hover IK on the twin) and arm the staged executor. False if it can't be planned."""
        if self.replay_path is not None:                      # a demonstration recorded with the leader: replay it as is
            from move_recorder import ReplayMove
            path, self.replay_path, self.runner = self.replay_path, None, None
            try:
                self.staged = ReplayMove(path, READY6, time.monotonic())
            except Exception as e:
                self.say(f"Cannot replay {Path(path).name}: {e}", "warn"); return False
            return True
        import staged as S
        from fake_arm import ScriptedSkill
        P = self.projector()
        if self._planner is None:
            self._planner = S.Planner(P)
        self.runner = None
        drv = ScriptedSkill(self._planner) if self.standin else SkillDriver(self, self.task())
        base = P.drop_xyz(self._sess)
        drop = base + np.array([*DROP_SPOTS[self.drop_k % len(DROP_SPOTS)], 0.0])
        try:
            self.staged = S.StagedMove(self._planner, drv, self.task(), self._sess, READY6, time.monotonic(), drop_xyz=drop)
        except ValueError as e:
            self.say(f"The staged player only runs moves: {e}", "warn"); return False
        if self.standin:
            drv.move = self.staged
        if self.staged.result is not None:
            self.say(f"Can't plan this move: {self.staged.result[1]}", "warn"); self.staged = None; return False
        return True

    def start_point(self, sq, h=None):
        """Board check without the camera: hold the closed jaws POINT_H above the centre of square sq, placed by the
        same IK and board position as the staged player's approach (while nudging: the unsaved preview position).
        sq None: straight up, then back to the ready pose."""
        import staged as S
        P = self.projector()
        if self._planner is None:
            self._planner = S.Planner(P)
        Pl, q0 = self._planner, self.q()
        if sq is None:
            W = list(Pl.rise_to(q0, S.APPROACH_Z)) + [READY6]
        else:
            if not self.pointed and not self.at_ready():
                self.say("Go to the ready pose first (H).", "warn"); return
            if sq != self.pointed:
                self._dz, self._point_h = 0.0, None
            if h is not None:
                self._point_h = h                      # a chosen height (the automatic calibration): kept for the follow moves
            xyz = P.square_xyz(sq, self._sess, z=(self._point_h or POINT_H_NEAR.get(sq, POINT_H)) + self._dz)
            xyz[2] += P.board_dz(self._sess, xyz)      # pointing is physical: the measured board top (touch-offs)
            q, _ = Pl.ik_near(xyz, self._point_tilt, S.PSI_DEG if self._point_psi is None else self._point_psi, q0,
                              READY6[5] if self._point_grip is None else self._point_grip)
            if q is None:
                self.say(f"No IK solution above {sq}.", "warn"); return
            bias = np.zeros(6)
            if not self.pointed:
                W = Pl.approach(q0, q)
            elif np.linalg.norm(Pl.fk(q0)[0] - Pl.fk(q)[0]) < 0.015:
                W, bias = [q0, q], self._point_bias.copy()     # a nudge: straight there at the same height, keeping
            else:                                              # the sag correction (q0 is where it holds the jaws)
                W = Pl.cart_path(q0, q, z_travel=Pl.z0 + POINT_TRAVEL)[0]
            self.pointed = sq
        # a follow move (nudge, lower, raise) keeps the sag correction it has and integrates no further: lowered onto the
        # board, the error can't close, and integrating it would press the jaws harder and harder into the board
        self.point, self.phase = {"traj": S.Trajectory(W), "t0": time.monotonic(), "sq": sq, "bias": bias if sq else np.zeros(6),
                                  "integrate": len(W) > 2 or self._point_hold}, "pointing"
        if len(W) > 2 or sq is None:
            self.say(f"Going down above {sq}..." if sq else "Up, then to the ready pose...")

    def centred(self, x=None, y=None, z=None):
        """The closed jaws are centred over the pointed square (after nudging): that is where the square really is.
        Refit the board (position, rotation, size along ranks / files) to every square centred so far and save it.
        x, y, z: the centre and the board top measured by the camera instead (calibration/auto_board_calib.py)."""
        t, P, sq = self._targets, self._proj, self.pointed
        cur = t.POSES["sessions"][self._sess]                 # the nudged preview, or the saved board
        self.taught[sq] = [float(x), float(y)] if x is not None else P.square_xyz(sq, self._sess)[:2].tolist()
        fit = P.fit_squares(self.taught, cur)
        corr = P.fit_correction(self.taught, fit)            # what the board model misses, square by square
        touch = None
        if z is not None:
            self.taught_z[sq] = [float(x), float(y), float(z)]
        elif self._dz < -0.0005:                             # lowered until the fingertips touched: the board top there
            P.s.set_joints(self.q()); t1, t2 = P.s.jaw_tip_positions()
            z = float(min(t1[2], t2[2])); xyz = P.square_xyz(sq, self._sess)
            touch = z - float(xyz[2] + P.board_dz(self._sess, xyz))
            self.taught_z[sq] = [*self.taught[sq], z]
        height = P.fit_height(self.taught_z) if self.taught_z else cur.get("height")
        key = str(max(900, max(int(k) for k in t.POSES["sessions"] if k.isdigit()) + 1))
        d = np.array([fit["dx"] - cur["dx"], fit["dy"] - cur["dy"]])
        t.POSES["sessions"][key] = {**{k: fit[k] for k in ("dx", "dy", "yaw_deg", "scale_rank", "scale_file")},
                                    "drop_xy": (np.array(cur["drop_xy"]) + d).tolist() if cur.get("drop_xy") else None,
                                    "spare_xy": cur.get("spare_xy"), "rms_mm": None, "taught": dict(self.taught),
                                    "per_square_mm": fit["per_square_mm"], "correction": corr,
                                    "taught_z": dict(self.taught_z), "height": height,
                                    "source": ("squares centred by the camera: " if x is not None else "squares centred by hand: ") + ", ".join(self.taught),
                                    "time": datetime.now().isoformat(timespec="seconds")}
        t.LATEST = key; self._sess = key; self._nudge = None
        (ROOT / "goal/board_poses.json").write_text(json.dumps({**t.POSES, "sessions": {k: v for k, v in t.POSES["sessions"].items() if k.isdigit()}}, indent=1))
        if self.policy is not None and getattr(self.policy, "proj", None) is not None:
            self.policy.session = key
        worst = max(fit["per_square_mm"].items(), key=lambda kv: kv[1])
        left = [q for q in POINT_SQUARES if q not in self.taught]
        self.say(f"{sq} saved ({len(POINT_SQUARES) - len(left)} of {len(POINT_SQUARES)}). Board size ranks x{fit['scale_rank']:.3f}, files "
                 f"x{fit['scale_file']:.3f}, rotation {fit['yaw_deg']:.1f} deg"
                 + (f"; the board model alone is off by up to {worst[1]:.1f} mm ({worst[0]})" if len(self.taught) > 1 else "")
                 + ("; the correction map takes that out. " if corr else ". ")
                 + (f"Touched {abs(1000 * touch):.1f} mm {'higher' if touch > 0 else 'lower'} than expected; the pointing height map "
                    "now includes it. " if touch is not None else "No touch-off here (not lowered). ")
                 + (f"Next: P for {left[0]}." if left else "Tour done: P again to check any square (it should land centred), H when finished."),
                 "good")

    def calib_clear(self):
        """Start the calibration tour over (the board was moved): forget the centred squares; the board pose stays."""
        self.taught, self.taught_z = {}, {}
        self.say(f"Calibration tour cleared: centre the {len(POINT_SQUARES)} squares again (P, arrows, C).")

    def low(self):
        """Away from the ready pose with the gripper below the approach height: a direct joint move home would dip
        toward the board (it skimmed it at 1-3 mm), so go straight up first."""
        import staged as S
        if self.at_ready():
            return False
        if self._planner is None:
            self._planner = S.Planner(self.projector())
        return self._planner.fk(self.q())[0][2] < self._planner.z0 + S.APPROACH_Z - 0.01

    def end_trial(self, reason):
        r, self.run, self.phase = self.run, None, "judging"   # stop driving the arm first, whatever happens next
        mv, self.staged = self.staged, None
        try:
            r["writer"].release()
            log = np.array(r["log"]) if r["log"] else np.zeros((0, 13))
            np.save(str(r["video"]).replace(".mp4", "_joints.npy"), log)
        except Exception as e:
            traceback.print_exc(); self.say(f"Could not save the trial video/log: {e}", "warn")
        t = EVAL["trials"][self.trial] if self.trial >= 0 else {"i": None, "category": "custom", "train_count": None}
        self.last = {"time": r["stamp"], "checkpoint": self.ckpt, "trial": t["i"], "category": t["category"],
                     "train_count": t.get("train_count"), "task": self.task(), "duration_s": round(time.monotonic() - r["t0"], 1),
                     "variant": self.variant, "async": self.async_exec,
                     "chunk_latency_s": round(float(np.median(self.runner.latencies)), 3) if getattr(self, "runner", None) and self.runner.latencies else None,
                     "end": reason, "video": __import__("os").path.relpath(r["video"], HERE), "slow_ticks": r["slow_ticks"]}
        if mv is not None:                               # skills: what each stage did (for the per-stage analysis)
            self.last.update(model_repo=self.model_repo, grasp_repo=self.grasp_repo, grasp_ckpt=self.grasp_ckpt if self.grasp_repo else None,
                             standin=self.standin, stages=mv.events, retries=mv.retries,
                             staged_result=list(mv.result) if mv.result else None, staged_info=mv.info,
                             close_h_cm=round(100 * mv.close_h, 2) if mv.close_h is not None else None)
            if mv.result and mv.result[0] == "placed" and mv.ring is not None:
                self.drop_k += 1                         # next capture goes to the next drop spot
        self.say(f"Trial ended ({reason}, {self.last['duration_s']} s). How did it go?")
        if self.game is not None:
            self.game.robot_finished(self.last)


    # ------------------------------------------------------------------ control loop
    def loop(self):
        period, n = 1 / FPS, 0
        while not self.stop:
            tick = time.perf_counter()
            try:
                obs = self.robot.get_observation()
                self.joints = {j: float(obs[f"{j}.pos"]) for j in JOINTS}
                img = obs.get("wrist")
                if img is not None:
                    self._last_img = img
                if self.phase == "homing":
                    h = self.home; u = min(1.0, (time.monotonic() - h["t0"]) / h["T"])
                    k = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
                    self.send(h["q0"] + (h["q1"] - h["q0"]) * k)
                    if h.get("lead") is not None:
                        self.leader_home_step(h["lead"], k)
                    if u >= 1.0 and time.monotonic() - h["t0"] > h["T"] + 0.4:
                        if h.get("lead") is not None:
                            self.leader_release()
                        self.phase, self.home = h["prev"] if h["prev"] != "judging" else "idle", None
                        self.say("At the ready pose." if self.at_ready() else "Auto-move done (not quite at the ready pose).",
                                 "good" if self.at_ready() else "warn")
                elif self.phase == "idle" and self._grip is not None:
                    base = self._last_cmd if self._last_cmd is not None else self.q()
                    g = float(np.clip(self._grip, base[5] - 4.0, base[5] + 4.0))
                    self.send(np.array([*base[:5], g]))
                    if abs(g - self._grip) < 0.01:
                        self._grip = None
                elif self.phase == "pointing":
                    import staged as S
                    pt = self.point; now = time.monotonic(); tr = pt["traj"]
                    pt["tau"] = tr.advance(pt.get("tau", 0.0), self.q(), now - pt.get("prev", pt["t0"])); pt["prev"] = now
                    err = tr.W[-1][:5] - self.q()[:5]
                    if not tr.done(pt["tau"]):
                        self.send(tr.at(pt["tau"]) + pt["bias"], PATH_STEP_DEG)
                    elif pt["sq"] and pt["integrate"] and np.abs(err).max() > S.SAG_TOL_DEG and now - pt.setdefault("held", now) < S.SAG_MAX_S:
                        # at the square: take out the gravity sag as the staged hover does, so the centring is consistent
                        pt["bias"][:5] = np.clip(pt["bias"][:5] + S.SAG_GAIN * err, -S.SAG_MAX_DEG, S.SAG_MAX_DEG)
                        self.send(tr.W[-1] + pt["bias"], PATH_STEP_DEG)
                    else:
                        self._point_bias = pt["bias"] if pt["sq"] else np.zeros(6)
                        self.phase, self.point = "idle", None
                        if pt["sq"] is None:
                            self.pointed = None
                            self.say("At the ready pose." if self.at_ready() else "Auto-move done (not quite at the ready pose).",
                                     "good" if self.at_ready() else "warn")
                            if self.leader is not None and not self.leader_ready():
                                self.cmds.put("go_ready")        # the straight-up route skipped the leader: bring it home too
                        else:
                            if pt["integrate"]:
                                self.say(f"The closed jaw tips should be right above the centre of {pt['sq']}, ~{1000 * POINT_H_NEAR.get(pt['sq'], POINT_H) - 6:.0f} mm "
                                         "up. Centre them with the arrows, lower them with S until they just touch the board, then C.", "good")
                elif self.phase == "teleop":                  # recording a demonstration: the follower copies the leader
                    now = time.monotonic(); act = self.leader.action6(); self.leader_q = act; b = self.rec_blend
                    u = (now - b["t0"]) / b.get("T", 0.6)          # ease over the ready tolerance only (both arms start at ready)
                    if u < 1.0:
                        k = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
                        act = b["q0"] + (act - b["q0"]) * k
                    sent = self.send(act, PATH_STEP_DEG)
                    if self.rec is not None:
                        self.rec.add(now, sent, self.q())
                elif self.phase == "running":
                    r = self.run; t_inf = time.perf_counter()
                    if self.staged is not None:
                        a = self.staged.step(self.q(), img, time.monotonic())
                    elif self.runner:
                        a = self.runner.step(self.q().astype(np.float32), img)
                    else:
                        a = self.policy.act(self.q().astype(np.float32), img, self.task())
                    if time.perf_counter() - t_inf > 2 * period:
                        r["slow_ticks"] += 1          # the control loop stalled on this tick (synchronous inference)
                    path = self.staged is not None and self.staged.on_path()
                    sent = self.send(a if a is not None else self.q(), PATH_STEP_DEG if path else MAX_STEP_DEG)
                    r["log"].append([time.monotonic() - r["t0"], *self.q(), *sent])
                    if r["n"] % 2 == 0 and img is not None:
                        r["writer"].write(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (640, 360)))
                    r["n"] += 1
                    if self.staged is not None:                # skills: the staged move decides when it is over
                        if self.staged.result is not None:
                            self.hold_here(); self.end_trial(" ".join(self.staged.result).strip())
                        elif time.monotonic() - r["t0"] > MAX_STAGED_S:
                            self.hold_here(); self.end_trial(f"time limit {MAX_STAGED_S:.0f} s ({self.staged.stage})")
                    else:
                        d = np.abs(self.q() - np.array([READY[j] for j in JOINTS]))[:5].max()
                        if d > LEFT_READY_DEG:
                            r["left"] = True
                        if r["left"] and self.at_ready():
                            r["back_since"] = r["back_since"] or time.monotonic()
                            if time.monotonic() - r["back_since"] >= BACK_HOLD_S:
                                self.end_trial("back at the ready pose")
                        else:
                            r["back_since"] = None
                        if self.run and time.monotonic() - r["t0"] > MAX_TRIAL_S:
                            self.hold_here(); self.end_trial(f"time limit {MAX_TRIAL_S:.0f} s")
                n += 1
                if self.leader is not None and self.phase != "teleop" and n % 3 == 0:
                    try:
                        self.leader_q = self.leader.action6()   # the leader's live pose for the readiness display
                    except Exception as e:
                        self._leader_err = f"{type(e).__name__}: {str(e)[:60]}"
                self.publish_frame(n, img)
                self.error = ""
            except Exception as e:
                self.error = f"Arm/camera/policy error: {e}"
                traceback.print_exc(); log_error()
                if self.phase == "running":
                    try:
                        self.hold_here(); self.end_trial("error")
                    except Exception:
                        pass
                time.sleep(0.2)
            while not self.cmds.empty():
                try:
                    self.handle(self.cmds.get_nowait())
                except Exception as e:
                    traceback.print_exc(); log_error(); self.say(f"Error: {e}", "warn")
            if self.game is not None:
                try:
                    self.game.tick(time.monotonic(), img)
                except Exception as exc:
                    self.game.status = "paused"
                    self.game.message = f"Game paused: {exc}"
                    traceback.print_exc()
            self.publish()
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def publish_frame(self, n, img):
        if n % 2 == 0 and img is not None:
            view = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (640, 360))
            if self.show_grid:
                self.draw_grid(view)
            if self.pointed and self.phase in ("idle", "pointing"):
                self.draw_under_jaws(view)
            if self.calib:
                for (u, v) in self.calib["px"]:
                    cv2.circle(view, (int(u), int(v)), 6, (0, 0, 255), 2, cv2.LINE_AA)
            self.jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def projector(self):
        if self._proj is None:
            sys.path.insert(0, str(ROOT / "goal"))
            import targets
            self._targets, self._proj, self._sess = targets, targets.Projector(), targets.LATEST
        return self._proj

    def calib_next(self):
        c = self.calib; c["i"] += 1
        if c["i"] < len(c["order"]):
            return
        clicks, self.calib = c["clicks"], None
        if len(clicks) < 2:
            self.say("Need at least 2 corners: calibration not changed.", "warn"); return
        fit = self._proj.fit_board(clicks)
        if fit["rms_mm"] > 20:                      # clicked points don't form the board's corners: refuse
            self.say(f"Those clicks don't fit the board ({fit['rms_mm']:.0f} mm error): calibration NOT changed. "
                     "Check which corner is which (h8 = bottom-left of the image, next to the H letter).", "warn")
            return
        t = self._targets
        key = str(max(900, max(int(k) for k in t.POSES["sessions"] if k.isdigit()) + 1))     # live calibrations are sessions 900+
        old = t.POSES["sessions"].get(t.LATEST, {})
        entry = {**{k: old.get(k) for k in ("drop_xy", "spare_xy", "scale_rank", "scale_file")}, **{k: fit[k] for k in ("dx", "dy", "yaw_deg", "rms_mm")},
                 "source": "live corner clicks", "corners": list(clicks), "time": datetime.now().isoformat(timespec="seconds")}
        if entry["drop_xy"] is not None:            # the drop zone moves with the board: shift it by the change
            d = np.array([fit["dx"] - old.get("dx", 0), fit["dy"] - old.get("dy", 0)])
            entry["drop_xy"] = (np.array(entry["drop_xy"]) + d).tolist()
        t.POSES["sessions"][key] = entry; t.LATEST = key; self._sess = key; self.taught, self.taught_z = {}, {}
        (ROOT / "goal/board_poses.json").write_text(json.dumps({**t.POSES, "sessions": {k: v for k, v in t.POSES["sessions"].items() if k.isdigit()}}, indent=1))
        if self.policy is not None and getattr(self.policy, "proj", None) is not None:
            self.policy.session = key              # coords / marker models use the new board pose from now on
        self.show_grid = True
        moved = np.hypot(fit["dx"] - old.get("dx", 0), fit["dy"] - old.get("dy", 0)) * 1000
        msg = (f"Board measured from {len(clicks)} corners: moved {moved:.0f} mm and {fit['yaw_deg'] - old.get('yaw_deg', 0):+.1f} deg "
               f"since the last session; corner fit {fit['rms_mm']:.0f} mm rms. Check the grid now matches.")
        self.say(msg, "good" if fit["rms_mm"] < 10 else "warn")

    def nudge(self, what):
        """Manual fine adjustment of the board pose used by the grid AND the coords / marker models.
        what: 'near' / 'far' (toward / away from the arm), 'a' / 'h' (toward the a- / h-file side), 'cw' / 'ccw' (0.5 deg),
        or 'save'. Each nudge moves 1 mm along the board's own axes."""
        self.projector(); t = self._targets
        if not hasattr(self, "_nudge") or self._nudge is None:
            base = t.POSES["sessions"][self._sess]
            self._nudge = {k: base.get(k) for k in ("dx", "dy", "yaw_deg", "scale_rank", "scale_file", "correction", "height", "drop_xy", "spare_xy")}
            self._nudge_from = self._sess
        n = self._nudge
        if what == "reset":                                 # drop the unsaved preview (the automatic calibration starts clean)
            self._nudge, self._sess = None, t.LATEST
            t.POSES["sessions"].pop("preview", None)
            self.say("Unsaved nudges dropped: back to the saved board position."); return
        if what == "save":
            key = str(max(900, max(int(k) for k in t.POSES["sessions"] if k.isdigit()) + 1))
            t.POSES["sessions"][key] = {**n, "rms_mm": None, "source": "manual nudge", "time": datetime.now().isoformat(timespec="seconds")}
            t.LATEST = key; self._sess = key
            (ROOT / "goal/board_poses.json").write_text(json.dumps({**t.POSES, "sessions": {k: v for k, v in t.POSES["sessions"].items() if k.isdigit()}}, indent=1))
            if self.policy is not None and getattr(self.policy, "proj", None) is not None:
                self.policy.session = key
            self._nudge = None
            self.say("Board position saved; the coords / marker models use it from now on.", "good"); return
        a = np.radians(n["yaw_deg"])
        ax_rank = np.array([np.cos(a), np.sin(a)])        # board +x: from rank 8 (arm side) toward rank 1
        ax_file = np.array([-np.sin(a), np.cos(a)])       # board +y: from the a-file toward the h-file
        step = {"far": ax_rank, "near": -ax_rank, "h": ax_file, "a": -ax_file}.get(what)
        if what.startswith("mm:"):                          # mm:<far>,<h>: a vector nudge in mm (the automatic calibration)
            far, hh = (float(v) for v in what[3:].split(","))
            step = far * ax_rank + hh * ax_file
        if step is not None:
            n["dx"] += 0.001 * step[0]; n["dy"] += 0.001 * step[1]
            if n.get("drop_xy"):
                n["drop_xy"] = (np.array(n["drop_xy"]) + 0.001 * step).tolist()
        elif what in ("cw", "ccw"):
            n["yaw_deg"] += 0.5 if what == "ccw" else -0.5
        # preview: a temporary session the grid draws from (not used by the models until saved)
        t.POSES["sessions"]["preview"] = {**n}; self._sess = "preview"; self.show_grid = True
        base = t.POSES["sessions"][self._nudge_from]
        self.say(f"Adjusting: {np.hypot(n['dx'] - base['dx'], n['dy'] - base['dy']) * 1000:.0f} mm, "
                 f"{n['yaw_deg'] - base['yaw_deg']:+.1f} deg from the saved position. Press Save when the grid fits.")

    def measure_board(self, img_rgb, joints):
        """Automatic board measurement: fit the projected grid to the live ready-pose image AND to a reference image from
        recording session 7 taken at the same pose; the difference is how far the board moved since then. Both fits use
        session 7's board size in the arm's frame (ranks x1.02, files x0.91, fitted to its demos); whatever error the
        camera model still has at the ready pose is the same in both fits, so it cancels. Good to ~1 cm: the "Point at"
        check and "Centred" finish the job."""
        try:
            sys.path.insert(0, str(ROOT / "goal"))
            import fit_board_image as fb
            self.projector()
            ref = json.loads((ROOT / "goal/reference_fit.json").read_text())["sized"]
            s7 = ref["session7"]; sizes = (ref["scale_rank"], ref["scale_file"])
            view = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (640, 360))
            live, _ = fb.fit(view, joints, verbose=False, base=[s7["dx"], s7["dy"], s7["yaw_deg"]], sizes=sizes)
            pose = {k: s7[k] + live[k] - ref[k] for k in ("dx", "dy", "yaw_deg")}
            t = self._targets
            cur = t.POSES["sessions"][self._sess]
            key = str(max(900, max(int(k) for k in t.POSES["sessions"] if k.isdigit()) + 1))
            d = np.array([pose["dx"] - s7["dx"], pose["dy"] - s7["dy"]])
            d7 = t.POSES["sessions"]["7"].get("drop_xy")
            entry = {**pose, "scale_rank": cur.get("scale_rank") or sizes[0], "scale_file": cur.get("scale_file") or sizes[1],
                     "rms_mm": None, "drop_xy": (np.array(d7) + d).tolist() if d7 else None,
                     "spare_xy": t.POSES["sessions"]["7"].get("spare_xy"),
                     "source": "automatic image measurement relative to session 7 (with the board size)",
                     "time": datetime.now().isoformat(timespec="seconds")}
            t.POSES["sessions"][key] = entry; t.LATEST = key; self._sess = key; self.taught, self.taught_z = {}, {}
            (ROOT / "goal/board_poses.json").write_text(json.dumps({**t.POSES, "sessions": {k: v for k, v in t.POSES["sessions"].items() if k.isdigit()}}, indent=1))
            if self.policy is not None and getattr(self.policy, "proj", None) is not None:
                self.policy.session = key
            self.show_grid = True
            self.say(f"Board measured: moved {np.hypot(*d) * 1000:.0f} mm and {pose['yaw_deg'] - s7['yaw_deg']:+.1f} deg since the "
                     "last recording session (good to ~1 cm). Now point at each check square (P), centre it with the "
                     "arrows and press C.", "good")
        except Exception as e:
            traceback.print_exc(); self.say(f"Board measurement failed: {e}", "warn")
        self._measuring = False

    def draw_grid(self, view):
        """Board lines as the board pose has them (size and correction map included, point by point: straight lines
        between the corners skewed the grid by the corners' own corrections), projected for the current joint state."""
        P, q6 = self.projector(), np.array([self.joints[j] for j in JOINTS])
        def pt(f, r):                                   # at the board's measured height (the touch-offs / camera calibration):
            xyz = P.board_point(f, r, self._sess)       # drawn at the model's height, the slanted view shifts the grid sideways
            xyz[2] += P.board_dz(self._sess, xyz)
            return xyz
        for i in range(9):
            for line in ([(i - 0.5, r) for r in np.linspace(0.5, 8.5, 33)], [(f, i + 0.5) for f in np.linspace(-0.5, 7.5, 33)]):
                uv, vis = P.project(q6, np.array([pt(f, r) for f, r in line]), view.shape[1])
                uv = uv[vis]
                if len(uv) > 1:
                    cv2.polylines(view, [uv.astype(np.int32)], False, (0, 220, 0) if i in (0, 8) else (0, 160, 0), 1, cv2.LINE_AA)

    def draw_under_jaws(self, view):
        """Calibration: a cyan crosshair on the board straight under the jaw tips. It needs only the camera's mounting on
        the gripper (the projected tips land on the real ones), not the arm's joints or the board pose: centre it on the
        real square's centre."""
        P, q6 = self.projector(), np.array([self.joints[j] for j in JOINTS])
        P.s.set_joints(q6); t1, t2 = P.s.jaw_tip_positions()
        mid = (np.asarray(t1) + np.asarray(t2)) / 2
        uv, vis = P.project(q6, np.array([[mid[0], mid[1], P.s.cfg.board_thickness]]), view.shape[1])
        if vis[0]:
            u, v = (int(x) for x in uv[0])
            cv2.drawMarker(view, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            cv2.circle(view, (u, v), 8, (255, 255, 0), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ state for the page
    def publish(self):
        trials = EVAL["trials"]
        by_ckpt = {}
        for r in self.results:
            mode = " sync" if (r.get("async") is False and r.get("variant", "v1") != "v1") else ""
            name = (r.get("model_repo") or "").split("/")[-1] or r.get("variant", "v1")     # skills: one row per model
            if r.get("grasp_repo"):
                name = f"grasp {r['grasp_repo'].rstrip('/').split('/')[-1]} {r.get('grasp_ckpt', '')} + place {name}"
            by_ckpt.setdefault(f"{name} {r['checkpoint']}{mode}", []).append(r)
        summary = []
        for ck, rs in by_ckpt.items():
            row = {"checkpoint": ck, "n": len(rs)}
            for v in ("success", "partial", "fail"):
                row[v] = sum(1 for r in rs if r["outcome"] == v)
            for cat in ("common", "drill_squares", "new_combination", "capture"):
                cs = [r for r in rs if r["category"] == cat]
                row[cat] = f"{sum(1 for r in cs if r['outcome'] == 'success')}/{len(cs)}" if cs else ""
            summary.append(row)
        latest = {}
        for r in self.results:
            if (r["checkpoint"] == self.ckpt and r.get("variant", "v1") == self.variant and r["trial"]
                    and r.get("model_repo") == self.model_repo and r.get("grasp_repo") == self.grasp_repo
                    and (r.get("async", False) == self.async_exec or self.variant == "v1")):
                latest[r["trial"]] = r["outcome"]
        st = {
            "phase": self.phase, "msg": self.msg, "msg_kind": self.msg_kind, "error": self.error,
            "checkpoint": f"{self.variant} {self.ckpt}{'' if self.async_exec else ' (sync)'}" if self.ckpt else None, "loading": self.loading, "checkpoints": CHECKPOINTS,
            "trial": self.trial, "task": self.task(),
            "trials": [{"i": t["i"], "task": t["task"], "category": t["category"], "train_count": t["train_count"],
                        "outcome": latest.get(t["i"])} for t in trials],
            "setup": "Black: normal starting position. White: only " + ", ".join(f"{p} {s}" for s, p in EVAL["setup_white"].items()),
            "run_s": round(time.monotonic() - self.run["t0"], 1) if self.run else 0,
            "at_ready": self.at_ready(),
            "leader": {"connected": self.leader is not None, "ready": self.leader_ready(), "offsets": self.leader_offsets(),
                       "tol": READY_TOL, "ok": self.leader_ok(), "backoff": self.LEADER_WRIST_BACKOFF, "near": self.LEADER_NEAR_DEG, "error": self._leader_err,
                       "q": (None if self.leader_q is None else {j: round(float(v), 1) for j, v in zip(JOINTS, self.leader_q)})},
            "joints": {j: round(float(v), 1) for j, v in self.joints.items()} if self.joints else None,
            "ready_pose": {j: float(READY[j]) for j in JOINTS},
            "ready": [{"name": {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "wrist",
                                "wrist_roll": "roll", "gripper": "grip"}[j], "off": round(self.joints.get(j, 0) - READY[j], 1),
                       "tol": READY_TOL[j]} for j in JOINTS],
            "last": self.last, "summary": summary, "grid": self.show_grid, "variant": self.variant,
            "pointed": self.pointed, "point_squares": POINT_SQUARES, "taught": list(self.taught), "taught_z": list(self.taught_z),
            "stage": self.staged.stage if self.staged else None, "retries": self.staged.retries if self.staged else 0,
            "repo": "scripted stand-in (no model)" if self.standin else (self.model_repo or __import__("policy_io").VARIANTS[self.variant])
                    + (f" · grasp: {self.grasp_repo.rstrip('/').split('/')[-1]} @ {self.grasp_ckpt}" if self.grasp_repo else ""),
            "active_model": self.active_model,
            "calib": ({"corner": self.calib["order"][self.calib["i"]], "step": self.calib["i"] + 1} if self.calib else None), "home_pct": round(100 * min(1, (time.monotonic() - self.home["t0"]) / self.home["T"])) if self.home else 0,
        }
        if self.game is not None:
            if getattr(self.game, "require_smolvla", False) and not self.model_repo and not self.standin:
                st["repo"] = "No model selected"
            st["game"] = self.game.state()
        with self.lock:
            self.state = json.dumps(st)


def serve(p: Player, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                page = HERE / ("live_game_ui.html" if getattr(p.game, "require_smolvla", False) else "guided_game_ui.html") if p.game is not None else PAGE
                self.send(page.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/trials":
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/record"):
                self.send((HERE / "record_ui.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/state":
                with p.lock:
                    self.send(p.state.encode(), "application/json")
            elif self.path == "/raw":                 # the raw camera frame + full-precision joints (the automatic calibration)
                import base64
                img = getattr(p, "_last_img", None)
                jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1] if img is not None else None
                self.send(json.dumps({"joints": p.joints, "phase": p.phase, "pointed": p.pointed, "sess": p._sess if p._proj is not None else None,
                                      "msg": p.msg, "jpeg_b64": base64.b64encode(jpg.tobytes()).decode() if jpg is not None else None}).encode(),
                          "application/json")
            elif self.path.startswith("/stream.mjpg"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
                last = None
                try:
                    while not p.stop:
                        j = p.jpeg
                        if j is not None and j is not last:
                            last = j
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(1 / 20)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            p.cmds.put(str(body.get("cmd", ""))); self.send(b"{}", "application/json")

    for port_ in range(port, port + 50):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port_), H); break
        except OSError:
            continue
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def fake_robot(full_board=False):
    """The twin-simulated follower with the eval setup on the latest calibrated board (dry runs, no hardware)."""
    sys.path.insert(0, str(ROOT / "goal"))
    import staged as S
    import targets
    from fake_arm import FakeArm
    P = targets.Projector()
    pieces = {sq: (p, P.square_xyz(sq, targets.LATEST)) for side in ("setup_black", "setup_white") for sq, p in EVAL[side].items()}
    if full_board:
        import chess
        pieces = {chess.square_name(sq): (chess.piece_name(piece.piece_type), P.square_xyz(chess.square_name(sq), targets.LATEST))
                  for sq, piece in chess.Board().piece_map().items()}
    return FakeArm(S.Planner(P), pieces, READY6, fps=FPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="final", help="final or a six-digit checkpoint step, e.g. 007500")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--variant", default="v1", choices=["v1", "coords", "marker", "skills"])
    ap.add_argument("--model-repo", default=None, help="load this Hub repo instead of the variant's default (step-3 models)")
    ap.add_argument("--grasp-repo", default=None, help="skills: a grasp-only model for the grasp stage (the place keeps --model-repo)")
    ap.add_argument("--grasp-checkpoint", default="final", help="checkpoint of --grasp-repo (final, 005000, ...)")
    ap.add_argument("--fake", action="store_true", help="no hardware: the follower simulated on the twin (play/fake_arm.py)")
    ap.add_argument("--fake-render", default=None, metavar="DX,DY,YAW",
                    help="--fake: the wrist camera rendered by the twin (pinhole) with an EMPTY board placed this far (m, m, deg) "
                         "from the model's board: a dry run of the automatic calibration (calibration/auto_board_calib.py --pinhole)")
    ap.add_argument("--standin", choices=["scripted"], default=None, help="skills: scripted grasp / place instead of a model")
    ap.add_argument("--sync", action="store_true", help="old behaviour: compute each 50-step chunk on the control thread")
    ap.add_argument("--scripted-game", action="store_true", help="guided 10-minute full game with ready-pose human move checks")
    ap.add_argument("--live-game", action="store_true", help="unscripted human White versus Stockfish with learned grasp/place")
    ap.add_argument("--follower-p", type=int, choices=(16, 24, 32), default=None, help="follower gain; live defaults to paper P=24, other modes P=16")
    ap.add_argument("--no-browser", action="store_true", help="do not automatically open the web page")
    args = ap.parse_args()
    if args.live_game and args.scripted_game:
        ap.error("Choose live or guided game, not both.")
    if args.live_game:
        args.variant, args.sync = "skills", True
        if args.standin and not args.fake:
            ap.error("Live-game stand-ins are allowed only with --fake.")
        # Leave the model empty when none is configured: the page selects it explicitly.
        args.model_repo = args.model_repo or so101_config.LIVE_MODEL_REPO
        import staged
        staged.KEEP_GOING = False
    args.follower_p = args.follower_p or (24 if args.live_game else 16)
    if args.scripted_game:
        args.variant = "skills"
        args.sync = True
        args.model_repo = "scripted:place"
        args.grasp_repo = args.grasp_repo or "astra:gpt-6.1-terra"

    if args.fake:                                      # dry runs never mix with the real evaluation log
        global RESULTS
        RESULTS = HERE / "results_dryrun"
    if args.fake:
        robot = fake_robot(full_board=args.scripted_game or args.live_game)
        if args.fake_render:
            import chess, mujoco
            dx, dy, yaw = (float(v) for v in args.fake_render.split(","))
            scene = robot.Pl.s if hasattr(robot, "Pl") else __import__("targets").Projector().s
            sess = __import__("targets").POSES["sessions"][__import__("targets").LATEST]
            scene.set_board_pose(sess["dx"] + dx, sess["dy"] + dy, sess["yaw_deg"] + yaw)
            board = chess.Board(None)
            for side, colour in (("setup_black", chess.BLACK), ("setup_white", chess.WHITE)):
                for sq, ptype in EVAL[side].items():
                    board.set_piece_at(chess.parse_square(sq), chess.Piece(chess.PIECE_SYMBOLS.index(ptype[0] if ptype != "knight" else "n"), colour))
            mapping = scene.set_position(board); mujoco.mj_forward(scene.model, scene.data)
            for k in robot.pieces:                                    # the truth: the pieces stand on the shifted board,
                robot.pieces[k][1] = np.asarray(scene.square_xyz(k, 0.0), float)   # the player keeps its (biased) belief
            def render_obs(get=robot.get_observation):
                o = get(); scene.set_joints(robot.q)
                for k, (t, pos) in robot.pieces.items():          # the fake arm moves pieces; the render follows
                    if k in mapping:
                        scene.set_piece_pose(mapping[k], pos)
                mujoco.mj_forward(scene.model, scene.data)
                o["wrist"] = scene.render(w=1280, h=720); return o
            robot.get_observation = render_obs
            print(f"fake render: the board {1000 * dx:+.0f}, {1000 * dy:+.0f} mm and {yaw:+.1f} deg from the model's")
    else:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
        cams = {"wrist": OpenCVCameraConfig(index_or_path=args.camera, width=1280, height=720, fps=FPS)}
        robot = SOFollower(SO101FollowerConfig(port=so101_config.FOLLOWER_PORT, id=so101_config.FOLLOWER_ID, cameras=cams,
                                               position_p_coefficient=args.follower_p))
        robot.bus.connect()
        here = robot.bus.sync_read("Present_Position")
        robot.bus.sync_write("Goal_Position", here)          # torque comes on holding the current pose
        robot.bus.disconnect(disable_torque=False)
        robot.connect()
    p = Player(robot, async_exec=not args.sync, variant=args.variant, model_repo=args.model_repo, standin=args.standin,
               grasp_repo=args.grasp_repo, grasp_ckpt=args.grasp_checkpoint, scripted_game=args.scripted_game,
               live_game=args.live_game, follower_p=args.follower_p)
    p.publish()
    httpd = serve(p, args.port)
    url = f"http://localhost:{httpd.server_address[1]}"
    print(f"\n  Policy player: {url}   (keep this Terminal window open)\n", flush=True)
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)
    if not args.live_game or args.model_repo or args.standin:
        p.load(args.checkpoint)
    try:
        p.loop()
    except KeyboardInterrupt:
        pass
    finally:
        if args.live_game:
            p.game.close()
        if not args.fake:
            print("\nHolding the arm where it is (torque stays on). Support it before switching the power off.")
            try:
                robot.bus.disconnect(disable_torque=False)
            except Exception:
                pass
        httpd.shutdown()


if __name__ == "__main__":
    main()
