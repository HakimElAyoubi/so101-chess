"""Grasp drill recorder: one grasp per episode, each starting exactly where the staged player hands over to the skill.

    scripts/record_grasp_drill.sh   (the page opens in your browser; plan: recording/grasp_plan.py)

For each episode of recording/grasp_drill.json:
  1. Put the piece on the square at the spot shown (by hand, or with the arm: between episodes the follower follows the
     leader). Enter / "Arm to the hover": both arms glide to the player's grasp hover above the square (the follower
     holds it exactly, gravity sag taken out; the leader's motors hold it there too).
  2. Check the live view: the green dot is what the model sees (the square's centre), the white ring where the piece
     should stand. Space / Start: the leader is released and the follower follows it from the hover.
  3. Grasp the piece, lift it ~3 cm, hold it a moment, Space / Stop: saved. R discards it (put the piece back, redo).
Episodes go to LeHak0/so101_chess_grasp (its own dataset), the conditions of each to recording/grasp_meta.jsonl
(board pose, square, spot, hover); progress to recording/grasp_drill_progress.json (resumable).
--fake: simulated arms and camera, no hardware, the dataset under --root: a dry run of the whole flow.
"""
import argparse
import json
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "play"), str(ROOT / "goal")]
from chess_recorder import FOLLOWER_P, FPS, JOINT_NAMES, env_state  # noqa: E402

PAGE = HERE / "grasp_ui.html"
PLAN = HERE / "grasp_drill.json"
PROGRESS = HERE / "grasp_drill_progress.json"
META = HERE / "grasp_meta.jsonl"
ENGAGE_T = 0.8              # s: the follower blends from the held hover onto the live leader at Start
LEADER_WRIST_MAX = 100.0    # deg: the leader's handle meets its own arm just above this (limit 102.4)
PATH_STEP_DEG = 8.0         # per tick, planned paths (as play/policy_player.py)
TELEOP_HZ = 75.0            # the leader -> follower link runs in its own thread at this rate (0: inside the 30 Hz loop, the old way)


class Drill:
    def __init__(self):
        d = json.loads(PLAN.read_text())
        self.eps, self.spot_mm = d["episodes"], d["spot_mm"]
        prog = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
        self.idx, self.done = int(prog.get("next", 0)), prog.get("done", {})

    def save(self):
        PROGRESS.write_text(json.dumps({"next": self.idx, "done": self.done}, indent=1))

    @property
    def cur(self):
        return self.eps[self.idx] if self.idx < len(self.eps) else None


class Session:
    """Owns the hardware; 30 Hz control loop in its own thread, the web handlers only queue commands."""

    def __init__(self, rig, push):
        import staged as S
        import targets
        self.rig, self.push, self.S, self.T = rig, push, S, targets
        self.P = targets.Projector(); self.Pl = S.Planner(self.P); self.sess = targets.LATEST
        self.ready6 = np.array([rig.ready[j] for j in JOINT_NAMES], float)
        self.drill = Drill()
        self.cmds, self.lock = queue.Queue(), threading.Lock()
        self.phase = "place" if self.drill.cur else "done"    # place | moving | ready | recording | saving | closing | done
        self.msg, self.msg_kind, self.error = "", "info", ""
        self.joints, self.jpeg, self.stop, self.session_eps = {}, None, False, 0
        self.move = None            # auto-move to the hover: {"traj", "tau", "prev", "lead0", "fol0", "bias", "held"}
        self.hover = None           # the hover joints of the current episode
        self.bias = np.zeros(6)     # sag correction the follower holds the hover with
        self.engage_t0 = None
        self.state = "{}"
        self.lk_fol, self.lk_lead = threading.Lock(), threading.Lock()     # one serial transaction at a time per bus
        self.teleop_hz, self.obs_latest, self.last_act, self.teleop_stats = TELEOP_HZ, None, None, {"hz": 0.0, "ms": 0.0}

    def say(self, m, k="info"):
        self.msg, self.msg_kind = m, k

    def q(self):
        return np.array([self.joints[j] for j in JOINT_NAMES])

    # ------------------------------------------------------------------ commands (control thread)
    def handle(self, c):
        ph, e = self.phase, self.drill.cur
        if c == "hover" and ph == "place" and e:
            self.start_move()
        elif c == "start" and ph == "ready":
            with self.lk_lead:
                self.rig.teleop.bus.disable_torque()      # the leader is yours, at the hover
            self.engage_t0, self.phase = time.monotonic(), "recording"
            self.say("Recording: grasp the piece, lift it ~3 cm, hold it a moment, then Stop (Space).")
        elif c == "stop" and ph == "recording":
            if not self.rig.dataset.has_pending_frames():
                self.phase = "place"; return
            self.phase = "saving"; self.publish()
            self.rig.dataset.save_episode()
            ep = self.rig.dataset.num_episodes - 1
            with META.open("a") as f:
                f.write(json.dumps({"episode_index": ep, "drill": e["i"], "task": e["task"], "colour": e["colour"],
                                    "piece": e["piece"], "square": e["square"], "spot": e["spot"],
                                    "offset_mm": e["offset_mm"], "board_session": self.sess,
                                    "hover": [round(float(v), 2) for v in self.hover], "follower_p": FOLLOWER_P, "teleop_hz": self.teleop_hz,
                                    "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            self.drill.done[str(e["i"])] = ep; self.drill.idx += 1; self.drill.save(); self.session_eps += 1
            self.next_episode("Saved. ")
        elif c == "redo" and ph == "recording":
            self.rig.dataset.clear_episode_buffer()
            self.phase = "place"
            self.say("Discarded. Put the piece back at the spot shown, then the arm to the hover again (Enter).", "warn")
        elif c == "skip" and ph in ("place", "ready") and e:
            self.release_leader()
            self.drill.idx += 1; self.drill.save()
            self.next_episode("Skipped. ")
        elif c == "back" and ph in ("place", "ready") and self.drill.idx > 0:
            self.release_leader()
            self.drill.idx -= 1; self.drill.save(); self.phase = "place"
            self.say("Back one: that grasp will be recorded again (the earlier recording stays in the dataset).")
        elif c == "follow" and ph == "ready":             # changed my mind: give the arm back without recording
            self.release_leader(); self.phase = "place"
            self.say("The follower follows the leader again.")
        elif c.startswith("fake_lead:") and getattr(self.rig, "fake", False):   # dry run: the 'human' moves the leader
            self.rig.teleop.set(json.loads(c[10:]))
        elif c == "finish" and ph in ("place", "ready", "done"):
            self.release_leader()
            self.phase = "closing"; self.say("Saving the dataset" + (" and uploading it..." if self.push else "...")); self.publish()
            try:
                self.rig.close(push=self.push)
                self.say(f"All done: {self.rig.dataset.num_episodes} episodes in {self.rig.dataset.repo_id}"
                         + (" (uploaded)." if self.push else "."), "good")
            except Exception as ex:
                self.say(f"Problem while saving: {ex}", "warn")
            self.phase, self.stop = "done", True

    def next_episode(self, prefix):
        e = self.drill.cur
        self.phase = "place" if e else "done"
        self.say(prefix + ("Next: " + self.instruction(e) if e else "Drill complete: every grasp recorded. Finish to save and upload."), "good")

    def instruction(self, e):
        return (f"{e['note'] + ' ' if e['note'] else ''}Put the {e['colour']} {e['piece']} on {e['square']}"
                + ("" if e["spot"] == "centre" else f", {e['spot']} (~{self.drill.spot_mm:.0f} mm off centre)") + ".")

    def release_leader(self):
        try:
            with self.lk_lead:
                self.rig.teleop.bus.disable_torque()
        except Exception:
            pass

    # ------------------------------------------------------------------ auto-move both arms to the hover
    def start_move(self):
        S, Pl, e = self.S, self.Pl, self.drill.cur
        q0 = self.q()
        hover, info = Pl.grasp_hover(self.P.square_xyz(e["square"], self.sess), e["piece"], self.ready6, S.HOVER_GRIP)
        if hover is None:
            self.say(f"No IK solution for the hover above {e['square']}: skip this one (S).", "warn"); return
        # from wherever the arm is: straight up first if it is low, then the approach (above the square, straight down)
        top = list(Pl.rise_to(q0, S.APPROACH_Z)) if Pl.fk(q0)[0][2] < Pl.z0 + S.APPROACH_Z - 0.01 else [q0]
        W = top + list(Pl.approach(top[-1], hover))[1:]
        lead_bus = self.rig.teleop.bus
        with self.lk_lead:
            here = lead_bus.sync_read("Present_Position")
            lead_bus.sync_write("Goal_Position", here)      # hold where it is: no jump when the torque comes on
            lead_bus.enable_torque()
        lead0 = np.array([float(here[j]) for j in JOINT_NAMES])
        self.hover, self.bias = hover, np.zeros(6)
        self.move = {"traj": S.Trajectory(W), "tau": 0.0, "prev": time.monotonic(), "lead0": lead0, "fol0": np.asarray(W[0], float),
                     "held": None}
        self.phase = "moving"
        self.say(f"Both arms to the hover above {e['square']}. Hands off the leader.")

    def move_step(self):
        S, m = self.S, self.move
        now, q = time.monotonic(), self.q()
        tr = m["traj"]
        m["tau"] = tr.advance(m["tau"], q, now - m["prev"]); m["prev"] = now
        k = min(1.0, m["tau"] / tr.T)
        target = tr.at(m["tau"])
        lead = target + (m["lead0"] - m["fol0"]) * (1 - k)  # the leader joins the follower's path along the way
        lead[3] = min(lead[3], LEADER_WRIST_MAX)
        with self.lk_lead:
            self.rig.teleop.send_feedback({f"{j}.pos": float(v) for j, v in zip(JOINT_NAMES, lead)})
        if not tr.done(m["tau"]):
            self.send(target + self.bias); return
        err = self.hover[:5] - q[:5]                       # at the hover: take out the sag, as the player does
        if np.abs(err).max() > S.SAG_TOL_DEG and now - (m["held"] or now) < S.SAG_MAX_S:
            m["held"] = m["held"] or now
            self.bias[:5] = np.clip(self.bias[:5] + S.SAG_GAIN * err, -S.SAG_MAX_DEG, S.SAG_MAX_DEG)
            self.send(self.hover + self.bias); return
        self.move, self.phase = None, "ready"
        self.say("At the hover. Check the piece against the white ring in the view (the green dot is what the model sees), "
                 "then Start (Space).", "good")

    def send(self, target, max_step=PATH_STEP_DEG):
        q = self.q()
        t = np.clip(np.asarray(target, float), q - max_step, q + max_step)
        with self.lk_fol:
            self.rig.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINT_NAMES, t)})

    def teleop_target(self, act, obs):
        """The leader's action -> (what the dataset records, what the follower is sent). At Start the follower blends
        from the held hover onto the live leader over ENGAGE_T."""
        act_p = self.rig.t_proc((act, obs))
        if self.phase == "recording" and time.monotonic() - self.engage_t0 < ENGAGE_T:
            u = (time.monotonic() - self.engage_t0) / ENGAGE_T; k = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            held = self.hover + self.bias
            act_p = {f"{j}.pos": float(held[i] + (float(act_p[f"{j}.pos"]) - held[i]) * k) for i, j in enumerate(JOINT_NAMES)}
            return act_p, act_p
        return act_p, self.rig.r_proc((act_p, obs))

    def teleop_loop(self):
        """The leader -> follower link in its own thread at TELEOP_HZ: the follower gets a fresh target every ~13 ms
        instead of once per 30 Hz tick (the tick also reads the follower and the camera and writes the dataset). Each
        bus is used by one thread at a time (lk_lead, lk_fol), and the follower is only written while the leader is in
        charge (phase place / recording), checked under the lock so a phase change never races a stale target."""
        rig, period = self.rig, 1 / self.teleop_hz
        n, t_win, lat = 0, time.perf_counter(), []
        deadline = time.perf_counter()                 # fixed deadlines: a late tick does not shorten the next period
        while not self.stop:
            deadline += period
            tick = time.perf_counter()
            obs = self.obs_latest
            if self.phase in ("place", "recording") and obs is not None:
                try:
                    with self.lk_lead:
                        act = rig.teleop.get_action()
                    act_p, cmd = self.teleop_target(act, obs)
                    with self.lk_fol:
                        if self.phase in ("place", "recording"):
                            rig.robot.send_action(cmd)
                    self.last_act = act_p
                    lat.append(time.perf_counter() - tick); n += 1
                except Exception as ex:
                    self.last_act = None; self.error = f"Teleop error: {ex}"; time.sleep(0.2)
            else:
                self.last_act = None
            if time.perf_counter() - t_win >= 1.0:
                self.teleop_stats = {"hz": round(n / (time.perf_counter() - t_win), 1), "ms": round(1000 * float(np.mean(lat)), 1) if lat else 0.0}
                n, t_win, lat = 0, time.perf_counter(), []
            now = time.perf_counter()
            if deadline < now - period:                # far behind (a slow bus call): restart the schedule from now
                deadline = now
            time.sleep(max(0.0, deadline - now))

    # ------------------------------------------------------------------ control loop
    def loop(self):
        from lerobot.utils.feature_utils import build_dataset_frame
        rig, period, n = self.rig, 1 / FPS, 0
        if self.teleop_hz > 0:
            threading.Thread(target=self.teleop_loop, daemon=True).start()
        while not self.stop:
            tick = time.perf_counter()
            try:
                with self.lk_fol:
                    obs = rig.robot.get_observation()
                self.obs_latest = obs
                self.joints = {j: float(obs[f"{j}.pos"]) for j in JOINT_NAMES}
                if self.phase == "moving":
                    self.move_step()
                elif self.phase == "ready":
                    self.send(self.hover + self.bias)          # hold the hover exactly
                elif self.phase in ("place", "recording"):
                    obs_p = rig.o_proc(obs)
                    act_p = self.last_act if self.teleop_hz > 0 else None
                    if act_p is None:                          # the old single-loop way, or the thread has not run yet
                        with self.lk_lead:
                            act = rig.teleop.get_action()
                        act_p, cmd = self.teleop_target(act, obs)
                        with self.lk_fol:
                            rig.robot.send_action(cmd)
                    if self.phase == "recording":
                        e = self.drill.cur
                        rig.dataset.add_frame({**build_dataset_frame(rig.hw_features, obs_p, prefix="observation"),
                                               **build_dataset_frame(rig.dataset.features, act_p, prefix="action"),
                                               "observation.environment_state": env_state(chess.parse_square(e["square"]), None),
                                               "task": e["task"]})
                n += 1
                if n % 2 == 0 and "wrist" in obs:
                    self.publish_frame(obs["wrist"])
                self.error = ""
            except Exception as ex:
                traceback.print_exc()
                self.error = f"Arm/camera error: {ex}"
                if self.phase == "moving":
                    self.move, self.phase = None, "place"; self.release_leader()
                time.sleep(0.2)
            while not self.cmds.empty():
                try:
                    self.handle(self.cmds.get_nowait())
                except Exception as ex:
                    traceback.print_exc(); self.say(f"Error: {ex}", "warn")
            self.publish()
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def publish_frame(self, rgb):
        view = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, 360))
        e = self.drill.cur
        if e and self.phase in ("ready", "moving", "recording"):
            c = self.P.square_xyz(e["square"], self.sess, z=self.T.PIECE_MID_Z)
            a = np.radians(self.T.POSES["sessions"][self.sess]["yaw_deg"])
            rank8 = -np.array([np.cos(a), np.sin(a)]); hfile = np.array([-np.sin(a), np.cos(a)])   # board axes in the world
            spot = c.copy(); spot[:2] += 0.001 * (e["offset_mm"][0] * rank8 + e["offset_mm"][1] * hfile)
            spot[2] = self.P.s.cfg.board_thickness              # the ring: where the piece's base stands (board level)
            uv, vis = self.P.project(self.q(), np.stack([c, spot]), 640)
            if vis[1] and self.phase != "recording":
                cv2.circle(view, tuple(int(v) for v in uv[1]), 11, (255, 255, 255), 2, cv2.LINE_AA)
            if vis[0]:
                cv2.circle(view, tuple(int(v) for v in uv[0]), 5, (0, 220, 0), -1, cv2.LINE_AA)
        self.jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def publish(self):
        d, e = self.drill, self.drill.cur
        parts = {p: (sum(1 for x in d.eps if x["part"] == p and str(x["i"]) in d.done), sum(1 for x in d.eps if x["part"] == p)) for p in "ACB"}
        st = {"phase": self.phase, "msg": self.msg, "msg_kind": self.msg_kind, "error": self.error,
              "idx": d.idx, "n": len(d.eps), "done": len(d.done), "parts": parts, "session_eps": self.session_eps,
              "cur": ({**e, "text": self.instruction(e)} if e else None),
              "rec_s": round(time.monotonic() - self.engage_t0, 1) if self.phase == "recording" else 0,
              "move_pct": round(100 * min(1.0, self.move["tau"] / self.move["traj"].T)) if self.move else 0,
              "repo": self.rig.dataset.repo_id, "episodes": self.rig.dataset.num_episodes, "board": self.sess, "teleop": self.teleop_stats,
              "fake": bool(getattr(self.rig, "fake", False))}
        with self.lock:
            self.state = json.dumps(st)


# ---------------------------------------------------------------------- dry run: simulated arms
class _Bus:
    def __init__(self, arm):
        self.arm = arm

    def sync_read(self, what):
        return {j: float(v) for j, v in zip(JOINT_NAMES, self.arm.qq)}

    def sync_write(self, what, vals):
        pass

    def enable_torque(self):
        pass

    def disable_torque(self):
        pass


class _FakeFollower:
    LAG_S = 0.15

    def __init__(self, q, features_from):
        self.qq, self.target, self.t = np.array(q, float), np.array(q, float), time.monotonic()
        self.bus, self.name = _Bus(self), "so_follower"
        self.action_features, self.observation_features = features_from.action_features, features_from.observation_features

    def get_observation(self):
        now = time.monotonic(); a = 1 - np.exp(-(now - self.t) / self.LAG_S); self.t = now
        self.qq += (self.target - self.qq) * a
        img = np.full((720, 1280, 3), 90, np.uint8)
        cv2.putText(img, "dry run (no camera)", (420, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        return {**{f"{j}.pos": float(v) for j, v in zip(JOINT_NAMES, self.qq)}, "wrist": img}

    def send_action(self, act):
        self.target = np.array([float(act[f"{j}.pos"]) for j in JOINT_NAMES])
        return act

    def disconnect(self):
        pass


class _FakeLeader:
    def __init__(self, q):
        self.qq, self.bus = np.array(q, float), None
        self.bus = _Bus(self)

    def get_action(self):
        return {f"{j}.pos": float(v) for j, v in zip(JOINT_NAMES, self.qq)}

    def send_feedback(self, fb):
        self.qq = np.array([float(fb[f"{j}.pos"]) for j in JOINT_NAMES])

    def set(self, q):
        self.qq = np.array(q, float)

    def disconnect(self):
        pass


class FakeRig:
    """Same interface as chess_recorder.Rig, simulated arms, a real dataset (under root) so the build can be tested."""
    fake = True

    def __init__(self, repo_id, root):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.processor import make_default_processors
        from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
        from lerobot.utils.feature_utils import combine_feature_dicts
        self.ready = json.loads((HERE / "ready_pose.json").read_text())
        r6 = [self.ready[j] for j in JOINT_NAMES]
        real = SOFollower(SO101FollowerConfig(port="none", id="follower_arm_1",
                                              cameras={"wrist": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=FPS)}))
        self.robot, self.teleop = _FakeFollower(r6, real), _FakeLeader(r6)
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
        else:
            self.dataset = LeRobotDataset.create(repo_id, FPS, features=features, root=root, robot_type="so_follower", use_videos=True, **kw)
        self.hw_features = {k: v for k, v in self.dataset.features.items() if k != "observation.environment_state"}

    def close(self, push, private=True):
        self.dataset.finalize()


def serve(session, port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/state":
                with session.lock:
                    self.send(session.state.encode(), "application/json")
            elif self.path.startswith("/stream.mjpg"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
                last = None
                try:
                    while not session.stop:
                        j = session.jpeg
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
            session.cmds.put(str(body.get("cmd", ""))); self.send(b"{}", "application/json")

    for p in range(port, port + 50):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), H); break
        except OSError:
            continue
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="LeHak0/so101_chess_grasp")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--fake", action="store_true", help="simulated arms and camera: a dry run (dataset under --root)")
    ap.add_argument("--root", default=None, help="dataset folder (default: the Hugging Face cache)")
    ap.add_argument("--teleop-hz", type=float, default=TELEOP_HZ, help="leader -> follower thread rate; 0 = inside the 30 Hz loop (the old way)")
    args = ap.parse_args()
    root = Path(args.root) if args.root else Path.home() / ".cache/huggingface/lerobot" / args.repo_id
    if args.fake:                       # a dry run keeps its progress and conditions next to its own dataset
        global PROGRESS, META
        PROGRESS, META = root.parent / "grasp_drill_progress.json", root.parent / "grasp_meta.jsonl"
        rig = FakeRig(args.repo_id, root)
    else:
        from chess_recorder import Rig
        rig = Rig(args.repo_id, root, args.camera, 1280, 720)
    s = Session(rig, push=not (args.no_push or args.fake)); s.teleop_hz = args.teleop_hz
    s.publish()
    httpd = serve(s, args.port)
    url = f"http://localhost:{httpd.server_address[1]}"
    print(f"\n  Grasp drill recorder: {url}   (keep this Terminal window open)\n", flush=True)
    import webbrowser; webbrowser.open(url)
    try:
        s.loop()
    except KeyboardInterrupt:
        print("\nCtrl-C: saving what was recorded (no upload)...")
        rig.close(push=False)
    time.sleep(2)
    httpd.shutdown()


if __name__ == "__main__":
    main()
