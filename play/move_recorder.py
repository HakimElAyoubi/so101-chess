"""Record a robot move by teleoperation and replay it as the same joint path.

The user drives the follower with the LEADER arm (5 V supply, leader_port in config.yaml) from the player's own control
loop at 30 Hz; every tick the command sent to the follower and the follower's measured joints are logged. A replay sends
the same command stream at the same timing, so the follower repeats the demonstration (same gains, same load, same
pieces in the scripted game). Files: play/games/recorded/<script>/plyNN_<uci>.npz with t (s), cmd (6, the commands),
meas (6, the follower), uci, ply, fps.
"""
import time
from pathlib import Path
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root
import so101_config  # noqa: E402  (ports and Stockfish path from config.yaml)

HERE = Path(__file__).resolve().parent
RECORD_DIR = HERE / "games" / "recorded"
LEADER_PORT, LEADER_ID = so101_config.LEADER_PORT, so101_config.LEADER_ID
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BLEND_S = 2.5            # at the start of a recording the follower eases from where it is to the leader's pose
TO_START_DEG_S = 40.0    # replay: joint speed of the move to the recording's first pose and of the return home
SETTLE_S = 0.4


def recording_path(script, ply, uci):
    return RECORD_DIR / script / f"ply{int(ply):02d}_{uci}.npz"


def list_recordings(script):
    """{ply: {"uci", "path", "seconds"}} of the saved demonstrations of this script."""
    out = {}
    for p in sorted((RECORD_DIR / script).glob("ply*_*.npz")) if (RECORD_DIR / script).exists() else []:
        try:
            d = np.load(p, allow_pickle=False)
            out[int(d["ply"])] = dict(uci=str(d["uci"]), path=str(p), seconds=round(float(d["t"][-1]) if len(d["t"]) else 0.0, 1))
        except Exception:
            continue
    return out


def minjerk(u):
    u = min(1.0, max(0.0, u))
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class LeaderTeleop:
    """The leader arm as a 6-vector source in JOINTS order (degrees, gripper %)."""

    def __init__(self, port=LEADER_PORT, id=LEADER_ID):
        self.port, self.id, self.teleop = port, id, None

    def connect(self):
        if self.teleop is None:
            from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader
            self.teleop = SOLeader(SO101LeaderConfig(port=self.port, id=self.id))
            self.teleop.connect()
        return self

    @property
    def connected(self):
        return self.teleop is not None

    def action6(self):
        act = self.teleop.get_action()
        return np.array([float(act[f"{j}.pos"]) for j in JOINTS])

    def disconnect(self):
        if self.teleop is not None:
            try:
                self.teleop.disconnect()
            finally:
                self.teleop = None


class MoveRecording:
    """A demonstration in progress: one row per control tick."""

    def __init__(self, path, ply, uci, t0, fps=30):
        self.path, self.ply, self.uci, self.t0, self.fps = Path(path), int(ply), uci, t0, fps
        self.rows = []

    def add(self, now, cmd6, meas6):
        self.rows.append([now - self.t0, *np.asarray(cmd6, float), *np.asarray(meas6, float)])

    @property
    def seconds(self):
        return self.rows[-1][0] if self.rows else 0.0

    def save(self):
        a = np.array(self.rows, float) if self.rows else np.zeros((0, 13))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, t=a[:, 0], cmd=a[:, 1:7], meas=a[:, 7:13], uci=self.uci, ply=self.ply, fps=self.fps)
        return self.path


class ReplayMove:
    """Plays a recording back through the player's loop with the StagedMove interface (step / result / events / info).
    to_start: ease from the current pose to the recording's first command. replay: the recorded commands on the recorded
    clock. return: ease to the ready pose. result = ("placed", "replayed N s")."""

    def __init__(self, path, ready6, t0):
        d = np.load(path, allow_pickle=False)
        self.t, self.cmd = np.asarray(d["t"], float), np.asarray(d["cmd"], float)
        if len(self.t) < 2:
            raise ValueError(f"recording {path} is empty")
        self.uci = str(d["uci"]); self.ready = np.asarray(ready6, float)
        self.stage, self.result, self.retries, self.close_h, self.ring = "to_start", None, 0, None, None
        self.events = [(round(t0, 2), "to_start", f"replay of {Path(path).name}")]
        self.info = {"replay": str(path), "recorded_s": round(float(self.t[-1]), 1)}
        self._seg = None                      # (q0, q1, t0, T) of the current eased move

    def on_path(self):
        return True

    def _ease(self, q0, q1, now):
        T = max(1.0, float(np.abs(np.asarray(q1)[:5] - np.asarray(q0)[:5]).max()) / TO_START_DEG_S)
        self._seg = (np.asarray(q0, float), np.asarray(q1, float), now, T)

    def _eased(self, now):
        q0, q1, t0, T = self._seg
        u = (now - t0) / T
        return q0 + (q1 - q0) * minjerk(u), u >= 1.0 and now - t0 >= T + SETTLE_S

    def step(self, q6, img, now):
        q6 = np.asarray(q6, float)
        if self.result is not None:
            return None
        if self.stage == "to_start":
            if self._seg is None:
                self._ease(q6, self.cmd[0], now)
            a, done = self._eased(now)
            if done:
                self.stage, self._t0, self._seg = "replay", now, None
                self.events.append((round(now, 2), "replay", f"{self.t[-1]:.1f} s of demonstration"))
            return a
        if self.stage == "replay":
            u = now - self._t0
            if u >= self.t[-1]:
                self.stage = "return"; self._ease(self.cmd[-1], self.ready, now)
                self.events.append((round(now, 2), "return", ""))
                return self.cmd[-1]
            return np.array([np.interp(u, self.t, self.cmd[:, k]) for k in range(6)])
        if self.stage == "return":
            a, done = self._eased(now)
            if done:
                self.result = ("placed", f"replayed {self.t[-1]:.1f} s")
                self.events.append((round(now, 2), "done", ""))
                return None
            return a
        return None
