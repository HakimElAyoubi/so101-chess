"""Dry-run stand-ins for the staged player (no hardware, no skill model).

FakeArm: the follower simulated on the twin. Joints follow the command with a first-order lag like the real P=16 servos
(~0.15 s); the gripper stalls at ~4 % when it closes on a piece standing between the jaws (the real follower reads 3.95 %
median while holding); a held piece follows the jaws and is put down where the jaws open. Same interface as LeRobot's
SOFollower for the player (get_observation / send_action), plus tick(dt) for headless tests.

ScriptedSkill: a stand-in for the learned grasp / place skill, used only to exercise the staged executor: open, go
straight down, close (grasp) / go down, open (place). It reads the target from the StagedMove instead of the image.
offset_m shifts the first grasp attempt sideways to provoke a miss (tests the retry path)."""
import numpy as np

import staged  # noqa: F401  (puts the project root on sys.path)
from chess_sim import config as C

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LAG_S, MAX_DEG_S = 0.15, 150.0
STALL_PCT, RELEASE_PCT = 4.0, 6.0


class FakeArm:
    def __init__(self, planner, pieces, q0, fps=30):
        """pieces: {square or name: (piece type, xyz of its base)}; q0: initial joints [6]."""
        self.Pl, self.pieces, self.fps = planner, {k: [t, np.asarray(p, float)] for k, (t, p) in pieces.items()}, fps
        self.q = np.asarray(q0, float).copy(); self.target = self.q.copy()
        self.held, self.hang = None, 0.0
        self.image = np.full((720, 1280, 3), 90, np.uint8)

    # ---- LeRobot-like interface (the player calls these once per 30 Hz tick)
    def get_observation(self):
        self.tick(1 / self.fps)
        return {**{f"{j}.pos": float(v) for j, v in zip(JOINTS, self.q)}, "wrist": self.image}

    def send_action(self, a):
        self.target = np.array([a[f"{j}.pos"] for j in JOINTS], float)

    # ---- physics (kinematic)
    def _between_jaws(self, site):
        for k, (t, p) in self.pieces.items():
            h = C.PIECE_SPECS[t][0]
            if np.hypot(*(site[:2] - p[:2])) < 0.010 and p[2] + 0.004 < site[2] < p[2] + h:
                return k
        return None

    def tick(self, dt):
        a = min(1.0, dt / LAG_S)
        step = np.clip((self.target - self.q) * a, -MAX_DEG_S * dt, MAX_DEG_S * dt)
        q = self.q + step
        site, _ = self.Pl.fk(q)
        if self.held is None and self.target[5] < STALL_PCT:
            k = self._between_jaws(site)
            if k is not None and q[5] <= STALL_PCT + 0.5:
                self.held = k; self.hang = site[2] - self.pieces[k][1][2]
        if self.held is not None:
            if self.target[5] > RELEASE_PCT:
                p = self.pieces[self.held][1]; p[:2] = site[:2]; p[2] = self.Pl.z0
                self.held = None
            else:
                q[5] = max(q[5], STALL_PCT)                     # the jaws stall on the piece
                self.pieces[self.held][1] = np.array([site[0], site[1], site[2] - self.hang])
        self.q = q


class ScriptedSkill:
    """start(text, which, to_xyz) / step(q6, img), as the learned skill driver."""

    def __init__(self, planner, fps=30, offset_m=0.0):
        self.Pl, self.dt, self.offset = planner, 1 / fps, offset_m
        self.move, self.plan, self.k, self.attempt = None, None, 0, 0

    def start(self, text, which, to_xyz):
        m, Pl = self.move, self.Pl
        self.k = 0
        if text.startswith("pick up"):
            self.attempt += 1
            xyz = m.from_xyz.copy()
            if self.attempt == 1 and self.offset:
                xyz[1] += self.offset                          # miss on purpose: retry test
            h = C.PIECE_SPECS[m.piece][0] * (0.5 if m.piece == "king" else 0.62)
            seq = [(0.5, None, 20.0), (1.5, h, 20.0), (0.6, h, 0.8), (1.0, h, 0.8)]     # (s, height or None = stay, gripper %)
        else:
            xyz = m.to_xyz.copy()
            h = m.close_h + 0.003
            seq = [(1.2, h, 0.8), (0.5, None, 12.0), (1.0, None, 12.0)]
        self.seq, self.xyz = seq, xyz

    def step(self, q6, img):
        if self.k == 0:
            self.q_start = np.asarray(q6, float).copy()
            self.pts = []
            t, s = self.Pl.orientation(q6)
            prev, g_prev, z_prev = self.q_start, q6[5], self.Pl.fk(q6)[0][2]
            for dur, h, g in self.seq:                          # waypoints: (time, joints)
                z = z_prev if h is None else self.Pl.z0 + h
                n = max(1, int(dur / self.dt))
                for i in range(1, n + 1):
                    zi = z_prev + (z - z_prev) * i / n
                    q, _ = self.Pl.ik_near([self.xyz[0], self.xyz[1], zi], t, s, prev, g_prev + (g - g_prev) * i / n)
                    if q is None:
                        q = prev.copy(); q[5] = g_prev + (g - g_prev) * i / n
                    self.pts.append(q); prev = q
                g_prev, z_prev = g, z
        a = self.pts[min(self.k, len(self.pts) - 1)]
        self.k += 1
        return a
