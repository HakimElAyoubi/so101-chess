"""Follower shakiness test: scripted moves around the ready pose, logging commanded vs measured positions at ~50 Hz.

    python calibration/shake_test.py

Starts from wherever the follower is (normally folded), glides to the ready pose, then:
holds, per-joint slow sine sweeps (small amplitudes, shoulder never leans far back), gripper open/close,
point-to-point reaches with holds, a quicker pan step (ringing), and returns to the starting pose with torque off.
Safety: min-jerk motion <= 25 deg/s (one 60 deg/s pan step), torque comes on holding the current pose (no jump),
stops and HOLDS if any joint lags its command by > 15 deg for 0.4 s, Ctrl-C holds the current pose.
Log -> calibration/shake_output/shake_<time>.npz, progress -> data/progress.txt
--far <square>: the same test around a far-reach pose, 9 cm above that square (board cleared of pieces!): small
sweeps only, the whole path checked in the twin to stay 5 cm above the board. --summarize <npz>: the numbers of a run.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root
import so101_config  # noqa: E402  (ports and Stockfish path from config.yaml)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "shake_output"
PROG = ROOT / "data" / "progress.txt"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HZ, SPEED = 50, 25.0
READY = json.loads((ROOT / "recording" / "ready_pose.json").read_text())


def minjerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def move(q0, q1, speed=SPEED, min_t=1.0):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    T = max(min_t, 1.875 * np.abs(q1 - q0)[:5].max() / speed)
    n = int(np.ceil(T * HZ))
    return [q0 + (q1 - q0) * minjerk(i / n) for i in range(1, n + 1)]


def hold(q, secs):
    return [np.asarray(q, float)] * int(secs * HZ)


def sine(q, joint, amp, freq=0.2, cycles=2):
    q = np.asarray(q, float)
    n = int(cycles / freq * HZ)
    out = []
    for i in range(n):
        p = q.copy()
        p[J.index(joint)] += amp * np.sin(2 * np.pi * freq * i / HZ)
        out.append(p)
    return out


def plan_far(start, R, via):
    """Around a far-reach pose R (9 cm above a far square): small sweeps, pan reaches and a pan step; start -> via -> R."""
    off = lambda **d: R + np.array([d.get(j, 0.0) for j in ["pan", "lift", "elbow", "wrist", "roll", "grip"]])
    segs = [("to via (high)", move(start, via, speed=20)), ("to far pose", move(via, R, speed=15)), ("hold far", hold(R, 4))]
    for name, j, a in [("sine pan ±10°", "shoulder_pan", 10), ("sine lift ±3°", "shoulder_lift", 3),
                       ("sine elbow ±5°", "elbow_flex", 5), ("sine wrist flex ±8°", "wrist_flex", 8),
                       ("sine wrist roll ±20°", "wrist_roll", 20)]:
        segs += [(name, sine(R, j, a)), ("settle", hold(R, 1))]
    g = off(grip=40)
    segs += [("gripper open/close", move(R, g, min_t=1) + hold(g, 1) + move(g, R, min_t=1)), ("settle", hold(R, 1))]
    P1, P2 = off(pan=10), off(pan=-10)
    segs += [("reach right", move(R, P1)), ("hold right", hold(P1, 3)), ("reach left", move(P1, P2)),
             ("hold left", hold(P2, 3)), ("back to far", move(P2, R)), ("hold far", hold(R, 2))]
    segs += [("fast pan step +10°", move(R, P1, speed=60, min_t=0.3)), ("ring hold", hold(P1, 2)),
             ("fast pan step -20°", move(P1, P2, speed=60, min_t=0.3)), ("ring hold", hold(P2, 2)),
             ("back to far", move(P2, R)), ("final hold", hold(R, 3)), ("to via (high)", move(R, via, speed=15)),
             ("back to start", move(via, start, speed=20)), ("settle", hold(start, 1))]
    return segs


def far_pose(square, h=0.09, h_via=0.15):
    """(far pose, via pose, planner) from the twin: the player's grasp-hover tilt above `square`, h / h_via above the board."""
    sys.path[:0] = [str(ROOT / "play"), str(ROOT / "goal"), str(ROOT / "calibration"), str(ROOT)]
    import staged as S, targets
    P = targets.Projector(); Pl = S.Planner(P); sess = targets.LATEST
    ready6 = np.array([READY[j] for j in J], float)
    xyz = P.square_xyz(square, sess)
    hov, info = Pl.grasp_hover(xyz, "pawn", ready6, 10.0)
    tilt = float(info.get("tilt", 0.0)) if hov is not None else 0.0
    q = via = None                      # at full stretch only a tilted wrist reaches 9 cm up: take the least tilt that does
    for t in (tilt, 20.0, 30.0, 40.0):
        q = q if q is not None else Pl.ik([xyz[0], xyz[1], xyz[2] + h], t, S.PSI_DEG, ready6, 10.0)
    for t in (30.0, 40.0, 50.0):
        via = via if via is not None else Pl.ik([xyz[0], xyz[1], xyz[2] + h_via], t, S.PSI_DEG, ready6, 10.0)
    if q is None or via is None:
        sys.exit(f"no reachable pose {h:.2f} / {h_via:.2f} m above {square}")
    return np.asarray(q, float), np.asarray(via, float), Pl


def summarize(path):
    """Per joint: rms tracking error, lag (cross-correlation of the velocities), time stuck (command moving, joint
    not), mean offset during the holds; printed, and returned as a dict."""
    d = np.load(path, allow_pickle=True); t, g, pz, lbl = d["t"], d["goal"], d["pos"], d["label"]
    hz = float(d["hz"]) if "hz" in d else HZ
    hz_real = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else hz
    out, names = {}, ["pan", "lift", "elbow", "wrist flex", "roll"]
    hold_mask = np.array([str(x).startswith(("hold", "ring hold", "final hold", "settle")) for x in lbl])
    print(f"{path.name}: {len(t)} ticks at {hz_real:.1f} Hz, gains {d['gains']}")
    print(f"   {'joint':10s} {'rms err':>8s} {'lag ms':>7s} {'stuck %':>8s} {'hold offset':>12s}")
    for j, name in enumerate(names):
        e = g[:, j] - pz[:, j]; rms = float(np.sqrt(np.mean(e ** 2)))
        dg, dp = np.diff(g[:, j]), np.diff(pz[:, j]); lag = np.nan
        if dg.std() > 1e-6 and dp.std() > 1e-6:
            a, b = (dg - dg.mean()) / dg.std(), (dp - dp.mean()) / dp.std()
            c = [np.mean(a[:len(a) - k] * b[k:]) for k in range(0, int(hz_real * 0.6))]
            lag = 1000 * int(np.argmax(c)) / hz_real
        moving = np.abs(dg) > 0.1; stuck = float(np.mean(np.abs(dp[moving]) < 0.02)) if moving.any() else 0.0
        offs = float(np.mean(e[hold_mask])) if hold_mask.any() else np.nan
        out[name] = dict(rms=rms, lag_ms=lag, stuck=stuck, hold_offset=offs)
        print(f"   {name:10s} {rms:8.2f} {lag:7.0f} {100 * stuck:8.0f} {offs:+12.2f}")
    return out


def plan(start):
    R = np.array([READY[j] for j in J])
    off = lambda **d: R + np.array([d.get(j, 0.0) for j in ["pan", "lift", "elbow", "wrist", "roll", "grip"]])
    segs = [("to ready", move(start, R, speed=20)), ("hold ready", hold(R, 4))]
    for name, j, a in [("sine pan ±15°", "shoulder_pan", 15), ("sine lift ±6°", "shoulder_lift", 6),
                       ("sine elbow ±10°", "elbow_flex", 10), ("sine wrist flex ±15°", "wrist_flex", 15),
                       ("sine wrist roll ±30°", "wrist_roll", 30)]:
        segs += [(name, sine(R, j, a)), ("settle", hold(R, 1))]
    g = off(grip=40)
    segs += [("gripper open/close", move(R, g, min_t=1) + hold(g, 1) + move(g, R, min_t=1) + move(R, g, min_t=1)
              + hold(g, 1) + move(g, R, min_t=1)), ("settle", hold(R, 1))]
    P1, P2 = off(pan=20, lift=5, elbow=-5, wrist=5), off(pan=-20, lift=5, elbow=-5, wrist=5)
    segs += [("reach right", move(R, P1)), ("hold right", hold(P1, 3)), ("reach left", move(P1, P2)),
             ("hold left", hold(P2, 3)), ("back to ready", move(P2, R)), ("hold ready", hold(R, 2))]
    S1, S2 = off(pan=20), off(pan=-20)
    segs += [("fast pan step +20°", move(R, S1, speed=60, min_t=0.3)), ("ring hold", hold(S1, 2)),
             ("fast pan step -40°", move(S1, S2, speed=60, min_t=0.3)), ("ring hold", hold(S2, 2)),
             ("back to ready", move(S2, R)), ("final hold", hold(R, 3)), ("back to start", move(R, start, speed=20)),
             ("settle", hold(start, 1))]
    return segs


def progress(i, n, t0, label):
    f = i / n
    el = time.time() - t0
    eta = el / f - el if f > 0 else 0
    line = (f"[{'#' * int(f * 30):<30}] {100 * f:5.1f}%  {i}/{n} ticks  elapsed {el:4.0f}s  ETA {eta:4.0f}s | {label}")
    PROG.write_text(line + "\n")
    print("\r" + line[:110], end="", flush=True)


def main():
    import argparse
    from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--far", default=None, metavar="SQUARE", help="run the test around a far-reach pose above this square (e.g. f1)")
    ap.add_argument("--summarize", default=None, metavar="NPZ", help="only print the numbers of an earlier run")
    args = ap.parse_args()
    if args.summarize:
        summarize(Path(args.summarize)); return
    far = via = None
    if args.far:
        far, via, Pl = far_pose(args.far)
        print(f"far pose above {args.far}: {np.round(far, 1).tolist()} (via {np.round(via, 1).tolist()})")
    # gains live in servo RAM: the next normal connect (recorder, teleop) writes LeRobot's defaults back
    robot = SOFollower(SO101FollowerConfig(port=so101_config.FOLLOWER_PORT, id=so101_config.FOLLOWER_ID,
                                           position_p_coefficient=args.p, position_d_coefficient=args.d,
                                           position_i_coefficient=args.i))
    bus = robot.bus
    bus.connect()
    here = bus.sync_read("Present_Position")
    bus.sync_write("Goal_Position", here)                  # torque will come on holding this exact pose
    robot.configure()                                      # PID etc.; re-enables torque
    gains = {j: {k: int(bus.read(k, j)) for k in ("P_Coefficient", "I_Coefficient", "D_Coefficient")} for j in J}
    start = np.array([here[j] for j in J], float)
    print("start pose:", np.round(start, 1))
    print("gains:", gains["shoulder_pan"], "(same on all joints)" if len({str(v) for v in gains.values()}) == 1 else gains)

    segs = plan_far(start, far, via) if args.far else plan(start)
    steps = [(name, q) for name, qs in segs for q in qs]
    if args.far:                                                 # the whole path must stay well above the board
        zs = np.array([Pl.fk(q)[0][2] - Pl.z0 for _, q in steps[::3]])
        print(f"gripper site above the board along the path: min {100 * zs.min():.1f} cm, max {100 * zs.max():.1f} cm")
        if zs.min() < 0.05:
            bus.disconnect(disable_torque=False); sys.exit("the path would come within 5 cm of the board: not running")
    n = len(steps)
    print(f"{len(segs)} segments, {n} ticks ({n / HZ:.0f} s)")
    T, GOAL, POS, LOAD, VOLT, LBL = [], [], [], [], [], []
    t0, lag_since, aborted = time.time(), None, ""
    try:
        for i, (name, q) in enumerate(steps):
            tick = time.perf_counter()
            bus.sync_write("Goal_Position", dict(zip(J, q)))
            p = bus.sync_read("Present_Position")
            pv = np.array([p[j] for j in J])
            if i % 5 == 0:
                ld = bus.sync_read("Present_Load", normalize=False)
                LOAD.append([time.time() - t0] + [int(ld[j]) for j in J])
                if i % 50 == 0:
                    VOLT.append(int(bus.read("Present_Voltage", "shoulder_lift", normalize=False)) / 10)
            T.append(time.time() - t0); GOAL.append(q); POS.append(pv); LBL.append(name)
            err = np.abs(pv - q)[:5]
            if err.max() > 15:
                lag_since = lag_since or time.time()
                if time.time() - lag_since > 0.4:
                    aborted = f"stopped: {J[int(err.argmax())]} lagged {err.max():.0f} deg during '{name}'"
                    bus.sync_write("Goal_Position", dict(zip(J, pv)))
                    break
            else:
                lag_since = None
            # buzz guard: commanded pose constant for the last 0.6 s but the joint keeps moving -> oscillation
            k = int(0.6 * HZ)
            if len(GOAL) > k and np.ptp(np.array(GOAL[-k:])[:, :5], axis=0).max() < 1e-6:
                swing = np.ptp(np.array(POS[-k:])[:, :5], axis=0)
                if swing.max() > 3.0:
                    aborted = f"stopped: {J[int(swing.argmax())]} oscillating ({swing.max():.1f} deg p-p) during '{name}'"
                    bus.sync_write("Goal_Position", dict(zip(J, q)))
                    break
            if i % 10 == 0:
                progress(i, n, t0, name)
            time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        aborted = "Ctrl-C: holding the current pose"
        p = bus.sync_read("Present_Position")
        bus.sync_write("Goal_Position", p)
    OUT.mkdir(exist_ok=True)
    f = OUT / f"shake_{time.strftime('%Y%m%d_%H%M%S')}_P{args.p}{'_I' + str(args.i) if args.i else ''}{'_far_' + args.far if args.far else ''}.npz"
    np.savez(f, t=np.array(T), goal=np.array(GOAL), pos=np.array(POS), label=np.array(LBL), load=np.array(LOAD),
             volt=np.array(VOLT), joints=np.array(J), gains=json.dumps(gains), hz=HZ)
    if aborted:
        print("\n" + aborted + "  (torque left ON, holding)")
        PROG.write_text(f"STOPPED | {aborted}\n")
        bus.disconnect(disable_torque=False)
    else:
        progress(n, n, t0, "DONE, back at the start pose, torque off")
        print()
        bus.disconnect(disable_torque=True)
    print("log:", f)
    summarize(f)


if __name__ == "__main__":
    main()
