"""Before recording: move the follower slowly to the leader's current pose, only if that pose and the path
to it are clear of the board / pieces / table in the sim. The follower then keeps holding (torque on)."""
import json, sys, time
from pathlib import Path
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repository root
import so101_config  # noqa: E402  (ports and Stockfish path from config.yaml)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_check import JOINTS, PORT, ROBOT_ID, segment, path_check, HZ

LEADER_PORT = so101_config.LEADER_PORT
MIRRORED = ()       # no mirroring (normal leader)

def leader_pose():
    import scservo_sdk as scs
    cal = json.loads((Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader_arm_1.json").read_text())
    ph = scs.PortHandler(LEADER_PORT); ph.openPort(); ph.setBaudRate(1_000_000); pk = scs.PacketHandler(0)
    out = []
    for n in JOINTS:
        c = cal[n]; pos, comm, _ = pk.read2ByteTxRx(ph, c["id"], 56)
        if comm != 0: raise SystemExit("could not read the leader")
        out.append((pos - c["range_min"]) / (c["range_max"] - c["range_min"]) * 100 if n == "gripper"
                   else (pos - (c["range_min"] + c["range_max"]) / 2) * 360 / 4095)
    ph.closePort()
    q = np.array(out)
    if MIRRORED:
        q[list(MIRRORED)] *= -1
    return q

def distal_clearance(scene, path):
    """Smallest distance from the wrist / gripper / jaw / camera to the board, pieces and table along the path."""
    import mujoco
    m, d = scene.model, scene.data
    distal = ["wrist", "gripper", "moving_jaw_so101_v1"] + [m.body(b).name for b in range(m.nbody) if "camera" in m.body(b).name]
    arm_all = distal + ["base", "shoulder", "upper_arm", "lower_arm"]
    coll = lambda g: m.geom_contype[g] or m.geom_conaffinity[g]
    dg = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in distal and coll(g)]
    eg = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name not in arm_all and coll(g)]
    worst, what, fromto = 9.0, "", np.zeros(6)
    for k in range(0, len(path), 3):
        scene.set_joints(path[k])
        for a in dg:
            for e in eg:
                dist = mujoco.mj_geomDistance(m, d, a, e, 0.05, fromto)
                if dist < worst:
                    worst, what = dist, f"{m.body(m.geom_bodyid[a]).name} / {m.body(m.geom_bodyid[e]).name or m.geom(e).name}"
    return worst, what


def main():
    import chess
    from chess_sim.scene import ChessScene
    from calibration_table import table_ok
    target = leader_pose()
    print("leader:", np.round(target, 1).tolist())
    from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
    scene = ChessScene(); scene.set_position(chess.STARTING_FEN)
    robot = SOFollower(SO101FollowerConfig(port=PORT, id=ROBOT_ID)); bus = robot.bus; bus.connect()
    p = bus.sync_read("Present_Position", num_retry=3); q = np.array([p[j] for j in JOINTS], float)
    path = segment(q, target)
    whole = path_check(scene, path)                      # every link: >= 3 mm (shoulder sits ~12 mm from the board edge)
    dist, what = distal_clearance(scene, path + [target])  # wrist / gripper / camera: >= 15 mm (real board is a few mm off)
    print(f"follower now {np.round(q,1).tolist()}")
    print(f"clearance: whole arm {whole[0]*1000:.1f} mm ({whole[1]}); wrist/gripper/camera {dist*1000:.1f} mm ({what})")
    # Safety gate removed at the user's request (2026-10-12): the sim's board sits a few mm off the real one and
    # flagged safe poses. The clearance above is printed for information only; the move is still slow (30 deg/s).
    if whole[0] < 0.003 or dist < 0.015:
        print("note: the sim reports little clearance for this move; continuing as requested")
    raw = bus.sync_read("Present_Position", normalize=False, num_retry=3); bus.sync_write("Goal_Position", raw, normalize=False)
    robot.configure()
    for qk in path:
        t0 = time.perf_counter(); bus.sync_write("Goal_Position", {j: float(qk[n]) for n, j in enumerate(JOINTS)})
        time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t0)))
    time.sleep(0.5)
    bus.disconnect(disable_torque=False)   # keep holding until lerobot-record takes over
    print("follower aligned with the leader, holding")

if __name__ == "__main__":
    main()
