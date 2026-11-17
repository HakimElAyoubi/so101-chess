"""Slowly move the follower to its folded rest pose (sim-checked path), then release torque so it rests on its stops."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_check import JOINTS, PORT, ROBOT_ID, segment, path_check, HZ

REST = np.array([-6.1, -103.5, 97.1, 6.5, -1.5, 0.9])   # measured on this arm (resting on its stops)

def main():
    import chess
    from chess_sim.scene import ChessScene
    from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
    scene = ChessScene(); scene.set_position(chess.STARTING_FEN)
    robot = SOFollower(SO101FollowerConfig(port=PORT, id=ROBOT_ID)); bus = robot.bus; bus.connect()
    p = bus.sync_read("Present_Position", num_retry=3); q = np.array([p[j] for j in JOINTS], float)
    if np.abs(q[:5] - REST[:5]).max() < 4:
        print("already at rest"); bus.disconnect(disable_torque=True); return
    path = segment(q, REST)
    worst = path_check(scene, path)
    print(f"path to rest: {len(path)} steps, closest approach {worst[0]*1000:.1f} mm ({worst[1]})")
    if worst[0] < 0.003:
        bus.disconnect(disable_torque=False); raise SystemExit("ABORT: path to rest not clear in the sim; move the arm to rest by hand")
    raw = bus.sync_read("Present_Position", normalize=False, num_retry=3); bus.sync_write("Goal_Position", raw, normalize=False)
    robot.configure()
    for qk in path:
        t0 = time.perf_counter(); bus.sync_write("Goal_Position", {j: float(qk[n]) for n, j in enumerate(JOINTS)})
        time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t0)))
    time.sleep(1.0)
    bus.disconnect(disable_torque=True)   # at rest on its stops: safe to go limp
    print("follower at rest")

if __name__ == "__main__":
    main()
