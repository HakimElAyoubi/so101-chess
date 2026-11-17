"""Open the chess scene in the native MuJoCo viewer (macOS: must be run with mjpython).

    mjpython chess_sim/view_scene.py [--fen FEN] [--watch-pose | --zero-pose]
        [--randomise] [--seed N] [--move e7e5] [--board-gap X ...]

--fen        position to show (default: the start position)
--watch-pose arm at the watch pose (default); --zero-pose: the MJCF zero pose (arm straight out)
--move       execute this move with the scripted expert in the window (repeatable with --loop), then idle
--randomise  domain randomisation (piece / board / light / colour / camera jitter)
Physics runs in real time while the window is open (the arm holds its pose).  Headless check without a window:
    python chess_sim/view_scene.py --fen ... --snapshot out.png   (renders the spectator camera to a file)
    python chess_sim/view_scene.py --previews chess_sim/assets/preview   (regenerates the preview PNGs: wrist_cam,
        overhead_cam, spectator, wrist_cam_corners (board corners projected and marked), wrist_cam_randomised)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import chess
import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_sim import config as C
from chess_sim.scene import ChessScene


def write_previews(scene: ChessScene, out_dir: str, seed: int = 3) -> dict:
    """Render the preview PNGs of the current scene build into out_dir; returns {name: path}."""
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    scene.reset(seed=0, randomise=False)
    for cam in C.CAMERA_NAMES:
        path = os.path.join(out_dir, f"{cam}.png")
        Image.fromarray(scene.render(cam)).save(path)
        out[cam] = path
    img = Image.fromarray(scene.render(C.WRIST_CAM))
    draw = ImageDraw.Draw(img)
    px = scene.project(scene.board_corners(), C.WRIST_CAM)
    for u, v in px:
        draw.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(255, 0, 0), width=3)
    draw.polygon([tuple(p) for p in px[[0, 1, 3, 2]]], outline=(255, 0, 0))
    path = os.path.join(out_dir, "wrist_cam_corners.png")
    img.save(path)
    out["wrist_cam_corners"] = path
    scene.reset(seed=seed, randomise=True)
    path = os.path.join(out_dir, "wrist_cam_randomised.png")
    Image.fromarray(scene.render(C.WRIST_CAM)).save(path)
    out["wrist_cam_randomised"] = path
    scene.reset(seed=0, randomise=False)
    return out


def main(argv=None) -> int:
    from chess_sim.build_scene import add_config_args, config_from_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--watch-pose", action="store_true", help="arm at the watch pose (default)")
    g.add_argument("--zero-pose", action="store_true", help="arm at the MJCF zero pose")
    ap.add_argument("--randomise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", default=None, help="execute this move (UCI/SAN) with the expert in the window")
    ap.add_argument("--loop", action="store_true", help="with --move: reset and repeat the move while the window is open")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    ap.add_argument("--snapshot", default=None, help="headless: render the spectator camera to this PNG and exit")
    ap.add_argument("--camera", default=C.SPECTATOR_CAM, help="camera for --snapshot (wrist_cam, overhead_cam, spectator)")
    ap.add_argument("--exit-after", type=float, default=None, help="close the window after this many seconds (tests)")
    ap.add_argument("--previews", default=None, metavar="DIR", help="headless: write the preview PNGs into DIR and exit")
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    scene = ChessScene(cfg, rebuild="auto")
    board = chess.Board(a.fen)
    arm_pose = None if a.zero_pose else "watch"
    scene.reset(seed=a.seed, randomise=a.randomise, fen=board.fen(), arm_pose=arm_pose)
    w = scene.watch_pose()
    print(f"scene {scene.xml_path}: {len(scene.board_state())} pieces on the board, arm at "
          f"{'the zero pose' if a.zero_pose else 'the watch pose ' + str(np.round(w, 1).tolist())}")
    if a.previews:
        out = write_previews(scene, a.previews, seed=a.seed)
        print("wrote " + ", ".join(os.path.relpath(p) for p in out.values()))
        return 0
    if a.snapshot:
        from PIL import Image
        img = scene.render(a.camera)
        Image.fromarray(img).save(a.snapshot)
        print(f"wrote {a.snapshot} ({img.shape[1]}x{img.shape[0]}, camera {a.camera})")
        return 0
    try:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
    except RuntimeError as e:
        print(f"cannot open the MuJoCo viewer ({e}).\nOn macOS run it with mjpython:\n"
              f"  mjpython {os.path.abspath(__file__)} {' '.join(sys.argv[1:])}")
        return 2
    dt = C.CONTROL_DT / max(1e-3, a.speed)
    mv = None
    if a.move:
        try:
            mv = chess.Move.from_uci(a.move)
        except ValueError:
            mv = board.parse_san(a.move)
    gen = None
    expert = None
    if mv is not None:
        from chess_sim.expert import get_expert, PlanError
        expert = get_expert(scene)

    def start_move():
        try:
            plan = expert.plan_move(mv, board)
            print(f"executing {mv.uci()}: {len(plan)} waypoints, planned in {plan.plan_seconds:.1f}s")
            return expert.action_generator(plan)
        except PlanError as e:
            print(f"cannot plan {mv.uci()}: {e}")
            return None

    if mv is not None:
        gen = start_move()
    t_next = time.time()
    t_start = time.time()
    with viewer:
        while viewer.is_running():
            if a.exit_after is not None and time.time() - t_start > a.exit_after:
                print(f"--exit-after {a.exit_after}s: closing the viewer")
                break
            if gen is not None:
                try:
                    scene.control_step(next(gen))
                except StopIteration:
                    gen = None
                    if expert is not None:
                        print(f"move done: {expert.state.get('abort') or 'ok'}")
                    if a.loop and mv is not None:
                        scene.settle(1.0)
                        scene.reset(seed=a.seed, randomise=a.randomise, fen=board.fen(), arm_pose=arm_pose)
                        gen = start_move()
            else:
                scene.control_step()
            viewer.sync()
            t_next += dt
            now = time.time()
            if t_next > now:
                time.sleep(t_next - now)
            else:
                t_next = now
    return 0


if __name__ == "__main__":
    sys.exit(main())
