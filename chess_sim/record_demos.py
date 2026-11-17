"""Record scripted-expert demonstrations of SO101ChessMove-v0 into a LeRobotDataset.

    python chess_sim/record_demos.py --episodes 10 --out data/demos_10 \
        [--repo-id user/so101_chess_sim] [--randomise] [--seed 0] [--images] [--keep-failed] [--fen FEN --move e7e5]
        [--progress-path FILE]

Dataset (LeRobot v3 format, fps = CONTROL_HZ = 30):
    observation.images.wrist        video (or --images: PNG) H x W x 3 uint8, the wrist camera
    observation.state               float32[6]    [5 joints deg, gripper %]
    observation.environment_state   float32[128]  the move to execute, one-hot from-square [0:64] + to-square [64:128]
                                                  (chess_env.encode_move); this is the move-conditioning channel that
                                                  ACT / diffusion consume (ACT's env_state_feature) -- the task string
                                                  below is only seen by language-conditioned policies (smolvla, pi0)
    action                          float32[6]    the commanded targets in the same convention
    task                            'move the piece from e7 to e5' (string per frame)
plus a sidecar <out>/episodes.json (fen, move, success, steps, seed, failure per episode).  Failed episodes
(expert abort, piece misplaced) are discarded and re-drawn unless --keep-failed.  Progress bar: chess_sim/data/
progress.txt (viewer: chess_sim/progress_view.command, double-click in the Finder).  After recording the dataset
is loaded back and its shapes / fps / episode count are printed (and checked).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_sim import config as C
from chess_sim.progress import ProgressBar
from chess_sim.chess_env import SO101ChessMoveEnv, MOVE_ENCODING_DIM, MOVE_FEATURE_NAMES

STATE_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
VIEWER_SCRIPT = """#!/bin/zsh
# Live view of chess_sim/data/progress.txt: refreshes every second, closes 5 s after the job ends.
F="$(dirname "$0")/data/progress.txt"
while true; do
  clear; echo "chess_sim job progress  ($(date +%H:%M:%S))"; echo; cat "$F" 2>/dev/null
  grep -q -E "DONE|FAILED" "$F" 2>/dev/null && { sleep 5; break; }
  sleep 1
done
"""


def ensure_viewer() -> str:
    """chess_sim/progress_view.command (self-refreshing Terminal viewer of the progress bar)."""
    path = os.path.join(C.PKG_DIR, "progress_view.command")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(VIEWER_SCRIPT)
        os.chmod(path, 0o755)
    return path


def dataset_features(h: int, w: int, use_videos: bool) -> dict:
    return {
        "observation.images.wrist": {"dtype": "video" if use_videos else "image", "shape": (h, w, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
        "observation.environment_state": {"dtype": "float32", "shape": (MOVE_ENCODING_DIM,), "names": MOVE_FEATURE_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
    }


def record(episodes: int, out: str, repo_id: str = "local/so101_chess_sim", randomise: bool = False, seed: int = 0,
           use_videos: bool = True, keep_failed: bool = False, fen: Optional[str] = None, move: Optional[str] = None,
           max_steps: int = 1500, max_tries: Optional[int] = None, env: Optional[SO101ChessMoveEnv] = None,
           progress_path: str = C.PROGRESS_TXT, verbose: bool = True, overwrite: bool = False) -> dict:
    """Run the expert in the env and write a LeRobotDataset to `out`.  Returns a summary dict."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out_path = Path(out)
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)
        elif any(out_path.iterdir()):
            raise FileExistsError(f"{out} exists and is not empty (use --overwrite)")
        else:
            out_path.rmdir()
    env = env or SO101ChessMoveEnv(randomise=randomise, max_steps=max_steps)
    env.randomise = randomise
    h, w = env.cfg.render_h, env.cfg.render_w
    ds = LeRobotDataset.create(repo_id, fps=C.CONTROL_HZ, features=dataset_features(h, w, use_videos), root=out_path,
                               robot_type="so101_sim", use_videos=use_videos)
    ensure_viewer()
    pb = ProgressBar(episodes, path=progress_path)
    pb.update(0, "record_demos: starting", force=True)
    max_tries = max_tries or 3 * episodes + 3
    records = []
    n_saved = n_failed = 0
    t0 = time.time()
    rng = np.random.default_rng(seed)
    try:
        for attempt in range(max_tries):
            if n_saved >= episodes:
                break
            ep_seed = int(rng.integers(0, 2 ** 31 - 1))
            options = {}
            if fen is not None:
                options["fen"] = fen
            if move is not None:
                options["move"] = move
            obs, info = env.reset(seed=ep_seed, options=options)
            task = info["task"]
            frames = 0
            terminated = truncated = False
            tail = 0
            t_ep = time.time()
            while not (terminated or truncated):
                action = env.expert_action(obs)
                frame = {"observation.images.wrist": np.ascontiguousarray(obs["pixels"]["wrist"]),
                         "observation.state": np.asarray(obs["agent_pos"], np.float32),
                         "observation.environment_state": np.asarray(obs["environment_state"], np.float32),
                         "action": np.asarray(action, np.float32), "task": task}
                ds.add_frame(frame)
                obs, r, terminated, truncated, info = env.step(action)
                frames += 1
                if frames % 15 == 0:
                    pb.update(n_saved, f"record_demos: episode {n_saved + 1}/{episodes} ({info['move']}) step {frames} "
                                       f"{info['phase']} | {info['expert']} | failed {n_failed}")
                if env.expert_done or env.expert_abort:
                    tail += 1
                    if tail > 15:
                        break
            success = bool(info["success"])
            rec = dict(seed=ep_seed, fen=info["fen"], move=info["move"], task=task, success=success, failure=info["failure"],
                       expert_abort=env.expert_abort, steps=frames, piece_err_mm=info["piece_err_mm"], wall_s=time.time() - t_ep,
                       randomise=randomise)
            if success or keep_failed:
                ds.save_episode()
                rec["episode_index"] = n_saved
                n_saved += 1
                records.append(rec)
            else:
                ds.clear_episode_buffer()
                n_failed += 1
                rec["episode_index"] = None
                records.append(rec)
            if verbose:
                print(f"  episode {attempt + 1}: {info['move']} {'OK' if success else 'FAIL ' + str(info['failure'] or env.expert_abort)} "
                      f"{frames} frames {rec['wall_s']:.1f}s  err {({k: round(v, 1) for k, v in info['piece_err_mm'].items()})}"
                      f"{'' if success or keep_failed else '  (discarded)'}")
                sys.stdout.flush()
            pb.update(n_saved, f"record_demos: {n_saved}/{episodes} saved, {n_failed} failed", force=True)
    finally:
        ds.finalize()
    with open(out_path / "episodes.json", "w") as f:
        json.dump(dict(repo_id=repo_id, fps=C.CONTROL_HZ, randomise=randomise, seed=seed, episodes=records), f, indent=1)
    summary = dict(out=str(out_path), repo_id=repo_id, saved=n_saved, failed=n_failed, requested=episodes,
                   seconds=time.time() - t0, use_videos=use_videos)
    if n_saved < episodes:
        pb.fail(f"record_demos: only {n_saved}/{episodes} episodes saved ({n_failed} failed)")
    else:
        pb.done(f"record_demos: {n_saved}/{episodes} episodes saved in {summary['seconds']:.0f}s ({n_failed} failed)")
    return summary


def _decoded_move(item) -> str:
    from chess_sim.chess_env import decode_move
    v = item["observation.environment_state"]
    return decode_move(v.numpy() if hasattr(v, "numpy") else v).uci()


def verify(out: str, repo_id: str = "local/so101_chess_sim", verbose: bool = True) -> dict:
    """Load the dataset back; returns dict(num_episodes, num_frames, fps, shapes) and prints it."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=out)
    item = ds[0]
    shapes = {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")}
    out_d = dict(num_episodes=ds.num_episodes, num_frames=ds.num_frames, fps=ds.fps, shapes=shapes, task=item.get("task"),
                 features=list(ds.meta.features.keys()),
                 move=_decoded_move(item) if "observation.environment_state" in item else None)
    if verbose:
        print(f"loaded back: {ds.num_episodes} episodes, {ds.num_frames} frames, fps {ds.fps}")
        for k, v in shapes.items():
            print(f"  {k:28s} {v}  {item[k].dtype}")
        print(f"  task: {item.get('task')!r}; move decoded from observation.environment_state: {out_d['move']}")
    return out_d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", required=True, help="dataset directory (created; must not exist unless --overwrite)")
    ap.add_argument("--repo-id", default="local/so101_chess_sim")
    ap.add_argument("--randomise", action="store_true", help="domain randomisation at every reset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", action="store_true", help="store PNG images instead of AV1 video")
    ap.add_argument("--keep-failed", action="store_true", help="also save episodes whose move failed")
    ap.add_argument("--fen", default=None, help="fixed position (default: random mid-game positions)")
    ap.add_argument("--move", default=None, help="fixed move (UCI or SAN) for --fen")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress-path", default=C.PROGRESS_TXT,
                    help="progress bar file (default chess_sim/data/progress.txt; use another file when a game / test runs concurrently)")
    a = ap.parse_args(argv)
    summary = record(a.episodes, a.out, a.repo_id, a.randomise, a.seed, not a.images, a.keep_failed, a.fen, a.move,
                     a.max_steps, overwrite=a.overwrite, progress_path=a.progress_path)
    print("summary:", summary)
    if summary["saved"] > 0:
        verify(a.out, a.repo_id)
    return 0 if summary["saved"] == a.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
