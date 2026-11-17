"""Variant B: LeHak0/so101_chess_moves_marker = the same 202 episodes with the target drawn INTO the wrist video:
green dot on the piece to take, magenta ring where it goes (goal/targets.py, projected per frame from the joint state,
per-session board pose). State / actions / tasks unchanged. Progress: data/progress.txt"""
import shutil, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from targets import Projector, session_of
from lerobot.datasets import LeRobotDataset

SRC = Path.home() / ".cache/huggingface/lerobot/LeHak0/so101_chess_moves"
DST = Path.home() / ".cache/huggingface/lerobot/LeHak0/so101_chess_moves_marker"
PROG = Path(__file__).resolve().parent.parent / "data/progress.txt"
CAM = "observation.images.wrist"


def main():
    src = LeRobotDataset("LeHak0/so101_chess_moves", root=SRC)
    if DST.exists():
        shutil.rmtree(DST)
    feats = {k: v for k, v in src.meta.features.items() if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")}
    dst = LeRobotDataset.create("LeHak0/so101_chess_moves_marker", src.fps, features=feats, root=DST,
                                robot_type=src.meta.robot_type, use_videos=True, streaming_encoding=True,
                                encoder_threads=4, image_writer_threads=4)
    P = Projector()
    loader = torch.utils.data.DataLoader(src, batch_size=None, shuffle=False, num_workers=4, prefetch_factor=8)
    n, cur, t0 = len(src), None, time.time()
    for i, f in enumerate(loader):
        e = int(f["episode_index"])
        if cur is not None and e != cur:
            dst.save_episode()
        cur = e
        img = (f[CAM].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        P.draw_markers(img, f["observation.state"].numpy(), f["task"], session_of(e))
        frame = {k: (f[k].numpy() if hasattr(f[k], "numpy") else f[k]) for k in feats if k != CAM}
        frame[CAM] = img
        frame["task"] = f["task"]
        dst.add_frame(frame)
        if i % 300 == 0:
            el = time.time() - t0; eta = el / max(i, 1) * (n - i)
            PROG.write_text(f"[{'#' * int(30 * i / n):<30}] {100 * i / n:5.1f}%  {i}/{n} frames  episode {e}  "
                            f"elapsed {el / 60:4.1f} min  ETA {eta / 60:4.1f} min | marker dataset\n")
    dst.save_episode()
    dst.finalize()
    PROG.write_text(f"[{'#' * 30}] 100.0%  {n}/{n} frames  done in {(time.time() - t0) / 60:.1f} min | marker dataset\n")
    print("episodes:", dst.num_episodes, "frames:", dst.num_frames)


if __name__ == "__main__":
    main()
