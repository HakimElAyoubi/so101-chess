"""Grasp-only training data: LeHak0/so101_chess_grasp_skill = the grasp clips of the 202 game demos (as in
LeHak0/so101_chess_skills) + every episode of the grasp drill (recording/grasp_recorder.py, LeHak0/so101_chess_grasp).

A drill episode is a whole clip: it starts at the staged player's grasp hover (exactly the state the skill starts from
when playing) and ends after the lift. The green dot is drawn at the square's centre on the board pose the episode was
recorded with (recording/grasp_meta.jsonl), while the piece stood off-centre on purpose: the model learns to grasp the
piece, not the dot. Task text "pick up the black rook"; the 130-d move encoding is dropped (as for the skills).
Held out for validation: the demo clips held out before (goal/skill_clips.json) + 10 % of the drill episodes -> the clip
list goes to goal/grasp_clips.json, the held-out episode list to train/grasp_val.txt (read by train/launch_grasp.sh).

    python goal/build_grasp_dataset.py plan            # clip list + summary only
    python goal/build_grasp_dataset.py build           # + build (progress: data/progress.txt), then push it:
    python goal/build_grasp_dataset.py push
    (--drill-root / --drill-meta / --out-root / --limit N: a dry run on other folders, N clips per source)
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from targets import Projector

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache/huggingface/lerobot"
MOVES = CACHE / "LeHak0/so101_chess_moves"
REPO = "LeHak0/so101_chess_grasp_skill"
SKILL_PLAN = Path(__file__).parent / "skill_clips.json"
PLAN = Path(__file__).parent / "grasp_clips.json"
VAL_OUT = ROOT / "train/grasp_val.txt"
PROG = ROOT / "data/progress.txt"
CAM = "observation.images.wrist"
VAL_FRAC = 0.10


def plan(a):
    clips = []
    old = [c for c in json.loads(SKILL_PLAN.read_text())["clips"] if c["kind"] == "grasp"][:a.limit or None]
    for c in old:
        clips.append(dict(source="moves", src_index=c["src_index"], k0=c["k0"], k1=c["k1"], task=c["task"], src_task=c["src_task"],
                          session=c["session"], split=c["split"], src=f"demo episode {c['src_ep']}"))
    meta = [json.loads(l) for l in open(a.drill_meta)] if Path(a.drill_meta).exists() else []
    if meta:
        data = pd.concat([pd.read_parquet(f) for f in sorted(Path(a.drill_root).glob("data/*/*.parquet"))])
        span = data.groupby("episode_index")["index"].agg(["min", "count"])
        last = {}
        for m in meta:                                  # the newest recording of each drill item (a re-done one replaces it)
            if m["episode_index"] in span.index:
                last[m["drill"]] = m
        drills = sorted(last)
        val = set(int(x) for x in np.random.default_rng(0).choice(drills, round(VAL_FRAC * len(drills)), replace=False)) if drills else set()
        for d in drills[:a.limit or None]:
            m = last[d]; s = span.loc[m["episode_index"]]
            clips.append(dict(source="drill", src_index=int(s["min"]), k0=0, k1=int(s["count"]), task=m["task"],
                              src_task=f"move the {m['colour']} {m['piece']} from {m['square']} to {m['square']}",
                              session=m["board_session"], split="val" if d in val else "train",
                              src=f"drill {d} ({m['square']}, {m['spot']})"))
    for i, c in enumerate(clips):
        c["clip"] = i
    val_eps = [c["clip"] for c in clips if c["split"] == "val"]
    out = dict(repo=REPO, val_episodes=val_eps, clips=clips)
    if not a.limit:
        PLAN.write_text(json.dumps(out, indent=1)); VAL_OUT.write_text(json.dumps(val_eps) + "\n")
    n = {s: sum(c["source"] == s for c in clips) for s in ("moves", "drill")}
    print(f"{len(clips)} grasp clips: {n['moves']} from the game demos, {n['drill']} from the drill, "
          f"{sum(c['k1'] - c['k0'] for c in clips)} frames; {len(val_eps)} held out for validation")
    return out


def build(p, a):
    import torch
    from lerobot.datasets import LeRobotDataset
    srcs = {"moves": LeRobotDataset("LeHak0/so101_chess_moves", root=MOVES)}
    if any(c["source"] == "drill" for c in p["clips"]):
        srcs["drill"] = LeRobotDataset("LeHak0/so101_chess_grasp", root=a.drill_root)
    out = Path(a.out_root)
    if out.exists():
        shutil.rmtree(out)
    m0 = srcs["moves"].meta
    drop = ("timestamp", "frame_index", "episode_index", "index", "task_index", "observation.environment_state")
    feats = {k: v for k, v in m0.features.items() if k not in drop}
    dst = LeRobotDataset.create(REPO, m0.fps, features=feats, root=out, robot_type=m0.robot_type, use_videos=True,
                                streaming_encoding=True, encoder_threads=4, image_writer_threads=4)
    P = Projector()
    n, done, t0 = sum(c["k1"] - c["k0"] for c in p["clips"]), 0, time.time()
    for source, ds in srcs.items():
        cl = [c for c in p["clips"] if c["source"] == source]
        idx, owner = [], []
        for c in cl:
            idx += [c["src_index"] + k for k in range(c["k0"], c["k1"])]; owner += [c["clip"]] * (c["k1"] - c["k0"])
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=None, shuffle=False, num_workers=4,
                                             prefetch_factor=8)
        cur = None
        for f, ci in zip(loader, owner):
            if cur is not None and ci != cur:
                dst.save_episode()
            cur, c = ci, p["clips"][ci]
            img = (f[CAM].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
            P.draw_markers(img, f["observation.state"].numpy(), c["src_task"], c["session"], which=("from",))
            frame = {k: f[k].numpy() for k in feats if k != CAM}
            frame[CAM] = img; frame["task"] = c["task"]
            dst.add_frame(frame)
            done += 1
            if done % 300 == 0:
                el = time.time() - t0
                PROG.write_text(f"[{'#' * int(30 * done / n):<30}] {100 * done / n:5.1f}%  {done}/{n} frames  clip {ci}/{len(p['clips'])}  "
                                f"elapsed {el / 60:4.1f} min  ETA {el / done * (n - done) / 60:4.1f} min | grasp dataset\n")
        dst.save_episode()
    dst.finalize()
    PROG.write_text(f"[{'#' * 30}] 100.0%  {n}/{n} frames  done in {(time.time() - t0) / 60:.1f} min | grasp dataset\n")
    print("episodes:", dst.num_episodes, "frames:", dst.num_frames, "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["plan", "build", "push"])
    ap.add_argument("--drill-root", default=str(CACHE / "LeHak0/so101_chess_grasp"))
    ap.add_argument("--drill-meta", default=str(ROOT / "recording/grasp_meta.jsonl"))
    ap.add_argument("--out-root", default=str(CACHE / REPO))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.what == "push":
        from lerobot.datasets import LeRobotDataset
        LeRobotDataset(REPO, root=a.out_root).push_to_hub(private=True); print("uploaded", REPO)
    else:
        p = plan(a)
        if a.what == "build":
            build(p, a)
