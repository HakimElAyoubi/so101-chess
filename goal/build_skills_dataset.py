"""Plan step 1 (train/PLAN_TO_95.md): LeHak0/so101_chess_skills = the GRASP and the PLACE of every demo, cut out of the
202 full-move episodes, so one square-agnostic policy learns both skills from ~180 examples each (instead of 1-2 per move).

Clip windows, measured from the demos in the twin (2026-10-25):
- grasp: first close - 2.5 s (gripper ~6 cm above the piece, nearly straight above it, target clear of the jaw in 86 %)
  -> carry close + 1.5 s (lifted ~4 cm). Failed attempts before the carry grasp stay in the clip: they show how to retry.
- place: carry release - 2.0 s (~4 cm above the square) -> release + 1.5 s (lifted clear).
Task text: "pick up the black bishop" / "put the black bishop down on the ring". Grasp clips show only the green dot,
place clips only the magenta ring (goal/targets.py, as in the marker dataset). The 130-d move encoding
(observation.environment_state) is dropped: the skills must not know the squares.
Clips whose grasp / release lands > 2.5 cm from the projected target are skipped (the marker would be off the piece).
10 % of the source episodes (both their clips) are held out for validation: goal/skill_clips.json "val_episodes"
-> train with --dataset.exclude_episodes=[...].

    python goal/build_skills_dataset.py plan     # clip list + summary only (seconds)
    python goal/build_skills_dataset.py build    # plan + build the dataset (~15 min). Progress: data/progress.txt
"""
import json, re, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from board_calib import events
from targets import Projector, session_of

SRC = Path.home() / ".cache/huggingface/lerobot/LeHak0/so101_chess_moves"
REPO = "LeHak0/so101_chess_skills"
DST = Path.home() / ".cache/huggingface/lerobot" / REPO
PLAN = Path(__file__).parent / "skill_clips.json"
PROG = Path(__file__).resolve().parent.parent / "data/progress.txt"
CAM = "observation.images.wrist"
FPS = 30
GRASP_BEFORE, GRASP_AFTER = 2.5, 1.5        # s before the first close / after the carry close
PLACE_BEFORE, PLACE_AFTER = 2.0, 1.5        # s before / after the carry release
MAX_MISS = 0.025                            # m: skip a clip whose grasp / release is this far from the marker
DROP_MAX_MISS = 0.08                        # m: drop-zone releases, from the session's median drop point
RELEASE_FRAC, RELEASE_MIN = 0.15, 3.0       # release = the command reopens by 15 % of the travel, at least 3 %
CLOSE_FRAC, CLOSE_MIN = 0.7, 3.0            # grasp = it falls below 70 % of the opening, by at least 3 %
VAL_FRAC = 0.10
PIECE = re.compile(r"(black|white) (pawn|knight|bishop|rook|queen|king)")


def plan():
    P = Projector(); s = P.s
    ep = pd.concat([pd.read_parquet(f) for f in sorted(SRC.glob("meta/episodes/*/*.parquet"))]).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in sorted(SRC.glob("data/*/*.parquet"))])
    clips, skipped = [], {"no grasp/release": [], "unknown task": [], "grasp off target": [], "place off target": []}
    for e, g in data.groupby("episode_index"):
        e = int(e); task = ep.loc[e, "tasks"][0]
        try:
            f, t = P.target_xyz(task, session_of(e))
        except ValueError:
            skipped["unknown task"].append(e); continue
        q = np.stack(g["observation.state"].values); a = np.stack(g["action"].values)
        ev = events(a, RELEASE_FRAC, RELEASE_MIN, CLOSE_FRAC, CLOSE_MIN)
        if not ev:
            skipped["no grasp/release"].append(e); continue
        xy = []
        for k in range(len(q)):
            s.set_joints(q[k]); xy.append(s.gripper_pos()[:2])
        xy = np.array(xy)
        carry = int(np.argmax([np.linalg.norm(xy[ko] - xy[kc]) for kc, ko in ev]))   # the pair that moved the piece
        kc, ko = ev[carry]
        color, piece = PIECE.search(task).groups()
        start = int(g["index"].min())
        drop = "drop zone" in task
        dg, dp = float(np.linalg.norm(xy[kc] - f[:2])), float(np.linalg.norm(xy[ko] - t[:2]))
        for kind, k0, k1, miss, txt in (
                ("grasp", ev[0][0] - int(GRASP_BEFORE * FPS), kc + int(GRASP_AFTER * FPS), dg, f"pick up the {color} {piece}"),
                ("place", ko - int(PLACE_BEFORE * FPS), ko + int(PLACE_AFTER * FPS), dp, f"put the {color} {piece} down on the ring")):
            drop_place = drop and kind == "place"
            if miss > (DROP_MAX_MISS if drop_place else MAX_MISS):
                skipped[f"{kind} off target"].append(e); continue
            c = dict(src_ep=e, kind=kind, k0=max(0, k0), k1=min(len(q), k1), src_index=start, task=txt, src_task=task,
                     session=session_of(e), miss_cm=round(100 * miss, 1), retries=carry)
            if drop_place:              # the drop zone is an area: the ring goes where the piece was really released
                c["to_xyz"] = [float(xy[ko][0]), float(xy[ko][1]), float(s.cfg.board_thickness)]
            clips.append(c)
    src_eps = sorted({c["src_ep"] for c in clips})
    val = sorted(int(x) for x in np.random.default_rng(0).choice(src_eps, round(VAL_FRAC * len(src_eps)), replace=False))
    for i, c in enumerate(clips):
        c["clip"] = i; c["split"] = "val" if c["src_ep"] in val else "train"
    out = dict(repo=REPO, windows_s=dict(grasp=[GRASP_BEFORE, GRASP_AFTER], place=[PLACE_BEFORE, PLACE_AFTER]),
               max_miss_m=MAX_MISS, val_src_episodes=val, val_episodes=[c["clip"] for c in clips if c["split"] == "val"],
               skipped={k: v for k, v in skipped.items()}, clips=clips)
    PLAN.write_text(json.dumps(out, indent=1))
    n = {k: sum(c["kind"] == k for c in clips) for k in ("grasp", "place")}
    secs = {k: np.array([(c["k1"] - c["k0"]) / FPS for c in clips if c["kind"] == k]) for k in n}
    print(f"{len(clips)} clips from {len(src_eps)} episodes: {n['grasp']} grasps (median {np.median(secs['grasp']):.1f} s), "
          f"{n['place']} places (median {np.median(secs['place']):.1f} s), {int(sum(v.sum() for v in secs.values()) * FPS)} frames")
    print("skipped:", {k: len(v) for k, v in skipped.items()}, "| grasp clips with failed attempts:",
          sum(c["kind"] == "grasp" and c["retries"] > 0 for c in clips))
    print(f"validation: {len(val)} source episodes = {len(out['val_episodes'])} clips")
    return out


def build(p):
    import torch
    from lerobot.datasets import LeRobotDataset
    src = LeRobotDataset("LeHak0/so101_chess_moves", root=SRC)
    if DST.exists():
        shutil.rmtree(DST)
    drop = ("timestamp", "frame_index", "episode_index", "index", "task_index", "observation.environment_state")
    feats = {k: v for k, v in src.meta.features.items() if k not in drop}
    dst = LeRobotDataset.create(REPO, src.fps, features=feats, root=DST, robot_type=src.meta.robot_type, use_videos=True,
                                streaming_encoding=True, encoder_threads=4, image_writer_threads=4)
    P = Projector()
    idx, owner = [], []
    for c in p["clips"]:
        idx += [c["src_index"] + k for k in range(c["k0"], c["k1"])]; owner += [c["clip"]] * (c["k1"] - c["k0"])
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(src, idx), batch_size=None, shuffle=False, num_workers=4,
                                         prefetch_factor=8)
    n, cur, t0 = len(idx), None, time.time()
    for i, (f, ci) in enumerate(zip(loader, owner)):
        if cur is not None and ci != cur:
            dst.save_episode()
        cur, c = ci, p["clips"][ci]
        img = (f[CAM].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        P.draw_markers(img, f["observation.state"].numpy(), c["src_task"], c["session"],
                       which=("from",) if c["kind"] == "grasp" else ("to",), to_xyz=c.get("to_xyz"))
        frame = {k: f[k].numpy() for k in feats if k != CAM}
        frame[CAM] = img
        frame["task"] = c["task"]
        dst.add_frame(frame)
        if i % 300 == 0:
            el = time.time() - t0; eta = el / max(i, 1) * (n - i)
            PROG.write_text(f"[{'#' * int(30 * i / n):<30}] {100 * i / n:5.1f}%  {i}/{n} frames  clip {ci}/{len(p['clips'])}  "
                            f"elapsed {el / 60:4.1f} min  ETA {eta / 60:4.1f} min | skills dataset\n")
    dst.save_episode()
    dst.finalize()
    PROG.write_text(f"[{'#' * 30}] 100.0%  {n}/{n} frames  done in {(time.time() - t0) / 60:.1f} min | skills dataset\n")
    print("episodes:", dst.num_episodes, "frames:", dst.num_frames)


if __name__ == "__main__":
    p = plan()
    if sys.argv[1:] == ["build"]:
        build(p)
