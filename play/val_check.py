"""Plan step 3: pick each skill model's checkpoint by its error on the 40 HELD-OUT clips (goal/skill_clips.json
"val_episodes", never trained on), not by training loss.

Open loop, as play/offline_check.py: every 0.5 s of a clip the model sees the recorded frame (its marker is already drawn in
the dataset) and the joint state, predicts a chunk, and the first 50 actions are compared with what the human did next
(cut at the clip end). Reported per checkpoint: mean |model - human| over the 5 arm joints (deg) and the gripper (%), for
grasp and place clips, plus the same on 20 training clips to show the generalisation gap.

    python play/val_check.py --repo LeHak0/smolvla_chess_skills          (all checkpoints; progress: data/progress_val.txt)
    python play/val_check.py --repo LeHak0/smolvla_chess_grasp_b --plan goal/grasp_clips.json --checkpoints 005000 010000
    -> play/results/val_<repo name>.json
The clip plan (--plan) gives the dataset and the split: goal/skill_clips.json (grasp + place clips, groups "kind") or
goal/grasp_clips.json (the grasp-only skill of November: game clips + drill episodes, groups "source" = moves / drill).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from policy_io import CHECKPOINTS, Policy                    # noqa: E402

PLAN, DATA = None, None                                       # set in main() from --plan
PROG = ROOT / "data" / __import__("os").environ.get("VAL_PROGRESS", "progress_val.txt")   # own bar on data/progress.html
CAM = "observation.images.wrist"
EVERY, HORIZON, N_TRAIN = 15, 50, 20


def frames(ds, clips):
    """[(clip, global frame index, state, image, future actions [<=HORIZON, 6])] every EVERY frames of each clip."""
    df = pd.concat([pd.read_parquet(f) for f in sorted(DATA.glob("data/*/*.parquet"))]).sort_values("index")
    out = []
    for c in clips:
        e = df[df.episode_index == c["clip"]]
        act, idx = np.stack(e["action"].values), e["index"].values
        for k in range(0, len(e) - 10, EVERY):
            f = ds[int(idx[k])]
            img = (f[CAM].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            out.append((c, int(idx[k]), f["observation.state"].numpy(), img, act[k:k + HORIZON]))
    return out


def score(policy, samples):
    rows = []
    noise = policy.fixed_noise(0)
    for c, _, state, img, fut in samples:
        policy.reset()
        pred = policy.chunk(state, img, c["task"], noise=noise)[:len(fut)]
        err = np.abs(pred - fut).mean(0)
        rows.append({"kind": c.get("kind") or c.get("source"), "arm": float(err[:5].mean()), "grip": float(err[5])})
    df = pd.DataFrame(rows)
    return {k: {"arm_deg": round(float(g.arm.mean()), 2), "grip_pct": round(float(g.grip.mean()), 2), "n": len(g)}
            for k, g in [("all", df)] + list(df.groupby("kind"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoints", nargs="+", default=[c for c in CHECKPOINTS if c != "final"])
    ap.add_argument("--plan", default="goal/skill_clips.json", help="clip plan: goal/skill_clips.json or goal/grasp_clips.json")
    ap.add_argument("--quick", action="store_true", help="3 held-out + 2 training clips (plumbing test)")
    args = ap.parse_args()
    global PLAN, DATA
    PLAN = json.loads((ROOT / args.plan).read_text())
    DATA = Path.home() / ".cache/huggingface/lerobot" / PLAN["repo"]
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset(PLAN["repo"], root=DATA)
    val = [c for c in PLAN["clips"] if c["split"] == "val"]
    train = [c for c in PLAN["clips"] if c["split"] == "train"][:: max(1, len(PLAN["clips"]) // N_TRAIN)][:N_TRAIN]
    if args.quick:
        val, train = val[:3], train[:2]
    sv, st = frames(ds, val), frames(ds, train)
    print(f"{len(sv)} held-out samples from {len(val)} clips, {len(st)} training samples from {len(train)} clips", flush=True)
    out, t0 = {"repo": args.repo, "plan": args.plan, "dataset": PLAN["repo"], "checkpoints": {}}, time.time()
    n = len(args.checkpoints)
    for i, ck in enumerate(args.checkpoints):
        el = time.time() - t0; eta = el / max(i, 1) * (n - i) if i else 0
        PROG.write_text(f"[{'#' * int(30 * i / n):<30}] {100 * i / n:5.1f}%  checkpoint {i + 1}/{n} ({ck})  elapsed {el / 60:4.1f} min  "
                        f"ETA {eta / 60:4.1f} min | val check {args.repo}\n")
        p = Policy(ck, "skills", repo=args.repo); p.goal = None          # the dataset frames already carry their marker
        out["checkpoints"][ck] = {"val": score(p, sv), "train": score(p, st)}
        v, t = out["checkpoints"][ck]["val"], out["checkpoints"][ck]["train"]
        groups = " / ".join(f"{k} {v[k]['arm_deg']:.2f}" for k in v if k != "all")
        print(f"{ck}: held-out arm {v['all']['arm_deg']:.2f} deg, gripper {v['all']['grip_pct']:.2f} % "
              f"({groups} deg) | training clips arm {t['all']['arm_deg']:.2f} deg", flush=True)
        del p
    best = min(out["checkpoints"], key=lambda k: out["checkpoints"][k]["val"]["all"]["arm_deg"])
    out["best"] = best
    path = HERE / "results" / f"val_{args.repo.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=1))
    PROG.write_text(f"[{'#' * 30}] 100.0%  {n}/{n} checkpoints  done in {(time.time() - t0) / 60:.1f} min | val check {args.repo}\n")
    print(f"best on the held-out clips: {best} -> {path}")


if __name__ == "__main__":
    main()
