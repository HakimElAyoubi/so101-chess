"""Open-loop sanity check of a checkpoint on recorded episodes: feed the recorded observation at each frame and compare
the policy's action with what the human did (first 50-step chunk from a few start points per episode)."""
import json, sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from policy_io import Policy, JOINTS
from lerobot.datasets import LeRobotDataset
from pathlib import Path

ckpt = sys.argv[1] if len(sys.argv) > 1 else "final"
R = Path.home() / ".cache/huggingface/lerobot/LeHak0/so101_chess_moves"
meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(Path(__file__).parent.parent / "recording/episode_meta.jsonl")}
EPS = [10, 60, 120, 150, 165, 180, 195]          # old ready pose, game moves, drills
ds = LeRobotDataset("LeHak0/so101_chess_moves", root=R, episodes=EPS)
t0 = time.time(); pol = Policy(ckpt); print(f"loaded {ckpt} in {time.time() - t0:.0f} s", flush=True)
idx = np.array(ds.hf_dataset["episode_index"]); res = []
for e in EPS:
    rows = np.where(idx == e)[0]; task = None
    for start_frac in (0.0, 0.35, 0.7):
        s = rows[int(start_frac * (len(rows) - 50))]
        pol.reset(); err = []
        for k in range(50):
            f = ds[int(s + k)]
            task = f["task"]
            img = (f["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            a = pol.act(f["observation.state"].numpy(), img, task)
            err.append(np.abs(a - f["action"].numpy()))
        err = np.array(err); res.append(dict(ep=e, start=start_frac, mae=err.mean(0).tolist(), drill="drill" in meta.get(e, {})))
        print(f"ep {e:3d} @{start_frac:.2f} {'drill' if 'drill' in meta.get(e, {}) else 'game ':5s} '{task[:40]}' MAE/joint "
              + " ".join(f"{j[:5]} {m:4.1f}" for j, m in zip(JOINTS, err.mean(0))), flush=True)
m = np.array([r["mae"] for r in res]); print(f"\n{ckpt}: mean |action error| per joint over {len(res)} chunks:",
                                             dict(zip([j[:5] for j in JOINTS], np.round(m.mean(0), 2))), "| overall", round(float(m.mean()), 2))
json.dump(dict(checkpoint=ckpt, chunks=res), open(Path(__file__).parent / f"offline_{ckpt}.json", "w"), indent=1)
