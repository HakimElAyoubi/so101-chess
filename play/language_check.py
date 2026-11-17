"""Does the policy listen to the task text? Same observation, different task strings -> how different are the predicted
50-step action chunks? Compared with how different the chunk is between two *noise samples* of the same task
(flow matching is stochastic), which is the floor."""
import json, sys, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from policy_io import Policy
from lerobot.datasets import LeRobotDataset
from pathlib import Path
R = Path.home() / ".cache/huggingface/lerobot/LeHak0/so101_chess_moves"
EPS = [120, 150, 170, 185, 199]
ds = LeRobotDataset("LeHak0/so101_chess_moves", root=R, episodes=EPS)
pol = Policy(sys.argv[1] if len(sys.argv) > 1 else "final")
idx = np.array(ds.hf_dataset["episode_index"])
OTHER = ["move the black pawn from a7 to a5", "move the black pawn from h7 to h6", "move the black knight from b8 to c6",
         "move the black queen from d8 to d2", "move the white rook on g1 to the drop zone"]
def chunk(f, task, seed):
    torch.manual_seed(seed); pol.reset()
    img = (f["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out = [pol.act(f["observation.state"].numpy(), img, task)]
    out += [pol.act(f["observation.state"].numpy(), img, task) for _ in range(49)]   # pops the rest of the queued chunk
    return np.array(out)
rows = []
for e in EPS:
    r = np.where(idx == e)[0]
    for frac in (0.05, 0.3):
        f = ds[int(r[int(frac * len(r))])]; true = f["task"]
        a0, a1 = chunk(f, true, 0), chunk(f, true, 1)
        noise = np.abs(a0 - a1)[:, :5].mean()
        diffs = [np.abs(a0 - chunk(f, t, 0))[:, :5].mean() for t in OTHER if t != true]
        human = np.stack([ds[int(r[int(frac * len(r))] + k)]["action"].numpy() for k in range(50)])
        err = np.abs(a0 - human)[:, :5].mean()
        rows.append(dict(ep=e, frac=frac, task=true, noise=noise, task_change=float(np.mean(diffs)), err_vs_human=err))
        print(f"ep {e} @{frac:.2f} '{true[:38]}': chunk change when the TASK changes {np.mean(diffs):5.2f} deg | "
              f"change from noise alone {noise:5.2f} deg | error vs human {err:5.2f} deg", flush=True)
m = lambda k: np.mean([x[k] for x in rows])
print(f"\nmean: task change {m('task_change'):.2f} deg vs noise {m('noise'):.2f} deg (ratio {m('task_change') / max(m('noise'), 1e-6):.1f}x)")
json.dump(rows, open(Path(__file__).parent / "results/language_check.json", "w"), indent=1, default=float)
