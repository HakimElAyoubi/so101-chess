"""How well the follower tracked the leader in the grasp-drill episodes, near squares vs far ones.
Per episode (30 Hz): tracking error action - state per joint (its mean = the sag, its rms), the lag of the follower
behind the leader (cross-correlation of the joint velocities), the follower's jitter (what is left after a 5-sample
moving average: shake) against the leader's own, and stale follower readings (identical consecutive samples: missed
bus updates). Grouped by the square's rank (8-7 near the robot, 1-3 far).
    python recording/teleop_lag_check.py [dataset root]
"""
import sys, json, glob, os
import numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/huggingface/lerobot/LeHak0/so101_chess_grasp")
J = ["pan", "lift", "elbow", "wflex", "roll", "grip"]
meta = {int(json.loads(l)["episode_index"]): json.loads(l) for l in open("recording/grasp_meta.jsonl") if l.strip()}
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet")))], ignore_index=True)


def lag_ticks(a, s, maxlag=15):
    da, ds = np.diff(a), np.diff(s)
    if da.std() < 1e-6 or ds.std() < 1e-6:
        return np.nan
    da, ds = (da - da.mean()) / da.std(), (ds - ds.mean()) / ds.std()
    c = [np.mean(da[:len(da) - k] * ds[k:]) for k in range(maxlag + 1)]
    return int(np.argmax(c))


rows = []
for ep, g in df.groupby("episode_index"):
    m = meta.get(int(ep))
    if not m or m["board_session"] != "1031":
        continue
    g = g.sort_values("frame_index"); A = np.stack(g["action"].to_numpy()); S = np.stack(g["observation.state"].to_numpy())
    err = A - S
    k5 = np.ones(5) / 5
    jit_s = [np.std(S[:, j] - np.convolve(S[:, j], k5, "same")) for j in range(4)]
    jit_a = [np.std(A[:, j] - np.convolve(A[:, j], k5, "same")) for j in range(4)]
    stale = int(np.sum(np.all(np.abs(np.diff(S[:, :5], axis=0)) < 1e-6, axis=1)))
    rows.append(dict(ep=int(ep), square=m["square"], rank=int(m["square"][1]), n=len(S),
                     sag=[float(err[:, j].mean()) for j in range(4)], rms=[float(np.sqrt((err[:, j] ** 2).mean())) for j in range(4)],
                     lag=[lag_ticks(A[:, j], S[:, j]) for j in range(4)], jit_s=jit_s, jit_a=jit_a, stale=stale, stale_pct=100 * stale / max(1, len(S) - 1)))
print("per episode: sag = mean(action - state) deg [pan lift elbow wflex], rms error, lag (ticks of 33 ms), follower jitter / leader jitter (deg), stale follower samples")
for r in rows:
    print(f"ep {r['ep']:2d} {r['square']}: sag {np.round(r['sag'], 1).tolist()} rms {np.round(r['rms'], 1).tolist()} lag {r['lag']} "
          f"jitter fol {np.round(r['jit_s'], 2).tolist()} lead {np.round(r['jit_a'], 2).tolist()} stale {r['stale']}/{r['n']} ({r['stale_pct']:.0f}%)")
for name, sel in (("near (ranks 7-8)", [r for r in rows if r["rank"] >= 7]), ("far (ranks 1-3)", [r for r in rows if r["rank"] <= 3])):
    if not sel:
        continue
    sag = np.mean([r["sag"] for r in sel], 0); rms = np.mean([r["rms"] for r in sel], 0)
    lag = np.nanmean([[x if x is not None else np.nan for x in r["lag"]] for r in sel], 0)
    js = np.mean([r["jit_s"] for r in sel], 0); ja = np.mean([r["jit_a"] for r in sel], 0); st = np.mean([r["stale_pct"] for r in sel])
    print(f"\n{name}: {len(sel)} episodes\n   sag (deg)   pan {sag[0]:+.1f} lift {sag[1]:+.1f} elbow {sag[2]:+.1f} wflex {sag[3]:+.1f}\n   rms error   pan {rms[0]:.1f} lift {rms[1]:.1f} elbow {rms[2]:.1f} wflex {rms[3]:.1f}"
          f"\n   lag (ms)    pan {33 * lag[0]:.0f} lift {33 * lag[1]:.0f} elbow {33 * lag[2]:.0f} wflex {33 * lag[3]:.0f}\n   jitter fol  {np.round(js, 2).tolist()}  leader {np.round(ja, 2).tolist()}\n   stale follower samples {st:.1f} %")
json.dump(rows, open("data/grasp_drill/20261101/teleop_quality.json", "w"), indent=1)
