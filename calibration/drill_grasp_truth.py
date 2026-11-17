"""Where the human really took the jaws, per grasp-drill episode: the hover (frame 0, measured joints) vs the lowest
point of the trajectory (the grasp), both through the twin's forward kinematics of the gripper site. The piece stands
at the printed square + the drill's offset, so the lateral motion hover -> grasp, averaged per square, is the model's
hover error at that square as the human corrected it (plus the gripper's own fixed offset, the same at every square
in the gripper's frame). No camera involved.
    python calibration/drill_grasp_truth.py [dataset root]   (default: the Hugging Face cache copy of the grasp dataset)
"""
import sys, json, glob, os
sys.path[:0] = ["play", "goal", "calibration", "recording", "."]
import numpy as np, pandas as pd
import staged as S, targets, auto_board_calib as A

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/huggingface/lerobot/LeHak0/so101_chess_grasp")
P = targets.Projector(); Pl = S.Planner(P); s = P.s
meta = {int(json.loads(l)["episode_index"]): json.loads(l) for l in open("recording/grasp_meta.jsonl") if l.strip()}
import time
parts = []
for f in sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet"))):
    for attempt in range(3):                      # the recorder rewrites the newest file after every episode
        try:
            parts.append(pd.read_parquet(f)); break
        except Exception as ex:
            if attempt == 2:
                print(f"skipping {os.path.basename(f)} (being written: {str(ex)[:60]})")
            time.sleep(1.5)
df = pd.concat(parts, ignore_index=True)
site = s.model.site("gripperframe").id


def site_pose(q6):
    s.set_joints(np.asarray(q6, float)); return s.data.site_xpos[site].copy(), s.data.site_xmat[site].reshape(3, 3).copy()


rows = []
for ep, g in df.groupby("episode_index"):
    m = meta.get(int(ep))
    if not m:
        continue
    Q = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())
    poses = [site_pose(q) for q in Q]
    z = np.array([p[2] for p, _ in poses]); k = int(np.argmin(z))
    p0, R0 = poses[0]; pk, Rk = poses[k]
    sess = str(m["board_session"]); d = pk - p0
    far, hh = A.to_board_axes(d[:2], sess)                                         # real mm toward rank 1, toward h
    a_open = R0[:2, 2] / (np.linalg.norm(R0[:2, 2]) + 1e-9); a_x = R0[:2, 0] / (np.linalg.norm(R0[:2, 0]) + 1e-9)
    along, across = 1000 * float(d[:2] @ a_open), 1000 * float(d[:2] @ a_x)      # in the gripper's frame at the hover
    # where the model's spot is vs the printed square: the piece stands at printed centre + offset; the hover site stands at
    # the model's centre + offset (+ its own fixed grasp offset): so (far, h) + fixed = model error
    rows.append(dict(ep=int(ep), item=m["drill"], square=m["square"], piece=m["piece"], sess=sess, off=m["offset_mm"], n=len(Q), k=k,
                     drop_mm=round(1000 * (p0[2] - pk[2]), 1), far=round(far, 1), h=round(hh, 1), along=round(along, 1), across=round(across, 1),
                     grip0=round(float(Q[0][5]), 1), gripk=round(float(Q[k][5]), 1)))
for r in rows:
    print(f"ep {r['ep']:3d} item {r['item']:3d} {r['square']} {r['piece']:6s} sess {r['sess']} off {r['off']}: hover->grasp {r['far']:+6.1f} mm toward rank 1, {r['h']:+6.1f} toward h "
          f"| jaws frame along {r['along']:+6.1f} across {r['across']:+6.1f} | down {r['drop_mm']:5.1f} mm at frame {r['k']}/{r['n']} grip {r['grip0']:.0f}->{r['gripk']:.0f}%")
by = {}
for r in rows:
    if r["sess"] == targets.LATEST:
        by.setdefault(r["square"], []).append(r)
print(f"\nper square (session {targets.LATEST}), mean lateral motion hover -> grasp:")
for sq, rs in by.items():
    f = np.mean([r["far"] for r in rs]); h = np.mean([r["h"] for r in rs]); al = np.mean([r["along"] for r in rs]); ac = np.mean([r["across"] for r in rs])
    sd = np.std([[r["far"], r["h"]] for r in rs], 0) if len(rs) > 1 else np.zeros(2)
    print(f"   {sq}: {f:+6.1f} toward rank 1, {h:+6.1f} toward h  (sd {sd[0]:.1f}/{sd[1]:.1f}, n={len(rs)}) | jaws: along {al:+6.1f}, across {ac:+6.1f}")
allr = [r for rs in by.values() for r in rs]
if allr:
    print(f"all {len(allr)} episodes: jaws frame along {np.mean([r['along'] for r in allr]):+.1f} +- {np.std([r['along'] for r in allr]):.1f}, across {np.mean([r['across'] for r in allr]):+.1f} +- {np.std([r['across'] for r in allr]):.1f} mm")
json.dump(rows, open("data/grasp_drill/20261101/grasp_truth.json", "w"), indent=1)
