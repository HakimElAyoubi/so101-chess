"""Variant A: LeHak0/so101_chess_moves_coords = the same 202 episodes, observation.state extended from 6 to 10 values:
[6 joints, from_x, from_y, to_x, to_y] (metres, arm base frame, per-session board pose). Videos are copied unchanged."""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from targets import Projector, session_of

SRC = Path.home() / ".cache/huggingface/lerobot/LeHak0/so101_chess_moves"
DST = Path.home() / ".cache/huggingface/lerobot/LeHak0/so101_chess_moves_coords"
GOAL_NAMES = ["goal_from_x", "goal_from_y", "goal_to_x", "goal_to_y"]


def main():
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    P = Projector()
    ep = pd.concat([pd.read_parquet(f) for f in sorted((DST / "meta/episodes").glob("*/*.parquet"))])
    goal = {}
    for e, tasks in zip(ep.episode_index, ep.tasks):
        f, t = P.target_xyz(tasks[0], session_of(e))
        goal[int(e)] = np.array([f[0], f[1], t[0], t[1]], np.float32)
    for fpath in sorted((DST / "data").glob("*/*.parquet")):
        df = pd.read_parquet(fpath)
        df["observation.state"] = [np.concatenate([np.asarray(s, np.float32), goal[int(e)]])
                                   for s, e in zip(df["observation.state"], df["episode_index"])]
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), fpath)
    info = json.loads((DST / "meta/info.json").read_text())
    st = info["features"]["observation.state"]
    st["shape"] = [10]
    names = st.get("names")
    if isinstance(names, list):
        st["names"] = names + GOAL_NAMES
    elif isinstance(names, dict):                  # {"motors": [...]} style
        k = next(iter(names)); names[k] = list(names[k]) + GOAL_NAMES
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4))
    print("goal coords ready; recomputing stats...", flush=True)
    r = subprocess.run([str(Path(sys.executable).parent / "lerobot-edit-dataset"), "--repo_id", "LeHak0/so101_chess_moves_coords",
                        "--root", str(DST), "--new_repo_id", "LeHak0/so101_chess_moves_coords", "--new_root", str(DST),
                        "--operation.type", "recompute_stats"], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-2500:])
    stats = json.loads((DST / "meta/stats.json").read_text())["observation.state"]
    print("state stats mean:", np.round(stats["mean"], 3), "\nstd:", np.round(stats["std"], 3))


if __name__ == "__main__":
    main()
