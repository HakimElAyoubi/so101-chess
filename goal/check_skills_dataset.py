"""Check LeHak0/so101_chess_skills after goal/build_skills_dataset.py: counts match the clip plan, tasks are the skill
strings, the move encoding is gone, and a contact sheet of sample clips (start / middle / event / end) with their
markers -> goal/check/skills_contact_sheet.jpg"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from lerobot.datasets import LeRobotDataset

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "skill_clips.json").read_text())
REPO = PLAN["repo"]
DST = Path.home() / ".cache/huggingface/lerobot" / REPO
CAM = "observation.images.wrist"


def main():
    ds = LeRobotDataset(REPO, root=DST)
    clips = PLAN["clips"]
    n_frames = sum(c["k1"] - c["k0"] for c in clips)
    print(f"episodes {ds.meta.total_episodes} (plan {len(clips)}), frames {ds.meta.total_frames} (plan {n_frames})")
    print("features:", sorted(ds.meta.features))
    print("tasks:", sorted(ds.meta.tasks.index) if hasattr(ds.meta.tasks, "index") else ds.meta.tasks)
    assert ds.meta.total_episodes == len(clips) and ds.meta.total_frames == n_frames
    assert "observation.environment_state" not in ds.meta.features
    # a few clips of each kind, spread over piece types; drop-zone places included
    want = []
    for kind in ("grasp", "place"):
        seen = set()
        for c in clips:
            key = (c["task"].split()[4] if kind == "grasp" else c["task"].split()[3], "to_xyz" in c)   # piece name
            if c["kind"] == kind and key not in seen and len(want) < (6 if kind == "grasp" else 12):
                seen.add(key); want.append(c)
    rows = []
    for c in want:
        a = int(ds.meta.episodes["dataset_from_index"][c["clip"]])
        n = c["k1"] - c["k0"]
        ev = (2.5 if c["kind"] == "grasp" else 2.0) * 30
        tiles = []
        for k in (0, n // 3, int(ev), n - 1):
            f = ds[a + min(k, n - 1)]
            img = cv2.cvtColor((f[CAM].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (384, 216))
            cv2.putText(img, f"{k / 30:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            tiles.append(img)
        row = np.hstack(tiles)
        label = np.full((24, row.shape[1], 3), 255, np.uint8)
        cv2.putText(label, f"clip {c['clip']} (ep {c['src_ep']}): {c['task']}  <- {c['src_task']}", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows.append(np.vstack([label, row]))
    out = ROOT / "check/skills_contact_sheet.jpg"
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("contact sheet:", out)


if __name__ == "__main__":
    main()
