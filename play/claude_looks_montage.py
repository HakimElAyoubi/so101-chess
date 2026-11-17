"""Montage of a Claude-skill trial's looks: python play/claude_looks_montage.py play/results*/claude/<dir> [out.png]
Each tile: the image as sent, Claude's target pixel (cyan cross), the measured offset, the truth (twin runs)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / "goal")]

d = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 else d / "montage.png"
tiles = []
for line in open(d / "looks.jsonl"):
    r = json.loads(line); a = r.get("answer", {})
    img = cv2.imread(str(d / f"look_{r['n']:02d}_{r['kind']}.png"))
    if img is None:
        continue
    if a.get("target_px"):
        u, v = (int(round(x)) for x in a["target_px"]); cv2.drawMarker(img, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 40, 2)
    if r.get("truth_mm") and r.get("land_w") and r.get("q") is not None:        # the twin's truth as a magenta diamond
        try:
            from llm_skill import ClaudeSkill, W_RAW
            global SK
            try:
                SK
            except NameError:
                SK = ClaudeSkill(repo="claude:mock")
            lw = np.asarray(r["land_w"], float); tw = lw.copy(); tw[:2] += np.asarray(r["truth_mm"], float) / 1000.0
            uv = SK.project(np.asarray(r["q"], float), tw)[0][0]
            px = (uv - [r.get("x0", 0), r.get("y0", 0)]) * r.get("scale", 1.5)
            if "x0" not in r:                                                   # crop origin from LAND's pixel
                lp = np.asarray(r["land_px"], float); uvl = SK.project(np.asarray(r["q"], float), lw)[0][0]
                px = lp + (uv - uvl) * 1.5
            cv2.drawMarker(img, (int(round(px[0])), int(round(px[1]))), (255, 0, 255), cv2.MARKER_DIAMOND, 34, 2)
        except Exception as e:
            print("truth marker failed:", e)
    txt = f"look {r['n']} {r['kind']} {r.get('phase', '')}: {r.get('action', r.get('error', '?'))}"
    if r.get("d_mm"):
        txt += f"  d ({r['d_mm'][0]:+.1f}, {r['d_mm'][1]:+.1f}) mm"
    if r.get("truth_mm"):
        txt += f"  truth ({r['truth_mm'][0]:+.1f}, {r['truth_mm'][1]:+.1f})"
    for col, th in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th)
        cv2.putText(img, f"conf {a.get('confidence')} {a.get('_latency_s', '')} s  {str(a.get('notes', ''))[:70]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th)
    tiles.append(cv2.resize(img, (560, 560)))
if not tiles:
    sys.exit("no looks")
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite(str(out), np.vstack(rows)); print("wrote", out)
