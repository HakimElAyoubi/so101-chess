"""Automatic board calibration with the wrist camera, through a running policy player (play/policy_player.py).

For every square: the player points the closed jaws at it (its own IK, sag hold, 11 cm travel between squares), the
camera measures where the square really is relative to the fingertips (play/board_vision.py: checkerboard corners +
PnP with the real 28 mm pitch, independent of the arm's joint errors), the target is nudged by the measured offset,
measured again (the second measurement cancels the arm's orientation error), and the square is taught to the player
(`centred:x,y,z`): the board model + correction map are refitted and saved after each square, so the run can be
interrupted and resumed. Then a verification pass measures every square once more without nudging.

    python calibration/auto_board_calib.py                 # all 64 squares, then verify
    python calibration/auto_board_calib.py --verify-only   # measure only, with the current calibration
    --squares a8,b8,...  --height-mm 45  --port 8830  --keep (don't clear the previous centred squares)
Progress: data/progress.txt (the progress page). Results: data/calibration/<time>/ (per-square JSON, debug images,
report.md). Stop at any time with Space in the player (the arm holds) or Ctrl-C here.
"""
import argparse, base64, json, sys, time
from datetime import datetime
from pathlib import Path
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "play"), str(ROOT / "goal")]
import staged  # noqa: E402
import targets  # noqa: E402
from board_vision import BoardVision  # noqa: E402

PROG = ROOT / "data/progress.txt"
SNAKE = [f"{f}{r}" for r in range(8, 0, -1) for f in ("abcdefgh" if r % 2 == 0 else "hgfedcba")]   # a8..h8, h7..a7, ...


class Player:
    def __init__(self, port):
        self.U = f"http://127.0.0.1:{port}"

    def state(self):
        return json.load(urllib.request.urlopen(self.U + "/state", timeout=5))

    def cmd(self, c):
        urllib.request.urlopen(urllib.request.Request(self.U, json.dumps({"cmd": c}).encode(), {"Content-Type": "application/json"}), timeout=5)

    def raw(self):
        r = json.load(urllib.request.urlopen(self.U + "/raw", timeout=5))
        img = cv2.imdecode(np.frombuffer(base64.b64decode(r["jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR) if r["jpeg_b64"] else None
        return r, (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None)

    def wait(self, cond, max_s, what):
        t0 = time.time()
        while time.time() - t0 < max_s:
            s = self.state()
            if cond(s):
                return s
            if s["msg"].startswith("Stopped"):
                raise KeyboardInterrupt("stopped in the player")
            time.sleep(0.15)
        raise TimeoutError(what)

    def move(self, c, max_s=60):
        """Send a move command and wait until the player is idle again."""
        self.cmd(c)
        try:
            # the path is planned in the player's control thread: an awkward target can take many seconds of IK
            self.wait(lambda s: s["phase"] in ("pointing", "homing"), 25, f"{c}: the player didn't start moving")
        except TimeoutError:
            s = self.state()
            raise RuntimeError(f"{c}: not started ({s['msg']})")
        return self.wait(lambda s: s["phase"] == "idle", max_s, f"{c}: still moving after {max_s} s")


def latest_session():
    """Reload the board poses the player keeps saving (in place: every module shares the dict)."""
    fresh = json.loads((ROOT / "goal/board_poses.json").read_text())
    targets.POSES["sessions"].update(fresh["sessions"]); targets.POSES["episodes"] = fresh["episodes"]
    return max((k for k in targets.POSES["sessions"] if k.isdigit()), key=int)


def progress(i, n, t0, text):
    el = time.time() - t0; eta = el / i * (n - i) if i else 0
    PROG.write_text(f"[{'#' * int(30 * i / n):<30}] {100 * i / n:5.1f}%  {i}/{n} squares  elapsed {el / 60:4.1f} min  "
                    f"ETA {eta / 60:4.1f} min | board calibration: {text}\n")


def measure(bv, pl, square, out_dir, tag, n_frames=3):
    """The median of n measurements on fresh frames; None if none succeeds."""
    J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    sess = latest_session()
    good, fails = [], []
    for k in range(n_frames):
        r, img = pl.raw()
        if img is None:
            time.sleep(0.2); continue
        q6 = np.array([r["joints"][j] for j in J])
        m = bv.measure(img, q6, square, sess, debug=(out_dir / f"{square}_{tag}{k}.jpg") if k == 0 else None)
        if m and "offset" in m and "why" not in m:
            m["q6"] = q6.tolist(); good.append(m)
        else:
            fails.append(m.get("why", "?") if m else "?")
        time.sleep(0.12)
    if not good:
        return None, fails
    # the frames must agree: keep the majority shift, then drop any frame more than 2 mm from the others' median
    shifts = [tuple(g["shift"]) for g in good]; maj = max(set(shifts), key=shifts.count)
    good = [g for g in good if tuple(g["shift"]) == maj]
    if len(good) >= 3:
        med0 = np.median([g["offset"] for g in good], axis=0)
        good = [g for g in good if np.linalg.norm(np.array(g["offset"]) - med0) < 0.002] or good
    med = {k: float(np.median([g[k] for g in good])) for k in ("h_tips", "h_tips_fk", "board_z", "tilt_deg", "jaw_vs_normal_deg", "scale", "rms_px", "colour_conf")}
    med["offset"] = np.median([g["offset"] for g in good], axis=0).tolist()
    med["centre"] = np.median([g["centre"] for g in good], axis=0).tolist()
    med["site"] = good[0]["site"]; med["shift"] = good[0]["shift"]; med["n_good"] = len(good)
    med["spread_mm"] = float(1000 * np.max(np.std([g["offset"] for g in good], axis=0))) if len(good) > 1 else 0.0
    med["used"] = int(np.median([g["used"] for g in good])); med["q6"] = good[0]["q6"]
    if med["spread_mm"] > 1.5 or med["tilt_deg"] > 8.0 or (med["colour_conf"] < 0.7 and med["shift"] != [0, 0]):
        return None, fails + [f"inconsistent (spread {med['spread_mm']:.1f} mm, tilt {med['tilt_deg']:.1f} deg, colour {med['colour_conf']:.2f}, shift {med['shift']})"]
    return med, fails


def grid_check(bv, pl, sq, out, tag):
    """After teaching: is the model's grid coherent with the real one around the square? The model's corners of the 3 x 3
    squares around it are projected into a fresh frame and compared with the detected corners; the picture (green: the
    model's grid, red crosses: detected corners) goes to <sq>_<tag>.jpg. Returns the mean / max disagreement in mm."""
    J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    sess = latest_session()
    r, img = pl.raw()
    if img is None:
        return None
    q6 = np.array([r["joints"][j] for j in J]); W = img.shape[1]; P = bv.P
    f0, r0 = "abcdefgh".index(sq[0]), int(sq[1])
    m = bv.measure(img, q6, sq, sess)
    view = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    def model_pt(f, rr):                                 # at the board's measured height there (the height map), not
        xyz = P.board_point(f, rr, sess)                 # the model's: 3-9 mm lower near rank 8, which the camera's
        xyz[2] += P.board_dz(sess, xyz)                  # slanted view would show as a 3-5 mm lateral shift
        return xyz
    for i in range(-1, 3):                               # the model's grid lines around the square
        for line in ([(f0 + i - 0.5, r0 + t) for t in np.linspace(-1.5, 1.5, 25)], [(f0 + t, r0 + i - 0.5) for t in np.linspace(-1.5, 1.5, 25)]):
            pts = np.array([model_pt(f, rr) for f, rr in line if -0.5 <= f <= 7.5 and 0.5 <= rr <= 8.5])
            if len(pts) < 2:
                continue
            uv, vis = P.project(q6, pts, W); uv = uv[vis]
            if len(uv) > 1:
                cv2.polylines(view, [uv.astype(np.int32)], False, (0, 220, 0), 2, cv2.LINE_AA)
    devs = []
    if m and "corners" in m:
        for (a_, b_), px in m["corners"]:
            e, _ = P.project(q6, np.array([model_pt(f0 + a_, r0 + b_)]), W)
            cv2.drawMarker(view, (int(round(px[0])), int(round(px[1]))), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
            d_px = np.linalg.norm(e[0] - np.array(px))
            # pixels -> mm through the local square size
            u, _ = P.project(q6, np.array([model_pt(f0 + a_ + 1, r0 + b_)]), W)
            devs.append(28.0 * d_px / max(np.linalg.norm(u[0] - e[0]), 1.0))
    res = dict(mean_mm=float(np.mean(devs)) if devs else None, max_mm=float(np.max(devs)) if devs else None, n=len(devs),
               measured_mm=(1000 * np.hypot(*m["offset"])) if m and "offset" in m and "why" not in m else None)
    cv2.putText(view, f"{sq}: model grid (green) vs detected corners (red): " + (f"mean {res['mean_mm']:.1f} max {res['max_mm']:.1f} mm over {res['n']} corners" if devs else "no corners")
                + (f"; tips vs centre {res['measured_mm']:.1f} mm" if res["measured_mm"] is not None else ""), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"{sq}_{tag}.jpg"), view)
    return res


def to_board_axes(offset_xy, sess):
    """World xy offset -> (far = toward rank 1, h = toward the h-file) in mm, the player's nudge axes."""
    a = np.radians(targets.POSES["sessions"][sess]["yaw_deg"])
    far = np.array([np.cos(a), np.sin(a)]); h = np.array([-np.sin(a), np.cos(a)])
    o = np.asarray(offset_xy)
    return 1000 * float(o @ far), 1000 * float(o @ h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--height-mm", type=float, default=0.0, help="gripper site above the board while measuring; 0 = each square's own "
                    "pawn grasp hover (height and jaw tilt as played)")
    ap.add_argument("--squares", default="all")
    ap.add_argument("--keep", action="store_true", help="keep the squares centred before (default: start the tour over)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tol-mm", type=float, default=0.3, help="stop nudging a square below this residual")
    ap.add_argument("--max-nudges", type=int, default=3)
    ap.add_argument("--pinhole", action="store_true", help="frames rendered by the twin (policy_player.py --fake --fake-render)")
    ap.add_argument("--from", dest="from_sq", default=None, help="resume the tour at this square (with --keep)")
    ap.add_argument("--max-nudge-mm", type=float, default=12.0, help="a measured offset beyond this is a misdetection, not a nudge")
    a = ap.parse_args()
    squares = SNAKE if a.squares == "all" else a.squares.split(",")
    if a.from_sq:
        squares = squares[squares.index(a.from_sq):]
    out = ROOT / "data/calibration" / datetime.now().strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True)
    pl, bv = Player(a.port), BoardVision(pinhole_fovy=47.0 if a.pinhole else None)
    Pl = staged.Planner(bv.P); ready6 = np.array([json.loads((ROOT / "recording/ready_pose.json").read_text())[j] for j in
                                                  ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")])
    s = pl.wait(lambda s: s["phase"] == "idle", 120, "the player is busy: still not idle after 2 min")
    pl.cmd("nudge:reset"); time.sleep(0.3)                   # no stale preview from earlier nudges (mine or the user's)
    pl.cmd(f"note:Automatic calibration starting: {len(squares)} squares. Please don't press anything on the page; Space stops it.")
    if not s["at_ready"] and not s["pointed"]:
        print("going to the ready pose first"); pl.move("go_ready", 60)
    log = {"height_mm": a.height_mm, "started": datetime.now().isoformat(timespec="seconds"), "squares": {}, "verify": {}}
    t0 = time.time()

    def pose_for(sq):
        """(site height mm, jaw tilt deg): the pawn grasp hover the staged player would use, or the chosen height."""
        if a.height_mm:
            return a.height_mm, 0.0
        sess = latest_session()
        q, info = Pl.grasp_hover(targets.Projector.square_xyz(bv.P, sq, sess), "pawn", ready6, staged.HOVER_GRIP)
        return (1000 * info["h"], info["tilt"]) if q is not None else (35.0, 0.0)

    def point(sq):
        h, tilt = pose_for(sq)
        for h_try in (h, h - 5, h - 10, h - 15):
            try:
                s = pl.move(f"point:{sq}:{h_try:.0f}:{tilt:.1f}", 60)
            except RuntimeError as e:
                if "No IK" in str(e) and h_try > h - 15:
                    continue
                raise
            if s["pointed"] != sq:
                raise RuntimeError(f"point:{sq} refused: {s['msg']}")
            time.sleep(0.5)                               # settle after the sag hold
            return h_try, tilt
        raise RuntimeError(f"point:{sq}: no IK solution down to {h - 15:.0f} mm")

    def calibrate_square(sq, h, tilt):
        m, fails = measure(bv, pl, sq, out, "a")
        if m is None:                                  # the camera can't see enough corners from the hover pose
            for h2, t2 in ((45.0, 0.0), (35.0, 0.0)):  # (a corner square, the jaws in the way): vertical jaws instead
                try:
                    pl.move(f"point:{sq}:{h2:.0f}:{t2:.1f}", 60); time.sleep(0.5)
                except RuntimeError:
                    continue
                m, fails2 = measure(bv, pl, sq, out, f"b{h2:.0f}"); fails += fails2
                if m is not None:
                    h, tilt = h2, t2; break
        rec = {"first": m, "first_fails": fails, "nudges": [], "pose": [h, tilt]}
        log["squares"][sq] = rec
        if m is None:
            print(f"{sq}: no measurement ({fails}); skipped"); rec["skipped"] = True; return
        print(f"{sq}: model off by {1000 * m['offset'][0]:+5.1f} {1000 * m['offset'][1]:+5.1f} mm (shift {m['shift']}), tips {1000 * m['h_tips']:.1f} mm up "
              f"(FK {1000 * m['h_tips_fk']:.1f}), tilt {m['tilt_deg']:.1f} deg, rms {m['rms_px']:.2f} px, {m['n_good']} frames spread {m['spread_mm']:.2f} mm")
        if np.hypot(*m["offset"]) * 1000 > a.max_nudge_mm:
            print(f"   {1000 * np.hypot(*m['offset']):.0f} mm off: not believable, skipped"); rec["skipped"] = True; return
        for k in range(a.max_nudges):
            if np.hypot(*m["offset"]) * 1000 <= a.tol_mm:
                break
            far, hh = to_board_axes(m["offset"], latest_session())
            if np.hypot(far, hh) > a.max_nudge_mm:
                print(f"   measured {np.hypot(far, hh):.1f} mm after a nudge: a misdetection, kept the previous measurement"); break
            try:
                pl.move(f"nudge:mm:{far:.2f},{hh:.2f}", 30); time.sleep(0.4)
            except (RuntimeError, TimeoutError) as e:
                print(f"   nudge {k + 1} failed ({e}); preview dropped, kept the previous measurement"); pl.cmd("nudge:reset"); time.sleep(0.3); break
            m2, fails2 = measure(bv, pl, sq, out, f"n{k}")
            rec["nudges"].append({"nudge_far_h_mm": [far, hh], "after": m2, "fails": fails2})
            if m2 is None:
                print(f"   nudge {k + 1}: no measurement afterwards ({fails2})"); break
            if np.hypot(*m2["offset"]) * 1000 > a.max_nudge_mm:
                print(f"   nudge {k + 1}: the next measurement jumped to {1000 * np.hypot(*m2['offset']):.0f} mm (a misdetection): kept the previous one"); break
            print(f"   nudge {k + 1} ({far:+.1f}, {hh:+.1f} mm) -> residual {1000 * m2['offset'][0]:+5.2f} {1000 * m2['offset'][1]:+5.2f} mm")
            m = m2
        x, y = m["centre"]; z = m["board_z"]
        pl.cmd(f"centred:{x:.6f},{y:.6f},{z:.6f}"); time.sleep(0.6)
        rec["taught"] = [x, y, z]; rec["final"] = m
        g = grid_check(bv, pl, sq, out, "grid")                 # the taught model against the real squares around it
        rec["grid"] = g
        if g and g["max_mm"] is not None:
            flag = "" if g["max_mm"] <= 2.5 else "   !! grid incoherent here"
            print(f"   grid check: model vs detected corners mean {g['mean_mm']:.2f} max {g['max_mm']:.2f} mm ({g['n']} corners){flag}")

    def verify(squares, out):
        t1 = time.time()
        for i, sq in enumerate(squares):
            progress(i, len(squares), t1, f"verifying {sq}")
            pl.cmd(f"note:AUTOMATIC CALIBRATION: verifying {sq} ({i + 1}/{len(squares)}). Please don't press anything; Space stops it.")
            try:
                point(sq)
            except (RuntimeError, TimeoutError) as e:
                print(f"verify {sq}: {e}; skipped"); log["verify"][sq] = {"error": str(e)}; continue
            m, fails = measure(bv, pl, sq, out, "v")
            log["verify"][sq] = m if m else {"fails": fails}
            g = grid_check(bv, pl, sq, out, "vgrid")
            if m:
                m["grid"] = g
            print(f"verify {sq}: " + (f"residual {1000 * m['offset'][0]:+5.2f} {1000 * m['offset'][1]:+5.2f} mm, tips {1000 * m['h_tips']:.1f} mm, rms {m['rms_px']:.2f} px" if m else f"no measurement ({fails})")
                  + (f" | grid vs corners mean {g['mean_mm']:.1f} max {g['max_mm']:.1f} mm" if g and g.get("max_mm") is not None else ""))
            (out / "log.json").write_text(json.dumps(log, indent=1))
        progress(len(squares), len(squares), t1, "verification done")

    try:
        if not a.verify_only:
            if not a.keep:
                pl.cmd("calib_clear"); time.sleep(0.3)
            for i, sq in enumerate(squares):
                progress(i, len(squares), t0, f"{sq}")
                pl.cmd(f"note:AUTOMATIC CALIBRATION RUNNING ({i + 1}/{len(squares)}: {sq}). Please don't press anything on this page; Space stops it.")
                try:
                    h, tilt = point(sq)
                except (RuntimeError, TimeoutError) as e:
                    print(f"{sq}: {e}; skipped"); log["squares"][sq] = {"skipped": True, "error": str(e)}; continue
                try:
                    calibrate_square(sq, h, tilt)
                except (RuntimeError, TimeoutError) as e:
                    print(f"{sq}: {e}; skipped"); log["squares"][sq] = {"skipped": True, "error": str(e)}
                (out / "log.json").write_text(json.dumps(log, indent=1))
            progress(len(squares), len(squares), t0, "tour done" + ("" if a.no_verify else ", verifying"))
        if not a.no_verify:
            verify(squares, out)
    except KeyboardInterrupt:
        print("\nstopped"); log["stopped"] = True
    finally:
        (out / "log.json").write_text(json.dumps(log, indent=1))
        try:
            s = pl.state()
            if s["phase"] == "idle":
                pl.cmd("note:Calibration run finished: H to go back to the ready pose.")
        except Exception:
            pass
    report(log, out)




def report(log, out):
    sq = log["squares"]; ver = log["verify"]
    lines = [f"# Board calibration {log['started']}", ""]
    first = {k: v["first"]["offset"] for k, v in sq.items() if v.get("first")}
    if first:
        e = np.array(list(first.values())) * 1000; n = np.linalg.norm(e, axis=1)
        lines.append(f"Before: the previous calibration was off by {n.mean():.1f} mm on average, worst {n.max():.1f} mm ({max(first, key=lambda k: np.hypot(*first[k]))}), over {len(first)} squares.")
    res = {k: v["offset"] for k, v in ver.items() if v and "offset" in v}
    if res:
        e = np.array(list(res.values())) * 1000; n = np.linalg.norm(e, axis=1)
        lines.append(f"After: residual {n.mean():.2f} mm on average, worst {n.max():.2f} mm ({max(res, key=lambda k: np.hypot(*res[k]))}), over {len(res)} squares"
                     + (f"; {len(ver) - len(res)} squares could not be verified." if len(ver) > len(res) else "."))
    fin = {k: v["final"] for k, v in sq.items() if v.get("final")}
    if fin:
        hz = {k: 1000 * (v["board_z"] - 0.010) for k, v in fin.items()}
        lines += ["", f"Board top vs the model (FK terms): {np.mean(list(hz.values())):+.1f} mm on average, from {min(hz.values()):+.1f} ({min(hz, key=hz.get)}) to {max(hz.values()):+.1f} ({max(hz, key=hz.get)}) mm.",
                  f"Board normal vs the FK vertical (the arm's orientation error at that pose, if the board is level: joint errors, "
                  f"bending): {np.mean([v['tilt_deg'] for v in fin.values()]):.1f} deg on average, max {max(v['tilt_deg'] for v in fin.values()):.1f} "
                  f"({max(fin, key=lambda k: fin[k]['tilt_deg'])}).",
                  f"Lens / 28 mm pitch consistency (PnP distance over FK distance): {np.mean([v['scale'] for v in fin.values()]):.3f} +- {np.std([v['scale'] for v in fin.values()]):.3f}.",
                  f"Plane fits: rms {np.mean([v['rms_px'] for v in fin.values()]):.2f} px, {np.mean([v['used'] for v in fin.values()]):.1f} corners on average.", ""]
        lines.append("| square | model was off (mm, x y) | residual after (mm) | board top (mm) | orientation error (deg) | pose (mm, deg) |")
        lines.append("|---|---|---|---|---|---|")
        for k in SNAKE:
            if k in fin:
                f, v = first.get(k), res.get(k)
                lines.append(f"| {k} | {1000 * f[0]:+.1f} {1000 * f[1]:+.1f} | " + (f"{1000 * np.hypot(*v):.2f}" if v else "-") + f" | {hz[k]:+.1f} | {fin[k]['tilt_deg']:.1f} | {sq[k]['pose'][0]:.0f}, {sq[k]['pose'][1]:.0f} |")
    skipped = [k for k, v in sq.items() if v.get("skipped")]
    if skipped:
        lines += ["", "Skipped (no measurement): " + ", ".join(skipped)]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12])); print("->", out / "report.md")


if __name__ == "__main__":
    main()
