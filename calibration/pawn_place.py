"""Physical check of the board calibration with one pawn: the robot picks the pawn up from wherever it stands on the
square (the camera finds it) and puts it down where IT believes the square's centre is; the pawn then stands at the
model's centre, and its position relative to the square's own corners in the photograph (pawn_vision: a plane
homography through the detected corners, the camera's pose model not involved) is the calibration error there, in
real millimetres, both axes at once. The jaws' geometry enters only through the shut gap's midpoint (twin): doing it
with the jaws rolled by 90 deg (psi 20 and 110) separates that from the calibration error.
     python pawn_place.py <out_dir> <square> [psi,psi] [--dry]
"""
import os, sys, json, time, numpy as np, cv2
sys.path[:0] = ["play", "goal", "calibration", "."]
import staged as S, targets
import auto_board_calib as A
import pawn_vision as PV
SQ = sys.argv[2]
OUT = os.path.join(sys.argv[1], time.strftime("%Y%m%d_%H%M%S") + f"_{SQ}"); os.makedirs(OUT, exist_ok=True)   # one folder per run
PSIS = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else [20.0]
REPEAT = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1       # cycles per roll (the grasp is only repeatable to ~2 mm)
PSIS = [p for p in PSIS for _ in range(REPEAT)]
DRY = "--dry" in sys.argv
WHERE = "--where" in sys.argv          # just measure where the pawn stands now (hover + camera-over photos), then ready
HOVER_ONLY = "--hover-only" in sys.argv   # grasp-free: the pawn from the top-down photo, the jaws' aim from hover photos (where the jaws
                                          # would come down relative to the pawn), no contact; writes hover_<square>.json
GRIP_OPEN, GRIP_SHUT, TIPS_H, GRASP_OFF, GRIP_ACROSS = 30.0, 1.0, 0.009, 0.014, 0.0025
TARGET_ALONG, TARGET_ACROSS = 0.0159, 0.0044   # m: the pawn relative to the site's vertical projection, along / across the jaws, in the
                                             # grasp-hover photo of every grasp that held (e4, a8, 30 Oct): the visual aim the jaws need   # jaws open (%), shut target, fingertips above the board at the grasp (m: the
                                                                      # pads clamp the pawn's body, not its head), fixed pad -> pawn axis aim (m)
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
pl = A.Player(8830); pv = PV.PawnVision(); P = pv.P; s = P.s; Pl = S.Planner(P)
r6 = np.array([json.load(open("recording/ready_pose.json"))[j] for j in J])
sess = A.latest_session(); SP = targets.POSES["sessions"][sess]
T0 = time.time()


def prog(k, n, text):
    el = time.time() - T0; pct = 100 * k / n; bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
    open("data/progress.txt", "w").write(f"[{bar}] {pct:3.0f}%  pawn test {SQ}: step {k}/{n}  {el:4.0f} s elapsed, ETA {el / max(k, 1) * (n - k):3.0f} s  {text}\n")


def raw():
    r, img = pl.raw(); return np.array([r["joints"][j] for j in J]), img


def grip(pct, wait=2.5):
    pl.cmd(f"grip:{pct:.1f}"); time.sleep(wait); q, _ = raw(); return float(q[5])


def gap_geom(q6):
    """At joints q6 (gripper included): opening axis in the world, the tips' midpoint offset from the site along it (m),
    the gap (m), the lower tip's z."""
    s.set_joints(q6); Rm = s.data.site_xmat[s.model.site("gripperframe").id].reshape(3, 3)
    t1, t2 = (np.asarray(t) for t in s.jaw_tip_positions()); site = s.data.site_xpos[s.model.site("gripperframe").id].copy()
    a_open = Rm[:, 2].copy()
    return a_open, float(np.dot((t1 + t2) / 2 - site, a_open)), float(np.linalg.norm(t1 - t2)), min(t1[2], t2[2])


CUR = dict(nudge=(0.0, 0.0), grip=None)       # what the player holds now: its nudge (far, h mm) and the jaw setting its re-points use


def point(h_m, tilt, psi, g, nudge=None, max_s=60):
    """Point the site h_m above the real board at SQ (+ nudge (far, h) in the arm's mm along the board axes), jaws at g %.
    A nudge makes the player re-point with the jaw setting of its LAST point command, so when both the nudge and the
    jaws change, the point command (new jaws, old nudge) goes first; never change both on a descent."""
    nd = (0.0, 0.0) if nudge is None else (float(nudge[0]), float(nudge[1]))
    same = np.hypot(nd[0] - CUR["nudge"][0], nd[1] - CUR["nudge"][1]) < 0.05
    cmd = f"point:{SQ}:{1000 * h_m:.0f}:{tilt:.1f}:hold,psi={psi:.0f},grip={g:.0f}"
    if not same and CUR["grip"] != g:
        pl.move(cmd, max_s); CUR["grip"] = g; time.sleep(0.3)
    if not same:
        pl.cmd("nudge:reset"); time.sleep(0.4); pl.wait(lambda x: x["phase"] == "idle", 30, "reset")
        if np.hypot(*nd) > 0.05:
            pl.cmd(f"nudge:mm:{nd[0]:.2f},{nd[1]:.2f}"); time.sleep(0.6); pl.wait(lambda x: x["phase"] == "idle", 40, "nudge")
        CUR["nudge"] = nd
    pl.move(cmd, max_s); CUR["grip"] = g; time.sleep(0.5)
    return raw()


def real_to_arm(toward_r1, toward_h):
    """Real board mm -> the arm's mm along the board axes (the board is scaled in the arm's frame)."""
    return toward_r1 * (SP.get("scale_rank") or 1.0), toward_h * (SP.get("scale_file") or 1.0)


def cam_pose(c, h_cams=(0.15, 0.16, 0.14, 0.13, 0.17, 0.12, 0.11)):
    """A reachable pose with the wrist camera about 150 mm above the board, its optical axis aimed at the square's
    centre and as vertical as the arm allows (the pawn's base and head are then nearly concentric in the photo and
    its centre unambiguous), the jaws well above the board.
    -> (nudge (far, h) arm mm for the site, site height m, tilt, psi, axis angle from vertical) or None."""
    from chess_sim import config as CC
    best = None
    for psi in (-70.0, -90.0, -50.0, -110.0, 20.0, 110.0):
        for tilt in range(0, 60, 5):
            for h_cam in h_cams:
                site = np.array([c[0], c[1], Pl.z0 + 0.04]); q = None          # start low: next to the base the arm cannot reach high
                for _ in range(6):
                    q = Pl.ik(site, float(tilt), psi, r6, GRIP_OPEN)
                    if q is None:
                        break
                    s.set_joints(q); pos, R = s.camera_pose(CC.WRIST_CAM)
                    ax = -R[:, 2]                                          # the camera looks along -z of its frame
                    if ax[2] > -0.3:
                        q = None; break
                    hit = pos + ax * ((Pl.z0 - pos[2]) / ax[2])            # where the optical axis meets the board
                    site = site + np.array([c[0] - hit[0], c[1] - hit[1], (Pl.z0 + h_cam) - pos[2]])
                if q is None:
                    continue
                s.set_joints(q); pos, R = s.camera_pose(CC.WRIST_CAM); ax = -R[:, 2]
                hit = pos + ax * ((Pl.z0 - pos[2]) / ax[2])
                ang = np.degrees(np.arccos(np.clip(-ax[2], -1, 1)))
                site_w = s.data.site_xpos[s.model.site("gripperframe").id].copy()
                t1, t2 = s.jaw_tip_positions()
                if min(t1[2], t2[2]) - Pl.z0 < 0.035:                       # the jaws stay well above the board and the pawn
                    continue
                miss = np.hypot(hit[0] - c[0], hit[1] - c[1])         # the optical axis may miss the centre by a little: the
                score = ang + 2.0 * max(0.0, 1000 * miss - 3)         # pawn only has to be near the image centre
                if miss < 0.012 and (best is None or score < best[0]):
                    best = (score, A.to_board_axes(site_w[:2] - c[:2], sess), site_w[2] - Pl.z0, tilt, psi, ang)
    if best is None:
        return None
    _, nudge, h_site, tilt, psi, ang = best
    return nudge, h_site, float(tilt), psi, ang


def photo_over(name, last=None):
    """Move to the camera-over pose (jaws as they are), photograph, measure the pawn against the square's corners."""
    nudge, h_site, tilt, psi, ang = CAM
    q, img = point(h_site, tilt, psi, CUR["grip"] if CUR["grip"] is not None else GRIP_OPEN, nudge, 60)
    return photo(name, img, q, last, guess_px=(img.shape[1] / 2, img.shape[0] / 2))      # the camera is right above the square


def pawn_vs_site(r, q6, img, pawn=None):
    """From a measured grasp-hover photo r (pawn_vision.measure): the pawn relative to the gripper site's vertical
    projection onto the real board, (along, across) the jaws' opening axis, real mm. The camera is rigid on the jaws,
    so this is where the jaws are aimed, whatever the board model says."""
    Hm = np.array(r["H"]); W, H = img.shape[1], img.shape[0]
    s.set_joints(q6); site = s.data.site_xpos[s.model.site("gripperframe").id].copy(); Rm = s.data.site_xmat[s.model.site("gripperframe").id].reshape(3, 3)
    su, _ = pv.bv.project(q6, np.array([[site[0], site[1], s.cfg.board_thickness + P.board_dz(sess, site)]]), W, H)
    S = (pv.px_to_board(Hm, su[0], W, H)[0] - np.array(r["shift"])) * PV.PITCH           # (file, rank) mm from the true centre
    D = (np.array([r["toward_h"], -r["toward_r1"]]) if pawn is None else np.array([pawn[0], -pawn[1]])) - S
    far, hh = A.to_board_axes(Rm[:2, 2], sess); ax = np.array([hh, -far]); ax /= np.linalg.norm(ax)
    return float(D @ ax), float(D @ np.array([-ax[1], ax[0]])), (float(far) / 1000, float(hh) / 1000)


def photo(name, img, q6, last=None, mark=True, guess_px=None):
    """Save the photo + sidecar, measure the pawn against the square's corners. -> measurement or None."""
    r = pv.measure(img, q6, SQ, sess, last=last, guess_px=guess_px, debug=f"{OUT}/{name}_corners.jpg", log=lambda t: print(t))
    json.dump(dict(q6=q6.tolist(), square=SQ, session=sess, last=last, result=r), open(f"{OUT}/{name}.json", "w"), indent=1)
    v = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    if r and mark:
        cv2.drawMarker(v, (int(r["base_px"][0]), int(r["base_px"][1])), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
        cv2.drawMarker(v, (int(r["model_centre_px"][0]), int(r["model_centre_px"][1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
    cv2.putText(v, name + (f": pawn {r['toward_h']:+.1f} h, {r['toward_r1']:+.1f} r1 mm" if r else ": no measurement"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(f"{OUT}/{name}.jpg", v)
    if r:
        print(f"   {name}: pawn {r['toward_h']:+.1f} mm toward h, {r['toward_r1']:+.1f} mm toward rank 1 of the true centre "
              f"({r['n_corners']} corners, homography rms {r['hom_rms_px']:.1f} px, {r['parity']}, labels shifted {r['shift']}; "
              f"the model's centre is {PV.PITCH * np.hypot(*r['model_off_sq']):.1f} mm from the true one in this photo)")
    return r


# ---------------------------------------------------------------- run
pl.wait(lambda x: x["phase"] == "idle", 120, "busy"); pl.cmd("nudge:reset"); time.sleep(0.3)
if not pl.state()["at_ready"] and not pl.state()["pointed"]:
    pl.move("go_ready", 60)
c = P.square_xyz(SQ, sess); dz = P.board_dz(sess, c)
q_h, info = Pl.grasp_hover(c, "pawn", r6, S.HOVER_GRIP); h_hover, tilt = info["h"], info["tilt"]
CAM = cam_pose(c)
if CAM is None:
    print(f"{SQ}: no pose with the camera above the square; stopping"); sys.exit(1)
print(f"{SQ}: session {sess}, model centre {np.round(1000 * c[:2], 1).tolist()} mm, board there {1000 * dz:+.1f} mm vs the model; hover {1000 * h_hover:.0f} mm, tilt {tilt:.1f}; "
      f"camera-over pose: site nudged {np.round(CAM[0], 1).tolist()} mm (far, h), {1000 * CAM[1]:.0f} mm up, tilt {CAM[2]:.0f}, psi {CAM[3]:.0f}, camera axis {CAM[4]:.1f} deg from vertical")
results = []; NSTEP = 7 * len(PSIS); last = None; cyc = {}
if HOVER_ONLY:
    tag = sys.argv[sys.argv.index("--hover-only") + 1] if len(sys.argv) > sys.argv.index("--hover-only") + 1 and not sys.argv[sys.argv.index("--hover-only") + 1].startswith("-") else "h"
    q0, img0 = point(h_hover, tilt, PSIS[0], GRIP_OPEN); coarse = photo(f"{SQ}_hover{tag}_0_hover_coarse", img0, q0)
    r_td = photo_over(f"{SQ}_hover{tag}_1_topdown", (coarse["toward_h"], coarse["toward_r1"]) if coarse else None)
    if r_td is None or np.hypot(r_td["toward_h"], r_td["toward_r1"]) > 20:
        print("   pawn not found on the square; stopping"); pl.cmd("nudge:reset"); time.sleep(0.2); pl.move("go_ready", 60); sys.exit(1)
    pawn = (r_td["toward_h"], r_td["toward_r1"]); Ds = []
    # the hover photos are taken with the site nudged 20 mm toward the board's centre (at an edge square the camera
    # otherwise looks past the board and finds too few corners); the nudge is taken out of the result
    f0, r0 = "abcdefgh".index(SQ[0]), int(SQ[1]); inward = np.array([-(4.5 - r0), (3.5 - f0)]); inward = 20.0 * inward / max(np.linalg.norm(inward), 1e-6)
    views = [(inward[0], inward[1]), (inward[0] + 3.0, inward[1] - 3.0), (inward[0] - 3.0, inward[1] + 3.0), (inward[0] + 4.0, inward[1] + 4.0), (inward[0] - 4.0, inward[1] - 4.0)]
    for k, (nf, nh) in enumerate(views):                                           # up to five photos, until three agree
        if len(Ds) >= 3:
            break
        q, img = point(h_hover, tilt, PSIS[0], GRIP_OPEN, (nf, nh))
        r = photo(f"{SQ}_hover{tag}_2_hover_{k}", img, q, pawn)
        if r is None:
            continue
        if r.get("jump_mm", 0) > 10:                    # the hover photo's own pawn reading disagrees with the top-down one: wrong labels
            print(f"   hover photo {k}: labels off ({r['jump_mm']:.0f} mm from the top-down pawn); dropped"); continue
        al, ac, ax_fh = pawn_vs_site(r, q, img, pawn)
        # the hover's own nudge moved the site: refer everything to the un-nudged site
        a_fh = np.array(ax_fh); p_fh = np.array([-a_fh[1], a_fh[0]]); n = np.array([nf / (SP.get("scale_rank") or 1), nh / (SP.get("scale_file") or 1)])
        al, ac = al + float(n @ a_fh), ac + float(n @ p_fh)
        if Ds and np.hypot(al - np.median([d[0] for d in Ds]), ac - np.median([d[1] for d in Ds])) > 6 and len(Ds) >= 2:
            print(f"   hover photo {k}: {al:+.1f} / {ac:+.1f}, far from the others; dropped"); continue
        Ds.append((al, ac)); print(f"   hover photo {k}: the pawn sits {al:+.1f} mm along / {ac:+.1f} mm across the jaws from the gripper site (un-nudged)")
    pl.cmd("nudge:reset"); time.sleep(0.2); pl.move("go_ready", 60)
    if Ds:
        D = np.mean(np.array(Ds), axis=0); dev = np.array(Ds) - D
        # the model's aim relative to the grasps that worked (e4/a8). There the site was placed at the pawn minus the
        # grasp offsets (GRASP_OFF - 1 mm along, GRIP_ACROSS across) and the pawn then sat at TARGET; here the site
        # is on the model's centre and the pawn wherever it stands (P): the same aim error is
        #     D - P - (TARGET - grasp offsets)
        P_al, P_ac = float(np.array([pawn[1], pawn[0]]) @ a_fh), float(np.array([pawn[1], pawn[0]]) @ p_fh)
        d_al = D[0] - P_al - 1000 * (TARGET_ALONG - (GRASP_OFF - 0.001)); d_ac = D[1] - P_ac - 1000 * (TARGET_ACROSS - GRIP_ACROSS)
        delta = d_al * a_fh + d_ac * p_fh                                             # (far, h) mm: move the model's centre by this
        out = dict(square=SQ, pawn_true_h_r1=pawn, D_along_across=D.tolist(), spread=np.abs(dev).max(0).tolist(), n=len(Ds),
                   axis_far_h=a_fh.tolist(), aim_error_along_across=[d_al, d_ac], model_shift_far_h=delta.tolist(), session=sess)
        json.dump(out, open(f"{OUT}/hover_{SQ}.json", "w"), indent=1)
        print(f"{SQ}: the pawn is {D[0]:+.1f} mm along / {D[1]:+.1f} mm across the jaws from the site at the hover ({len(Ds)} photos, spread {np.round(np.abs(dev).max(0), 1).tolist()}); "
              f"vs the grasps that worked: {d_al:+.1f} / {d_ac:+.1f} -> the model's centre should move {delta[0]:+.1f} mm toward rank 1, {delta[1]:+.1f} mm toward h")
    print("done"); sys.exit(0)
if WHERE:
    tag = sys.argv[sys.argv.index("--where") + 1] if len(sys.argv) > sys.argv.index("--where") + 1 else "now"
    q0, img0 = point(h_hover, tilt, PSIS[0], GRIP_OPEN); coarse = photo(f"{SQ}_where_{tag}_hover", img0, q0)
    r = photo_over(f"{SQ}_where_{tag}_topdown", (coarse["toward_h"], coarse["toward_r1"]) if coarse else None)
    pl.cmd("nudge:reset"); time.sleep(0.2); pl.move("go_ready", 60); print("done"); sys.exit(0)
for n_psi, psi in enumerate(PSIS):
    k0 = 7 * n_psi; cyc[psi] = cyc.get(psi, 0) + 1; cy = f"c{cyc[psi]}"
    q_test, _ = Pl.ik_near([c[0], c[1], Pl.z0 + h_hover], tilt, psi, r6, GRIP_OPEN)
    if q_test is None:
        print(f"psi {psi:.0f}: no IK at the hover; skipped"); continue
    # 1. hover, jaws open, photo: where is the pawn relative to the true square?
    prog(k0 + 1, NSTEP, f"psi {psi:.0f}: hover photo"); print(f"psi {psi:.0f}:")
    if DRY:                                        # two different camera poses over the same pawn: a consistency check
        CAM = cam_pose(c, (0.15,) if n_psi == 0 else (0.165,)); print(f"   dry: camera {150 if n_psi == 0 else 165} mm up, site {1000 * CAM[1]:.0f} mm up, axis {CAM[4]:.1f} deg from vertical")
    qA0, imgA0 = point(h_hover, tilt, psi, GRIP_OPEN)                       # coarse position first (the model's labels are reliable here)
    coarse = photo(f"{SQ}_roll{psi:.0f}{cy}_0_hover_coarse", imgA0, qA0, last)
    offA = photo_over(f"{SQ}_roll{psi:.0f}{cy}_1_topdown_before_grasp", (coarse["toward_h"], coarse["toward_r1"]) if coarse else last)
    qA, imgA = point(h_hover, tilt, psi, GRIP_OPEN)
    g_empty = grip(GRIP_SHUT, 2.5); g_open = grip(GRIP_OPEN, 1.5); qA, imgA = raw()      # the jaws shut on nothing: the reference reading
    print(f"   jaws shut on nothing read {g_empty:.1f} %, open {g_open:.1f} %")
    if offA is None:
        print("   stopping"); break
    pawn_A = (offA["toward_h"], offA["toward_r1"]); last = pawn_A
    if DRY:
        results.append(dict(square=SQ, psi=psi, dry=True, pawn_before=offA)); continue
    if offA is None:
        print("   stopping"); break
    # 2.-3. grasp: the FIXED pad GRASP_OFF from the pawn's axis along the opening axis (the moving jaw sweeps the
    #    pawn against it; the fixed jaw is rigid, 1 mm on the -z side of the site; the twin's moving jaw is not
    #    trusted), the pads centred on the pawn across it: the pads sit GRIP_ACROSS from the site's plane, and when
    #    that still misses (the pawn is pushed sideways, not held) the across aim is shifted 4 then 8 mm each way,
    #    the pawn re-measured before every try. The shift that holds is itself a coarse across error measurement.
    held = False; g_closed = None; tries = []
    for attempt, bias in enumerate([0.0, 0.004, -0.004, 0.008, -0.008]):
        if attempt:
            offA = photo_over(f"{SQ}_roll{psi:.0f}{cy}_1_topdown_before_grasp_try{attempt + 1}", last)
            if offA is None or np.hypot(offA["toward_h"], offA["toward_r1"]) > 20:
                print("   lost the pawn (not found, or measured off the square); stopping"); offA = None; break
            last = (offA["toward_h"], offA["toward_r1"])
        a_open, mid_off, gap, _ = gap_geom(np.r_[qA[:5], g_open]); a_x = np.array([-a_open[1], a_open[0]])   # across, in the world
        far_c, h_c = A.to_board_axes(-(GRASP_OFF - 0.001) * a_open[:2] - (GRIP_ACROSS + bias) * a_x, sess)
        far_p, h_p = real_to_arm(offA["toward_r1"], offA["toward_h"])
        nudge = (far_c + far_p, h_c + h_p)
        print(f"   try {attempt + 1}: opening axis {np.round(a_open[:2], 2).tolist()}; nudge {np.round(nudge, 1).tolist()} (far, h): the fixed pad "
              f"{1000 * GRASP_OFF:.0f} mm from the pawn's axis, pads {1000 * (GRIP_ACROSS + bias):+.1f} mm across")
        prog(k0 + 2, NSTEP, f"psi {psi:.0f}: over the pawn (try {attempt + 1})"); q2, img2 = point(h_hover, tilt, psi, GRIP_OPEN, nudge)
        servo = []
        for it in range(2):                                  # visual aim: the pawn must sit at TARGET (along, across) of the site in the photo
            r2 = photo(f"{SQ}_roll{psi:.0f}{cy}_2_hover_over_pawn_jaws_open_try{attempt + 1}_{it}", img2, q2, last)
            if r2 is None:
                break
            al, ac, _ = pawn_vs_site(r2, q2, img2)
            d_al, d_ac = al - 1000 * TARGET_ALONG, ac - 1000 * TARGET_ACROSS
            print(f"   visual aim: the pawn is {al:+.1f} mm along / {ac:+.1f} mm across the jaws from the site; off the target by {d_al:+.1f} / {d_ac:+.1f}")
            servo.append([d_al, d_ac])
            if np.hypot(d_al, d_ac) < 1.5:
                break
            if np.hypot(d_al, d_ac) > 15:                    # not a real aim error: a wrong pawn or wrong labels in an oblique photo
                print("   (ignored: too large to be true)"); break
            d_al, d_ac = 0.6 * d_al, 0.6 * d_ac             # damped: the oblique measurement is noisy
            a_open, _, _, _ = gap_geom(q2); a_x = np.array([-a_open[1], a_open[0]])
            far_s, h_s = A.to_board_axes(0.001 * d_al * a_open[:2] + 0.001 * d_ac * a_x, sess)
            far_s, h_s = real_to_arm(far_s, h_s)
            nudge = (nudge[0] + far_s, nudge[1] + h_s)
            q2, img2 = point(h_hover, tilt, psi, GRIP_OPEN, nudge)
            if np.hypot(d_al, d_ac) < 6:
                break
        prog(k0 + 3, NSTEP, f"psi {psi:.0f}: grasp (try {attempt + 1})"); q_low, _ = point(TIPS_H + 0.0063, tilt, psi, GRIP_OPEN, nudge, 40)
        _, _, _, z_tip = gap_geom(q_low); print(f"   at the grasp height: fingertips {1000 * (z_tip - s.cfg.board_thickness - dz):.1f} mm above the real board (FK)")
        g_closed = grip(GRIP_SHUT, 3.0)
        qC, imgC = point(h_hover, tilt, psi, GRIP_SHUT, nudge, 40)
        print(f"   shut: jaws read {g_closed:.1f} % (empty {g_empty:.1f}; gap {1000 * gap_geom(np.r_[qC[:5], g_closed])[2]:.1f} mm by the twin)")
        still = pv.blob_near(imgC, qC, SQ, sess, last); cv2.imwrite(f"{OUT}/{SQ}_roll{psi:.0f}{cy}_3_hover_after_grasp_lifted_try{attempt + 1}.jpg", cv2.cvtColor(imgC, cv2.COLOR_RGB2BGR))
        held = still is None
        where = "empty" if still is None else f"still occupied ({still:.1f} mm from before)"
        print(f"   the pawn's spot is {where} -> {'HELD' if held else 'NOT held'}" + ("" if g_closed >= g_empty + 0.8 or not held else " (though the jaws read as empty: the photos decide)"))
        tries.append(dict(bias_mm=1000 * bias, held=held, closed_pct=g_closed, visual_aim_corrections_mm=servo, nudge=list(nudge)))
        if held:
            break
        print("   not holding the pawn: opening, re-measuring, trying another across aim"); grip(GRIP_OPEN, 1.5)
        point(h_hover, tilt, psi, GRIP_OPEN, nudge, 40)
    if not held:
        print("   no grasp after all the tries; please re-centre the pawn")
        results.append(dict(square=SQ, psi=psi, held=False, pawn_before=offA, tries=tries)); break
    last = (offA["toward_h"], offA["toward_r1"])
    # 4. place with the SITE on the model's centre, no jaw compensation: the pawn lands at (fixed pad + its body's
    #    radius) along the opening axis, the same offset at every roll, which the opposite rolls cancel
    a_open, mid_c, gap_c, _ = gap_geom(np.r_[qC[:5], g_closed])
    nudge_p = (0.0, 0.0)
    print(f"   placing with the site on the model's centre (twin: shut gap {1000 * gap_c:.1f} mm)")
    prog(k0 + 4, NSTEP, f"psi {psi:.0f}: carry"); point(h_hover, tilt, psi, GRIP_SHUT, nudge_p, 40)
    prog(k0 + 5, NSTEP, f"psi {psi:.0f}: place"); q_pl, _ = point(TIPS_H + 0.0063, tilt, psi, GRIP_SHUT, nudge_p, 40)
    grip(GRIP_OPEN, 2.5)
    prog(k0 + 6, NSTEP, f"psi {psi:.0f}: photo after"); point(h_hover, tilt, psi, GRIP_OPEN, nudge_p, 40)
    offB = photo_over(f"{SQ}_roll{psi:.0f}{cy}_4_topdown_after_placement", (0.0, 0.0))             # the pawn should now stand near the true centre
    far_a, h_a = A.to_board_axes(a_open[:2] / np.linalg.norm(a_open[:2]), sess)
    rec = dict(square=SQ, psi=psi, held=True, pawn_before=offA, closed_pct=g_closed, empty_pct=g_empty, open_pct=g_open,
               gap_shut_mm=1000 * gap_c, mid_shut_mm=1000 * mid_c, after=offB, open_axis_far_h=[far_a / 1000, h_a / 1000], tries=tries)
    results.append(rec)
    if offB is None:
        print("   no measurement after the placement"); continue
    last = (offB["toward_h"], offB["toward_r1"])
    print(f"   PLACED {offB['toward_h']:+.1f} mm toward h, {offB['toward_r1']:+.1f} mm toward rank 1 of the true centre: where the robot thinks "
          f"the centre is (the jaws' midpoint bias acts along (far {far_a / 1000:+.2f}, h {h_a / 1000:+.2f}))")
    prog(k0 + 7, NSTEP, f"psi {psi:.0f}: done")
pl.cmd("nudge:reset"); time.sleep(0.2)
if not DRY:
    pl.move("go_ready", 60)
json.dump(results, open(f"{OUT}/results_{SQ}.json", "w"), indent=1)
done = [r for r in results if r.get("after")]
if done:
    # the pawn lands at (model centre's true position) + f' * axis, f' = fixed pad -> pawn axis (same at every roll):
    # along_i = e . axis_i + f'; the across component should be ~0 (the pawn keeps its across position in the jaws)
    Amat = []; y = []
    for r in done:
        ax = np.array(r["open_axis_far_h"]); m = np.array([r["after"]["toward_r1"], r["after"]["toward_h"]])
        along, across = float(m @ ax), float(m @ np.array([-ax[1], ax[0]]))
        r["along_mm"], r["across_mm"] = along, across
        print(f"   psi {r['psi']:.0f}: placed {along:+.1f} mm along the opening axis (far {ax[0]:+.2f}, h {ax[1]:+.2f}), {across:+.1f} across")
        Amat.append([ax[0], ax[1], 1.0]); y.append(along)
    rolls = sorted(set(r["psi"] for r in done))
    if len(rolls) == 1:
        M = np.array([[r["after"]["toward_r1"], r["after"]["toward_h"]] for r in done]); mean = M.mean(0); sd = M.std(0, ddof=1) if len(M) > 1 else np.zeros(2)
        print(f"\n{SQ} at roll {rolls[0]:.0f} ({len(M)} cycle{'s' if len(M) > 1 else ''}): a pawn put down with the site on the model's centre lands "
              f"{mean[0]:+.1f} mm toward rank 1, {mean[1]:+.1f} mm toward h of the true centre (spread {sd[0]:.1f} / {sd[1]:.1f} mm)")
        json.dump(dict(square=SQ, psi=rolls[0], err_far_mm=float(mean[0]), err_h_mm=float(mean[1]), sd_far_mm=float(sd[0]), sd_h_mm=float(sd[1]),
                       n=len(M), results=results), open(f"{OUT}/results_{SQ}.json", "w"), indent=1)
    elif len(done) >= 3:
        sol, res, *_ = np.linalg.lstsq(np.array(Amat, float), np.array(y), rcond=None)
        fit = np.array(Amat) @ sol
        print(f"\n{SQ}: calibration error {sol[0]:+.1f} mm toward rank 1, {sol[1]:+.1f} mm toward h (real mm; the model's centre is that far from the "
              f"true one); fixed pad -> pawn axis {sol[2]:+.1f} mm; fit residuals {np.round(np.array(y) - fit, 1).tolist()} mm")
        json.dump(dict(square=SQ, err_far_mm=float(sol[0]), err_h_mm=float(sol[1]), pad_mm=float(sol[2]), results=results), open(f"{OUT}/results_{SQ}.json", "w"), indent=1)
prog(NSTEP, NSTEP, "done"); print("done")
