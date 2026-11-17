"""Scene tests (plain asserts).  Run from the project root:
    python chess_sim/tests/test_scene.py
Tests (h), (i) build per-config scenes in a temporary directory and run the watch-pose search with its DEFAULT
budget (progress bar in chess_sim/data/progress.txt); the whole file takes ~1-2 minutes.
"""
from __future__ import annotations

import atexit
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import mujoco
import chess

from chess_sim import config as C
from chess_sim.progress import ProgressBar
from chess_sim.scene import ChessScene, SceneConfigMismatch, START_FEN
from chess_sim.build_scene import (write_scene, embedded_config, embedded_jaw_geometry, measure_jaw_geometry, jaw_geometry_agrees,
                                   load_piece_bands, load_obj, piece_mesh_path, section_points, caliper_width, width_profile,
                                   grasp_table, measure_axis_radii, _mesh_world_vertices, GRASP_AXIS_LOCAL_DEG)
from chess_sim.kinematics import Kinematics

SCENE: ChessScene | None = None
TMP = tempfile.mkdtemp(prefix="chess_sim_test_")      # removed by main() and at interpreter exit (ad-hoc imports)
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
SEARCH_PROGRESS = os.path.join(TMP, "search_progress.txt")   # watch-pose searches of the tests write their bar here


def scene() -> ChessScene:
    global SCENE
    if SCENE is None:
        SCENE = ChessScene()
    return SCENE


BANDS_SCENE: ChessScene | None = None


def bands_scene() -> ChessScene:
    """The mesh + 'bands' collision scene (its own scene/chess_scene_<hash>.xml, generated when missing)."""
    global BANDS_SCENE
    if BANDS_SCENE is None:
        BANDS_SCENE = ChessScene(C.ChessConfig(piece_collision="bands"))
        BANDS_SCENE.reset(seed=0, randomise=False)
    return BANDS_SCENE


def luminance(rgb) -> float:
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


def corners_inside(s: ChessScene, margin_frac: float = 0.0) -> bool:
    """The watch region (cfg.watch_region of the live board) inside the usable wrist-image window (+ margin)."""
    return s.watch_target_visible(margin_frac)


def check_piece_bodies(s: ChessScene) -> None:
    """The 34 piece bodies of a scene: masses = spec, collision geoms per cfg.piece_collision ('cylinders': base +
    stem cylinders with the spec dimensions; 'bands': one convex mesh hull per band of assets/pieces/bands, the
    stack spanning the piece height), visuals per cfg.piece_visual (group 1, no collision; 'mesh': one mesh geom
    with the manifest's triangle count)."""
    m = s.model
    bands = load_piece_bands()
    for n in s.piece_names:
        t = s.piece_type[n]
        H, D, mass = C.PIECE_SPECS[t]
        assert abs(m.body_mass[s.piece_body_id[n]] - mass * s.cfg.piece_mass_scale) < 1e-6, (n, m.body_mass[s.piece_body_id[n]])
        col = [g for g in range(m.ngeom) if m.geom_bodyid[g] == s.piece_body_id[n] and m.geom_contype[g] != 0]
        assert all(m.geom_group[g] == 3 and abs(m.geom_friction[g, 0] - 1.0) < 1e-9 and m.geom_condim[g] == 4 for g in col), n
        if s.cfg.piece_collision == "cylinders":
            gb, gs = m.geom(f"{n}_col_base"), m.geom(f"{n}_col_stem")
            assert len(col) == 2 and {gb.id, gs.id} == set(col)
            height = max(gb.pos[2] + gb.size[1], gs.pos[2] + gs.size[1])
            assert abs(height - H) < 0.001, (n, height, H)
            assert abs(2 * gb.size[0] - D) < 0.001, (n, 2 * gb.size[0], D)
            assert abs(2 * gs.size[0] - 0.6 * D) < 0.001
            assert abs(2 * gb.size[1] - 0.3 * H) < 0.001
        else:
            assert bands is not None, "assets/pieces/bands/bands.json missing"
            rec = bands["pieces"][t]
            assert len(col) == rec["n_bands"] == len(rec["bands"]) and 4 <= len(col) <= C.BAND_MAX_COUNT, (n, len(col))
            assert all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in col)
            assert abs(rec["cuts"][0]) < 1e-9 and abs(rec["cuts"][-1] - H) < 2e-4, rec["cuts"]
            for b in rec["bands"]:
                g = m.geom(f"{n}_col_band{b['index']}").id
                assert g in col
                mid = m.geom_dataid[g]
                assert m.mesh_facenum[mid] == b["hull_faces"] and m.mesh_vertnum[mid] == b["hull_vertices"], (n, b["index"])
        vis = [g for g in range(m.ngeom) if m.geom(g).name.startswith(n + "_vis")]
        assert all(m.geom_group[g] == 1 and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in vis), n
        if s.cfg.piece_visual == "mesh":
            assert len(vis) == 1 and m.geom(vis[0]).name == f"{n}_vis_mesh" and m.geom_type[vis[0]] == mujoco.mjtGeom.mjGEOM_MESH
            mid = m.geom_dataid[vis[0]]
            assert m.mesh_facenum[mid] == C.PIECE_MESH_TRIANGLES[t], (n, m.mesh_facenum[mid], C.PIECE_MESH_TRIANGLES[t])
            assert m.mesh_vertnum[mid] == C.PIECE_PROFILE[t]["vertices"]
            # the mesh stands on the piece base (z_min = 0) and reaches the spec height; the material is the colour's
            v = _mesh_world_vertices(m, s.data, vis[0]) - s.piece_pos(n)
            assert abs(v[:, 2].min()) < 2e-4 and abs(v[:, 2].max() - H) < 2e-4, (n, v[:, 2].min(), v[:, 2].max())
            mat = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, m.geom_matid[vis[0]])
            assert mat == ("piece_white" if n.startswith("w_") else "piece_black"), (n, mat)
        else:
            assert len(vis) >= 3


# ---------------------------------------------------------------- (a) dimensions read back from the compiled model
def test_a_dimensions():
    s = scene()
    m = s.model
    box = m.geom("board_box")
    play = 2 * box.size[0] - 2 * C.BOARD_BORDER
    assert abs(play - 0.224) <= 0.001, play
    assert abs(2 * box.size[2] - s.cfg.board_thickness) < 1e-6
    # square pitch from square_xyz
    e4, e5, d4 = s.square_xyz("e4"), s.square_xyz("e5"), s.square_xyz("d4")
    assert abs(np.linalg.norm(e4 - e5) - 0.028) < 1e-9
    assert abs(np.linalg.norm(e4 - d4) - 0.028) < 1e-9
    assert e5[0] < e4[0], "rank 5 must be farther from the robot than rank 4"
    assert d4[1] < e4[1], "file d must be on the robot's right (-y) of file e"
    assert abs(s.square_xyz("a8")[0] - (s.cfg.board_gap + 0.014)) < 1e-9 and abs(s.square_xyz("a8")[1] + 0.098) < 1e-9
    assert abs(s.square_xyz("a1")[2] - s.cfg.board_thickness) < 1e-9
    # the corner sites of the compiled model agree with the config formula
    a8 = s.square_xyz("a8") + np.array([-0.014, -0.014, 0])
    assert np.allclose(s.data.site_xpos[s.corner_sites["corner_a8"]], a8, atol=1e-6)
    # 34 free bodies, heights / base diameters from collision geoms, masses
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(free) == 34, len(free)
    assert len(s.piece_names) == 34
    assert s.cfg.piece_visual == "mesh" and s.cfg.piece_collision == "cylinders", "the default scene: mesh skins over the two-cylinder collision model"
    check_piece_bodies(s)
    assert abs(m.opt.timestep - 0.002) < 1e-12
    assert m.ncam == 3 and all(n in [m.camera(i).name for i in range(m.ncam)] for n in C.CAMERA_NAMES)
    assert abs(m.cam_fovy[s.cam_ids[C.WRIST_CAM]] - s.cfg.wrist_fovy) < 1e-9
    # drop-zone / spare cells
    assert np.allclose(s.drop_cell_xyz(0, 0)[:2], (s.cfg.board_gap - 0.010 + 0.014, -(0.112 + 0.046 + 0.014)))
    assert np.allclose(s.spare_xyz("w")[:2], (s.cfg.board_gap + 0.014, +(0.112 + 0.046 + 0.014)))
    # unit helpers (gripper percent end points = the real arm's measured 7 / 80 mm fingertip gaps)
    assert C.GRIPPER_PERCENT_MODE == "measured_gap"
    assert abs(C.gripper_gap_for_percent(0) - 0.007) < 1e-9 and abs(C.gripper_gap_for_percent(100) - 0.080) < 1e-9
    assert abs(C.gripper_gap(C.gripper_ctrl_for_gap(0.045)) - 0.045) < 1e-9
    assert abs(C.gripper_clearance(C.gripper_ctrl_for_clearance(0.012)) - 0.012) < 1e-9
    assert abs(C.gripper_frac_to_percent(C.gripper_percent_to_frac(37.0)) - 37.0) < 1e-9
    assert C.GRIPPER_Q_CLOSED <= C.GRIPPER_Q_PCT0 < C.GRIPPER_Q_PCT100 <= C.GRIPPER_Q_OPEN
    v = np.array([10, -20, 30, -40, 50, 60.0])
    assert np.allclose(C.qpos_to_lerobot(C.lerobot_to_qpos(v)), v)
    assert C.CONTROL_HZ == 30 and C.SUBSTEPS == 17 and abs(C.SUBSTEPS_MEAN - 50 / 3) < 1e-9 and C.CONTROL_HZ_EFFECTIVE == 30.0
    # the XML records the config it was built from
    assert embedded_config(m) == s.cfg.scene_key() == C.DEFAULT_SCENE_KEY
    assert s.xml_path == C.SCENE_XML == C.DEFAULT_CONFIG.scene_xml_path()


# ---------------------------------------------------------------- (a2) piece models: envelope, grasp table, band accuracy
def test_a2_piece_models():
    """The mesh assets against the manifest and config: triangle counts, heights / foot diameters, the analytic
    envelope (axis radius per slice) contains every mesh vertex, the 'bands' grasp table equals a fresh derivation
    from the meshes + finger boxes (and agrees with the manifest's stem profile at the grasp heights), and the
    band hulls follow the real profile within 1 mm (10 heights per piece vs the manifest slices, every 0.5 mm vs
    the mesh with a 0.5 mm step tolerance)."""
    s = bands_scene()
    m = s.model
    assert s.cfg.piece_collision == "bands" and s.cfg.piece_visual == "mesh"
    check_piece_bodies(s)
    bands = load_piece_bands()
    assert bands is not None and set(bands["pieces"]) == set(C.PIECE_TYPES)
    jg = measure_jaw_geometry()
    assert "table_levels" in jg and np.array(jg["table_levels"]).shape == (len(jg["table_q"]), 3)
    fresh = grasp_table(jg=jg)
    worst_env = worst_band10 = worst_band = 0.0
    n_under = 0
    for t in C.PIECE_TYPES:
        V, F = load_obj(piece_mesh_path(t))
        H, D, _ = C.PIECE_SPECS[t]
        assert len(F) == C.PIECE_MESH_TRIANGLES[t] == C.PIECE_PROFILE[t]["triangles"], (t, len(F))
        assert abs(V[:, 2].max() - H) < 2e-4 and abs(V[:, 2].min()) < 1e-6, (t, V[:, 2].max())
        foot = V[V[:, 2] <= 0.2 * H][:, :2]
        assert abs(caliper_width(foot) - D) < 3e-4, (t, caliper_width(foot))
        # envelope: the config table equals a fresh measurement (vertices + exact cuts) and every vertex AND every
        # planar cut of each slice (every 0.25 mm) is inside the slice's radius (knight: head / mane beyond the half
        # width; sparse cones: the cuts are wider than any vertex of the slice)
        fresh_r = measure_axis_radii(t)
        assert all(abs(a - b) < 0.011 for a, b in zip(C.PIECE_AXIS_RADIUS_MM[t], fresh_r)), (t, C.PIECE_AXIS_RADIUS_MM[t], fresh_r)
        for z0, z1, r in C.PIECE_ENVELOPE[t]:
            sel = V[(V[:, 2] >= z0 - 1e-9) & (V[:, 2] <= z1 + 1e-9)]
            pts = [sel[:, :2]] if len(sel) else []
            pts += [sec[:, :2] for sec in (section_points(V, F, z) for z in np.arange(z0 + 0.000125, z1, 0.00025)) if len(sec)]
            excess = float(max(np.linalg.norm(p, axis=1).max() for p in pts) - r)
            worst_env = max(worst_env, excess)
            assert excess <= 1e-5, (t, z0, excess)
        assert C.piece_height_within(t, 0.0, "bands") == H and C.piece_height_within(t, 1.0, "bands") == 0.0
        assert abs(C.piece_max_radius(t, 0.0, "bands") * 2 - max(2 * r for _, _, r in C.PIECE_ENVELOPE[t])) < 1e-9
        assert C.piece_height_within(t, 0.0, "cylinders") == H and abs(C.piece_radius_at(t, 0.5 * H, "cylinders") - 0.3 * D) < 1e-9
        # grasp table == fresh derivation (rule in build_scene.grasp_table), zone contains the height
        rec, ref = C.GRASP_TABLE_BANDS[t], fresh[t]
        for key in ("height", "width", "envelope", "zone_lo", "zone_hi"):
            assert abs(rec[key] - ref[key]) < 1e-5, (t, key, rec[key], ref[key])
        assert rec["axis_deg"] == ref["axis_deg"] == GRASP_AXIS_LOCAL_DEG.get(t)
        assert rec["zone_lo"] - 1e-9 <= rec["height"] <= rec["zone_hi"] + 1e-9 and 0.3 * H <= rec["height"] <= 0.7 * H
        if rec["axis_deg"] is not None:
            assert len(rec["extent_mm"]) == 36 and all(abs(a - b) < 0.011 for (a, _), (b, _) in zip(rec["extent_mm"], ref["extent_mm"]))
        # the stall width is a width of the manifest's stem profile / slices inside the pad window (within 0.5 mm)
        p = C.PIECE_PROFILE[t]
        lo, hi = rec["height"], rec["height"] + 0.010
        man = [w for z, w in p["stem"] if lo - 1e-9 <= z <= hi + 1e-9]
        if rec["axis_deg"] is None:
            stem_top = p["stem"][-1][0]
            man += [w for z0, z1, w in p["slices"] if z1 > stem_top and z0 < hi]   # slice maxima above the stem profile
            in_stem = [w for z, w in p["stem"] if lo - 1e-9 <= z <= min(hi, stem_top) + 1e-9]
            assert man and rec["width"] <= max(man) + 5e-4, (t, rec["width"], max(man))
            assert not in_stem or rec["width"] >= max(in_stem) - 5e-4, (t, rec["width"], max(in_stem))
        else:
            ys = [w for z0, z1, w in p["slices_y"] if z1 > lo and z0 < hi]
            assert abs(rec["width"] - max(ys)) < 1e-3, (t, rec["width"], ys)
        # band hulls vs the real profile (the mesh's exact planar cuts, the geometry the manifest was measured from):
        # at the 10 manifest slice centres the hull is at most 1 mm wider than the exact cut; every 0.5 mm it is at
        # most 1 mm wider than the exact cut or the wider section 0.5 mm above it (an undercut: no convex band can
        # be narrower than the widest section above), and never narrower than the mesh (it contains it).  Against
        # the manifest's 10 slices: the hull's widest cut in a slice is >= the manifest's value (a VERTEX maximum:
        # it under-reports a sparse cone, e.g. the queen's 16-20 mm slice 7.98 mm vs the exact 9.87) and <= the
        # exact slice maximum + 1 mm
        zs, ws = width_profile(V, F, step=0.0005)
        n = f"w_{t}_1"
        p0 = s.piece_pos(n)
        hulls = [(m.geom(f"{n}_col_band{b['index']}").id, b) for b in bands["pieces"][t]["bands"]]
        verts = {g: _mesh_world_vertices(m, s.data, g) - p0 for g, _ in hulls}
        faces = {g: m.mesh_face[m.mesh_faceadr[m.geom_dataid[g]]:m.mesh_faceadr[m.geom_dataid[g]] + m.mesh_facenum[m.geom_dataid[g]]] for g, _ in hulls}

        def hull_width(z):
            out = 0.0
            for g, b in hulls:
                for zz in (z - 2e-6, z + 2e-6):      # a sample exactly on a cut plane would miss the hull's cap polygon
                    if b["z0"] - 1e-9 <= zz <= b["z1"] + 1e-9:
                        sec = section_points(verts[g], faces[g], zz)
                        if len(sec):
                            out = max(out, caliper_width(sec[:, :2]))
            return out
        for z0, z1, w in p["slices"]:
            zc = (z0 + z1) / 2
            hw = hull_width(zc)
            real_c = caliper_width(section_points(V, F, zc)[:, :2])
            worst_band10 = max(worst_band10, hw - real_c)
            assert real_c - 0.0003 <= hw <= real_c + 0.001, (t, zc, hw, real_c)
            msl = (zs >= z0 - 1e-9) & (zs <= z1 + 1e-9)
            sel = V[(V[:, 2] >= z0 - 1e-9) & (V[:, 2] <= z1 + 1e-9)]
            z_wide = float(sel[np.linalg.norm(sel[:, :2], axis=1).argmax(), 2])      # height of the slice's widest vertex
            heights = list(zs[msl]) + [z_wide]
            exact_max = max(float(ws[msl].max()), caliper_width(sel[:, :2]))        # the 0.5 mm cuts + the slice's vertices
            hull_max = max(hull_width(z) for z in heights)
            n_under += exact_max > w + 0.0005
            assert w <= exact_max + 1e-4, (t, z0, w, exact_max)              # the manifest's vertex maximum never exceeds the cuts
            # (the manifest's value is a pairwise VERTEX distance, for the knight between two heights, so the lower
            # bound is the widest planar cut of the slice; the upper bound the exact slice maximum + 1 mm)
            assert float(ws[msl].max()) - 0.0003 <= hull_max <= exact_max + 0.001, (t, z0, hull_max, w, exact_max)
        for i, z in enumerate(zs):
            if z < 0.0005 or z > H - 0.0005:
                continue
            ref_w = ws[i:i + 2].max()                  # the exact width, or the wider section 0.5 mm above (undercut)
            hw = hull_width(z)
            worst_band = max(worst_band, hw - ref_w)
            assert hw <= ref_w + 0.001, (t, z, hw, ref_w)
            assert hw >= ws[i] - 0.0003, (t, z, hw, ws[i])          # the hull contains the mesh
    print(f"    6 meshes: triangles / heights / foot diameters as in the manifest; envelope excess over the vertices and cuts {worst_env*1e6:.1f} um; "
          f"grasp table == fresh derivation; {sum(p['n_bands'] for p in bands['pieces'].values())} band hulls, hull width - real width: "
          f"max {worst_band10*1000:.2f} mm at the 10 slice centres, {worst_band*1000:.2f} mm every 0.5 mm (limit 1 mm); "
          f"{n_under}/60 manifest slices under-report the exact cuts by > 0.5 mm (vertex maxima)")
    print("    grasp table (tip height / stall width / descent envelope, mm): " +
          ", ".join(f"{t} {C.GRASP_TABLE_BANDS[t]['height']*1000:.1f}/{C.GRASP_TABLE_BANDS[t]['width']*1000:.1f}/{C.GRASP_TABLE_BANDS[t]['envelope']*1000:.1f}" for t in C.PIECE_TYPES))


# ---------------------------------------------------------------- (a3) the other piece models still build; mesh vs primitive renders
def test_a3_piece_model_switches():
    """ChessConfig(piece_collision='cylinders', piece_visual='primitive') builds its own scene (per-config XML) with the
    two-cylinder collision model and the primitive visuals, a 'cylinders' + 'mesh' mix too; the mesh skins render
    differently from the primitives (pixel statistics of the overhead view) and the watch pose is shared."""
    s = scene()
    assert C.PIECE_COLLISION_DEFAULT == "cylinders" and C.PIECE_VISUAL_DEFAULT == "mesh"
    cfg_c = C.ChessConfig(piece_collision="cylinders", piece_visual="primitive")
    assert cfg_c.scene_xml_path() != C.SCENE_XML and cfg_c.watch_key() == s.cfg.watch_key()
    s_c = ChessScene(cfg_c, xml_path=os.path.join(TMP, "chess_scene_primitive.xml"))
    check_piece_bodies(s_c)
    assert C.stem_diameter("king", "cylinders") == 0.012 and abs(C.grasp_height("king", "cylinders") - 0.031) < 1e-9
    assert abs(C.stem_diameter("king") - 0.012) < 1e-9 and abs(C.stem_diameter("king", "bands") - C.GRASP_TABLE_BANDS["king"]["width"]) < 1e-9
    assert abs(C.grasp_height("king", "bands") - C.GRASP_TABLE_BANDS["king"]["height"]) < 1e-9
    assert C.grasp_dz_candidates("king") == C.GRASP_DZ_CYLINDERS and C.grasp_dz_candidates("queen", "bands") == (0.0, C.GRASP_DZ_LAST_RESORT)
    assert C.grasp_dz_tier("queen", C.GRASP_DZ_LAST_RESORT, "bands") == 1 and C.grasp_dz_tier("rook", 0.0016, "bands") == 0
    s_b = bands_scene()
    assert s_b.cfg.scene_xml_path() != C.SCENE_XML and s_b.xml_path == s_b.cfg.scene_xml_path() and s_b.cfg.watch_key() == s.cfg.watch_key()
    check_piece_bodies(s_b)
    nb = {t: load_piece_bands()["pieces"][t]["n_bands"] for t in C.PIECE_TYPES}
    assert s_b.model.ngeom - s.model.ngeom == sum(C.PIECE_COUNTS[t] * 2 * (nb[t] - 2) for t in C.PIECE_TYPES) + 2 * (nb["queen"] - 2)
    try:
        C.ChessConfig(piece_collision="spheres").scene_key()
        raise AssertionError("bad piece_collision accepted")
    except ValueError:
        pass
    # renders: same position, same arm pose; the mesh and primitive images differ substantially on the pieces
    imgs = {}
    for name, sc in (("mesh", s), ("primitive", s_c)):
        sc.reset(seed=0, randomise=False)
        imgs[name] = sc.render(C.OVERHEAD_CAM).astype(int)
    diff = np.abs(imgs["mesh"] - imgs["primitive"]).max(axis=2)
    frac = float((diff > 40).mean())
    print(f"    primitive scene {os.path.basename(s_c.xml_path)}: ngeom {s_c.model.ngeom}; default mesh/cylinders {s.model.ngeom}; mesh/bands {s_b.model.ngeom}; "
          f"overhead render: {frac*100:.1f} % of the pixels differ by > 40 (mesh vs primitive pieces), mean |diff| {diff.mean():.1f}")
    assert 0.005 < frac < 0.5, frac
    s.reset(seed=0, randomise=False)


# ---------------------------------------------------------------- (b) texture orientation (overhead render, empty board)
def test_b_texture_orientation():
    s = scene()
    s.reset(seed=0, randomise=False, fen=chess.Board(None).fen(), arm_pose=np.array([0, -90, 0, 0, 0, 0.0]), settle_seconds=0)
    img = s.render(C.OVERHEAD_CAM)
    # verify projection against a rendered marker: put the white king on e4 and check the projected top is bright
    def px(sq):
        u, v = s.project(s.square_xyz(sq), C.OVERHEAD_CAM)[0]
        return img[int(round(v)), int(round(u))]
    la1, la8, lh1, lh8 = (luminance(px(q)) for q in ("a1", "a8", "h1", "h8"))
    assert la1 < 110 and lh8 < 110, (la1, lh8)
    assert la8 > 150 and lh1 > 150, (la8, lh1)
    for sq in ("e4", "d4", "c6", "f3"):
        l = luminance(px(sq))
        assert (l < 110) == C.square_is_dark(*C.square_index(sq)), (sq, l)
    # projection check with a marker: the black king's top on e4 should render nearly black, the white king's bright
    s.set_piece_pose("b_king_1", s.square_xyz("e4", 0.0005))
    s.set_piece_pose("w_king_1", s.square_xyz("d5", 0.0005))
    mujoco.mj_forward(s.model, s.data)
    img2 = s.render(C.OVERHEAD_CAM)
    for name, sq, bright in (("b_king_1", "e4", False), ("w_king_1", "d5", True)):
        top = s.piece_pos(name) + np.array([0, 0, 0.040])     # the king's body (sphere/cross region)
        u, v = s.project(top, C.OVERHEAD_CAM)[0]
        l = luminance(img2[int(round(v)), int(round(u))])
        assert (l > 170) == bright, (name, l)
        # sanity: the marker moved the pixel relative to the empty board
        assert abs(l - luminance(img[int(round(v)), int(round(u))])) > 30 or bright


# ---------------------------------------------------------------- (c) stability
def test_c_stability():
    """All 34 pieces stand still for 2 s of physics (moved < 1 mm, tilt < 1 deg) on the default scene (two-cylinder
    collision) and on the mesh 'bands' scene (the hull bands resting on the board)."""
    for s in (scene(), bands_scene()):
        s.reset(seed=0, randomise=False, settle_seconds=0)
        p0 = {n: s.piece_pos(n) for n in s.piece_names}
        s.step(int(2.0 / s.model.opt.timestep))
        worst = max(np.linalg.norm(s.piece_pos(n) - p0[n]) for n in s.piece_names)
        tilt = max(s.piece_tilt_deg(n) for n in s.piece_names)
        assert worst < 0.001, (s.cfg.piece_collision, worst)
        assert tilt < 1.0, (s.cfg.piece_collision, tilt)
        print(f"    stability ({s.cfg.piece_collision}, 34 pieces, 2 s): max drift {worst*1000:.3f} mm, max tilt {tilt:.3f} deg")


# ---------------------------------------------------------------- (d) board_state round trip
def test_d_board_state_roundtrip():
    s = scene()
    s.reset(seed=0, randomise=False, settle_seconds=0)
    st = s.board_state()
    assert len(st) == 32, len(st)
    assert s.to_chess_board().board_fen() == chess.Board().board_fen()
    # a mid-game position with a promoted queen (uses the spare queen) and captured pieces (parked)
    fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1"
    s.set_position(fen)
    assert s.to_chess_board().board_fen() == chess.Board(fen).board_fen()
    fen2 = "QQ6/8/8/8/8/8/8/k6K w - - 0 1"     # two white queens -> w_queen_1 + w_queen_spare_1
    mapping = s.set_position(fen2)
    assert set(mapping.values()) >= {"w_queen_1", "w_queen_spare_1"}
    assert s.to_chess_board().board_fen() == chess.Board(fen2).board_fen()
    assert s.free_drop_cell() == (0, 0)
    s.set_piece_pose("w_pawn_1", s.drop_cell_xyz(0, 0, 0.0005))
    mujoco.mj_forward(s.model, s.data)
    assert s.free_drop_cell() == (1, 0)


# ---------------------------------------------------------------- (e) watch pose (cached for the default config)
def test_e_watch_pose():
    s = scene()
    w = s.watch_pose()                     # strict: raises if the cached/computed pose is infeasible
    assert w.shape == (6,)
    s.reset(seed=0, randomise=False)
    W, H = s.cfg.render_w, s.cfg.render_h
    px = s.project(s.target_corners_at(0, 0, 0), C.WRIST_CAM, W, H)      # the watch region (the first scan view)
    assert not np.any(np.isnan(px))
    assert corners_inside(s, C.WATCH_MARGIN - 1e-6), px
    assert np.all(px[:, 0] <= C.WRIST_IMAGE_WINDOW[1] * W), "the region must stay left of the real jaw"
    info = s._watch
    assert info["feasible"] and not info["occluded"] and not info["collision"], info
    assert info["margin_achieved"] >= 0.05 - 1e-9          # the WORST margin over the board-pose range (see (n))
    assert info["config"] == s.cfg.watch_key()
    # independent re-evaluation of the pose: silhouette + ray-cast occlusion, collision, margin
    r = s.watch_metrics(w, ext_limit=np.inf)
    assert r["feasible"] and r["occ_px"] == 0.0 and r["occluded"] is False and r["collision"] is False, r
    s.reset(seed=0, randomise=False)
    # board occupies >= 15 % of the pixels (by colour: board squares or border wood)
    img = s.render(C.WRIST_CAM).astype(int)
    r_, g, b = img[..., 0], img[..., 1], img[..., 2]
    light = (r_ > 150) & (g > 130) & (b > 80) & (r_ > b + 40) & (r_ >= g)
    dark = (r_ > 60) & (r_ < 160) & (g > 35) & (g < 110) & (b < 90) & (r_ > g + 15) & (g > b + 5)
    frac = float((light | dark).mean())
    print(f"    watch pose {np.round(w, 1)}  corners px {np.round(px).tolist()}  board pixels {frac*100:.1f}%  area {info['area_frac']:.3f}  margin {info['margin_achieved']*100:.2f}%")
    assert frac >= 0.10, frac
    # the gripper is closed and the arm is above the table
    assert s.get_joints()[5] < 1.0
    assert s.gripper_pos()[2] > 0.03


# ---------------------------------------------------------------- (f) cameras render non-blank images
def test_f_renders():
    s = scene()
    s.reset(seed=0, randomise=False)
    for cam in C.CAMERA_NAMES:
        img = s.render(cam)
        assert img.shape == (s.cfg.render_h, s.cfg.render_w, 3) and img.dtype == np.uint8
        assert img.std() > 20, (cam, img.std())
        assert len(np.unique(img.reshape(-1, 3), axis=0)) > 100, cam
    small = s.render(C.WRIST_CAM, 160, 120)
    assert small.shape == (120, 160, 3)


# ---------------------------------------------------------------- (g) randomised reset keeps the position, jitter bounds
def test_g_randomised_reset():
    """The first build's ranges (Randomisation.preset('narrow')); the 'wide' default is covered by (l)-(o)."""
    s = scene()
    saved_R = s.cfg.randomisation
    s.cfg.randomisation = C.Randomisation.preset("narrow")
    try:
        _randomised_reset_checks(s)
    finally:
        s.cfg.randomisation = saved_R
        s.reset(seed=0, randomise=False)


def _randomised_reset_checks(s: ChessScene) -> None:
    s.reset(seed=0, randomise=False)
    ref = s.board_state_symbols()
    R = s.cfg.randomisation
    c = s.cam_ids[C.WRIST_CAM]
    nom_pos, nom_quat = s._nominal["cam_pos"][c], s._nominal["cam_quat"][c]
    n_seeds = 200
    max_disp = max_ang = max_board = 0.0
    n_outside = 0
    Rn = np.zeros(9)
    mujoco.mju_quat2Mat(Rn, nom_quat)
    Rn = Rn.reshape(3, 3)
    s.reset(seed=0, randomise=False)
    nominal_black = float((s.render(C.WRIST_CAM).max(axis=2) < 25).mean())
    max_black, max_back = 0.0, -1.0
    for seed in range(n_seeds):
        s.reset(seed=seed, randomise=True)
        # the lens never moves back into the camera module (the housing starts 1 mm behind it: a 4-5 mm backward
        # jitter used to render the housing, 20-60 % near-black wrist frames on seeds 25/41/99)
        back = float((Rn.T @ (s.model.cam_pos[c] - nom_pos))[2])
        max_back = max(max_back, back)
        assert back <= R.wrist_cam_back_m + 1e-9, (seed, back)
        if seed < 100:
            black = float((s.render(C.WRIST_CAM).max(axis=2) < 25).mean())
            max_black = max(max_black, black)
            assert black <= nominal_black + 0.05, (seed, black, nominal_black)
        if seed < 5:        # full checks on a few seeds (physics settle included)
            st = s.board_state_symbols()
            assert st == ref or (len(st) == 32 and s.to_chess_board().board_fen() == chess.Board().board_fen()), (seed, st)
            for sq, n in s.board_state().items():
                d = np.linalg.norm(s.piece_pos(n)[:2] - s.square_xyz(sq)[:2])
                assert d <= 0.0045, (seed, sq, n, d)
        assert np.linalg.norm(s.board_offset[:2]) > 0 and np.all(np.abs(s.board_offset[:2]) <= R.board_jitter_m + 1e-9)
        assert abs(s.board_yaw) < 1e-12                      # 'narrow': no board yaw
        max_board = max(max_board, float(np.linalg.norm(s.board_offset[:2])))
        # wrist camera jitter: displacement in a ball of 5 mm, rotation <= 2 deg
        disp = float(np.linalg.norm(s.model.cam_pos[c] - nom_pos))
        qd = np.zeros(4)
        mujoco.mju_mulQuat(qd, np.array([nom_quat[0], -nom_quat[1], -nom_quat[2], -nom_quat[3]]), s.model.cam_quat[c])
        ang = float(np.degrees(2 * np.arccos(np.clip(abs(qd[0]), -1, 1))))
        max_disp, max_ang = max(max_disp, disp), max(max_ang, ang)
        assert disp <= R.wrist_cam_m + 1e-9, (seed, disp)
        assert ang <= R.wrist_cam_deg + 1e-6, (seed, ang)
        # the wrist camera still sees all four board corners (keep_board_visible)
        if not corners_inside(s):
            n_outside += 1
    print(f"    {n_seeds} randomised resets: max cam displacement {max_disp*1000:.2f} mm (backward <= {max_back*1000:.2f} mm), "
          f"max cam rotation {max_ang:.2f} deg, max board offset {max_board*1000:.2f} mm, corners outside the image: {n_outside}; "
          f"near-black wrist pixels <= {max_black*100:.1f} % over 100 seeds (nominal {nominal_black*100:.1f} %)")
    assert n_outside == 0, n_outside
    s.reset(seed=0, randomise=False)
    assert np.allclose(s.board_offset, 0) and abs(s.board_yaw) < 1e-12
    assert np.allclose(s.model.geom_rgba, s._nominal["geom_rgba"]) and np.allclose(s.model.cam_quat, s._nominal["cam_quat"])
    # without the guard the same jitter budget does crop corners sometimes (documents why the guard exists)
    R.keep_board_visible = False
    try:
        n_out_unguarded = 0
        for seed in range(n_seeds):
            s.reset(seed=seed, randomise=True, settle_seconds=0)
            n_out_unguarded += 0 if corners_inside(s) else 1
    finally:
        R.keep_board_visible = True
    print(f"    without keep_board_visible: {n_out_unguarded}/{n_seeds} resets crop a corner")
    s.reset(seed=0, randomise=False)


# ---------------------------------------------------------------- (h) config <-> XML consistency
def test_h_config_consistency():
    cfg = C.ChessConfig(board_gap=0.095)          # a non-default gap (the default was tuned to 0.100 by the Motion phase)
    assert cfg.scene_xml_path() != C.SCENE_XML and cfg.scene_hash() != C.DEFAULT_CONFIG.scene_hash()
    tuned_xml = os.path.join(TMP, "chess_scene_gap095.xml")
    write_scene(cfg, tuned_xml)
    m = mujoco.MjModel.from_xml_path(tuned_xml)
    assert embedded_config(m) == cfg.scene_key()
    # the tuned scene is self-consistent: config formula == corner sites == drawn board
    s2 = ChessScene(cfg, xml_path=tuned_xml)
    assert np.allclose(s2.model.body_pos[s2.board_id], cfg.board_centre)
    assert abs(s2.square_xyz("a8")[0] - 0.109) < 1e-9
    assert np.allclose(s2.data.site_xpos[s2.corner_sites["corner_a8"]][:2], (0.095, -0.112), atol=1e-6)
    s2.set_position(START_FEN)
    assert len(s2.board_state()) == 32
    assert abs(s2.piece_pos("b_rook_1")[0] - 0.109) < 1e-6
    # mismatches are refused in both directions ...
    for bad_cfg, xml in ((C.DEFAULT_CONFIG, tuned_xml), (cfg, C.SCENE_XML), (C.ChessConfig(wrist_fovy=60.0), C.SCENE_XML)):
        try:
            ChessScene(bad_cfg, xml_path=xml)
            raise AssertionError(f"mismatch not detected: {bad_cfg.scene_key()} vs {xml}")
        except SceneConfigMismatch as e:
            assert "rebuild" in str(e)
    # ... unless rebuild='auto' regenerates the file for the given config
    auto_xml = os.path.join(TMP, "chess_scene_auto.xml")
    write_scene(C.DEFAULT_CONFIG, auto_xml)
    s3 = ChessScene(cfg, xml_path=auto_xml, rebuild="auto")
    assert embedded_config(s3.model) == cfg.scene_key() and np.allclose(s3.model.body_pos[s3.board_id], cfg.board_centre)
    # an XML without the record (hand-written / old) is refused too (derived from the TMP-built file so meshdir resolves)
    old_xml = os.path.join(TMP, "chess_scene_norecord.xml")
    with open(tuned_xml) as f:
        txt = f.read()
    i, j = txt.index("<custom>"), txt.index("</custom>") + len("</custom>")
    with open(old_xml, "w") as f:
        f.write(txt[:i] + txt[j:])
    assert embedded_config(mujoco.MjModel.from_xml_path(old_xml)) is None
    try:
        ChessScene(cfg, xml_path=old_xml)
        raise AssertionError("XML without a config record was accepted")
    except SceneConfigMismatch:
        pass
    # an XML whose jaw-geometry record disagrees with config (arm model changed) is refused with a specific message
    jaw_xml = os.path.join(TMP, "chess_scene_badjaw.xml")
    i = txt.index('name="jaw_model"')
    j = txt.index('data="', i) + len('data="')
    with open(jaw_xml, "w") as f:
        f.write(txt[:j] + "0.09 " + txt[j:].split(" ", 1)[1])      # r_fixed 86.1 -> 90 mm
    try:
        ChessScene(cfg, xml_path=jaw_xml)
        raise AssertionError("stale jaw geometry was accepted")
    except SceneConfigMismatch as e:
        assert "jaw" in str(e) and "print-jaw" in str(e)
    # the default XML on disk still matches the default config (nothing above touched it)
    assert ChessScene().xml_path == C.SCENE_XML


# ---------------------------------------------------------------- (i) watch-pose search: default budget, re-tuned config, strictness
def test_i_watch_search():
    # (1) a re-tuned config (board_gap 0.10, 320x180 image, same 16:9 aspect as the default) must reach the margin with the DEFAULT search budget
    cfg = C.ChessConfig(board_gap=0.100, render_w=320, render_h=180)
    s2 = ChessScene(cfg, xml_path=os.path.join(TMP, "chess_scene_gap100.xml"))
    cache = os.path.join(TMP, "watch_gap100.json")
    t0 = time.time()
    w = s2.watch_pose(path=cache, progress_path=SEARCH_PROGRESS)   # default budget, warm-started from the prior
    info = s2._watch
    print(f"    re-tuned config search: feasible {info['feasible']} margin {info['margin_achieved']*100:.2f}% area {info['area_frac']:.3f} "
          f"rounds {info['rounds']} evals {info['n_evals']} in {time.time() - t0:.0f}s")
    assert info["feasible"] and info["margin_achieved"] >= 0.05 - 1e-9, info
    assert os.path.exists(cache) and info["config"] == cfg.watch_key()
    s2.set_joints(w)
    assert corners_inside(s2, 0.05)
    # (2) a COLD search (no warm starts) with the default budget on the default scene
    s = scene()
    t0 = time.time()
    r = s.compute_watch_pose(seed=7, warm_starts=[], verbose=True, progress_path=SEARCH_PROGRESS)
    print(f"    cold search: feasible {r['feasible']} margin {r['margin_achieved']*100:.2f}% area {r['area_frac']:.3f} "
          f"rounds {r['rounds']} evals {r['n_evals']} in {time.time() - t0:.0f}s")
    assert r["feasible"], r
    # (3) an impossible camera (fovy 40) -> strict watch_pose raises, strict=False warns, and nothing silently passes
    cfg3 = C.ChessConfig(wrist_fovy=40.0)
    s3 = ChessScene(cfg3, xml_path=os.path.join(TMP, "chess_scene_fovy40.xml"))
    cache3 = os.path.join(TMP, "watch_fovy40.json")
    small = dict(n_samples=3000, n_keep=4, n_refine=150, max_rounds=1, progress_path=SEARCH_PROGRESS)
    try:
        s3.watch_pose(path=cache3, **small)
        raise AssertionError("infeasible watch pose did not raise")
    except RuntimeError as e:
        assert "NOT feasible" in str(e)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w3 = s3.watch_pose(path=cache3, strict=False, **small)
    assert w3.shape == (6,) and any("NOT feasible" in str(c.message) for c in caught)
    assert not s3._watch["feasible"]
    try:
        s3.reset(seed=0)                   # reset uses the strict watch pose
        raise AssertionError("reset accepted an infeasible watch pose")
    except RuntimeError:
        pass
    s3.reset(seed=0, arm_pose=None)        # explicit pose still works


# ---------------------------------------------------------------- (j) gripper: jaw model == meshes, commands, a physical squeeze
VERTICAL_POSE = np.array([0.0, -31.87, 45.32, 75.81, 0.0, 0.0])   # jaws vertical, fingertips ~41 mm above the table at x 0.16


def test_j_gripper_model():
    s = scene()
    m = s.model
    # the jaw record embedded in the XML == a fresh measurement of the meshes == config constants (0.2 mm)
    rec = embedded_jaw_geometry(m)
    assert rec is not None and jaw_geometry_agrees(rec, measure_jaw_geometry())
    cfg_rec = dict(C.JAW_MODEL, table_q=list(C.JAW_CLEARANCE_Q), table_clearance=list(C.JAW_CLEARANCE))
    assert jaw_geometry_agrees(rec, cfg_rec)
    # model vs mesh fingertip gap / clearance over the whole joint range and at every command angle
    s.reset(seed=0, randomise=False, fen=chess.Board(None).fen(), arm_pose=None, settle_seconds=0)
    cmds = {"0%": C.GRIPPER_Q_PCT0, "100%": C.GRIPPER_Q_PCT100, "approach": C.gripper_open_ctrl(), "q_closed": C.GRIPPER_Q_CLOSED,
            "q_open": C.GRIPPER_Q_OPEN}
    cmds.update({f"close_{t}": C.gripper_close_ctrl(t) for t in C.PIECE_TYPES})
    cmds.update({f"stall_{t}": C.gripper_stall_ctrl(t) for t in C.PIECE_TYPES})
    worst_gap = worst_clr = 0.0
    for name, q in list(cmds.items()) + [(f"q{v:.2f}", v) for v in np.linspace(C.GRIPPER_Q_CLOSED, C.GRIPPER_Q_OPEN, 13)]:
        s.data.qpos[s.qpos_adr[5]] = q
        mujoco.mj_forward(m, s.data)
        worst_gap = max(worst_gap, abs(s.jaw_gap() - C.gripper_gap(q)))
        worst_clr = max(worst_clr, abs(s.jaw_clearance() - C.gripper_clearance(q)))
    print(f"    jaw model vs meshes: max |gap| error {worst_gap*1000:.3f} mm, max |clearance| error {worst_clr*1000:.3f} mm")
    assert worst_gap < 1e-4 and worst_clr < 2e-4, (worst_gap, worst_clr)
    # the simulated fingertip gaps of the commands are what the spec asks for
    assert abs(C.gripper_gap(C.GRIPPER_Q_PCT0) - 0.007) < 1e-6 and abs(C.gripper_gap(C.GRIPPER_Q_PCT100) - 0.080) < 1e-6
    assert abs(C.gripper_gap(C.gripper_open_ctrl()) - 0.045) < 1e-6
    assert abs(C.gripper_gap(C.GRIPPER_Q_OPEN) - 0.1411) < 2e-4, "MJCF full range = 141 mm, not 80 mm"
    for t in C.PIECE_TYPES:
        stem = C.stem_diameter(t)
        assert abs(C.gripper_gap(C.gripper_close_ctrl(t)) - max(stem - 0.002, C.gripper_gap(C.GRIPPER_Q_CLOSED))) < 1e-6, t
        assert C.gripper_clearance(C.gripper_close_ctrl(t)) < stem - 0.002, t       # the jaws will stall on the stem
        assert abs(C.gripper_clearance(C.gripper_stall_ctrl(t)) - stem) < 1e-6, t
        assert C.gripper_close_ctrl(t) < C.gripper_stall_ctrl(t) < C.gripper_open_ctrl()
    # ---- physics: close on a king standing next to the fixed fingertip, then lift it
    s.reset(seed=0, randomise=False, fen=chess.Board(None).fen(), arm_pose=None, settle_seconds=0)
    pose = VERTICAL_POSE.copy()
    pose[5] = C.gripper_frac_to_percent(C.gripper_open_ctrl())
    s.set_joints(pose)
    xaxis = s.gripper_mat()[:, 0]
    assert np.degrees(np.arccos(np.clip(-xaxis[2], -1, 1))) < 2.0, xaxis          # jaws vertical
    tf, tm = s.jaw_tip_positions()
    stem_r = C.stem_diameter("king") / 2
    lat = tm - tf
    lat[2] = 0.0
    lat /= np.linalg.norm(lat)
    centre = tf + lat * (stem_r + 0.0015)                                           # 1.5 mm from the fixed jaw
    # the fingertips lie ~6 mm beyond the gripperframe site along the jaw direction: report the offset (Motion phase
    # must target the grasp height with the TIPS) and check the tips are on the king's stem (30..100 % of its height)
    off = s.gripper_mat().T @ (tf - s.gripper_pos())
    print(f"    fixed fingertip offset from the gripperframe site (site frame): {np.round(off*1000, 1)} mm; tip z {tf[2]*1000:.1f} mm")
    H = C.PIECE_SPECS["king"][0]
    assert s.cfg.board_thickness + 0.3 * H + 0.003 < tf[2] < s.cfg.board_thickness + H - 0.003, tf[2]
    s.set_piece_pose("w_king_1", np.array([centre[0], centre[1], s.cfg.board_thickness + 0.0005]))
    mujoco.mj_forward(m, s.data)
    s.settle(0.3)
    p0 = s.piece_pos("w_king_1")
    assert np.linalg.norm(p0[:2] - centre[:2]) < 0.002, "piece disturbed by the open jaws"
    close = pose.copy()
    close[5] = C.gripper_frac_to_percent(C.gripper_close_ctrl("king"))
    s.set_ctrl(close)
    s.settle(1.0)
    q = s.gripper_q()
    clr = s.jaw_clearance()
    p1 = s.piece_pos("w_king_1")
    stem = C.stem_diameter("king")
    print(f"    squeeze: gripper q {q:+.4f} (stall model {C.gripper_stall_ctrl('king'):+.4f}, cmd {C.gripper_close_ctrl('king'):+.4f}) "
          f"clearance {clr*1000:.2f} mm (stall width {stem*1000:.1f}: the pads sink {(stem - clr)*1000:.2f} mm into the {s.cfg.piece_collision}), "
          f"piece shift {np.linalg.norm(p1[:2]-p0[:2])*1000:.2f} mm tilt {s.piece_tilt_deg('w_king_1'):.2f} deg")
    # the jaws stall on the king's stem (the stall width of the scene's collision model: the 12 mm cylinder, or the
    # 11.4 mm collar ring for 'bands', where the soft contacts of the box fingers sink ~0.75 mm per side into the
    # hull band at the squeeze force; the cylinder: < 0.1 mm): clearance = stall width - 0..2.5 mm, angle within 0.05 rad
    assert abs(q - C.gripper_stall_ctrl("king")) < 0.05, q
    assert -0.0025 <= clr - stem <= 0.0015, clr
    assert np.linalg.norm(p1[:2] - p0[:2]) < 0.005 and s.piece_tilt_deg("w_king_1") < 10.0
    # lift: raise the arm 10 deg at the shoulder (this also tilts the gripper by ~10 deg): the piece must come along
    # and stay aligned with the jaw direction (no slip in the pinch)
    def jaw_alignment_deg():
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, s.piece_quat("w_king_1"))
        up = R.reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(up @ (-s.gripper_mat()[:, 0]), -1, 1))))
    lift = close.copy()
    lift[1] -= 10.0
    site0 = s.gripper_pos()
    s.set_ctrl(lift)
    s.settle(1.0)
    dz_site, dz_piece = s.gripper_pos()[2] - site0[2], s.piece_pos("w_king_1")[2] - p1[2]
    print(f"    lift: fingertips +{dz_site*1000:.1f} mm, piece +{dz_piece*1000:.1f} mm, piece tilt {s.piece_tilt_deg('w_king_1'):.2f} deg "
          f"(gripper tilted), piece-to-jaw misalignment {jaw_alignment_deg():.2f} deg")
    assert dz_site > 0.01 and dz_piece > 0.6 * dz_site and jaw_alignment_deg() < 5.0
    # lower it back and release the way the expert does (open, then retreat straight up): the king must stand
    # upright where it was picked.  Right after opening it may still touch the fixed pad ('bands': the soft pads
    # sink ~0.5 mm into the collar hull band and the position-controlled arm comes back ~1 mm higher than it left,
    # so the king drops onto the board leaning on the pad: 2.2 deg measured); it stands up once the jaw retreats
    s.set_ctrl(close)
    s.settle(1.0)
    release = close.copy()
    release[5] = C.gripper_frac_to_percent(C.gripper_open_ctrl())
    s.set_ctrl(release)
    s.settle(1.0)
    p_open = s.piece_pos("w_king_1")
    tilt_open = s.piece_tilt_deg("w_king_1")
    k = Kinematics(m, s.cfg.piece_collision)
    q5 = C.lerobot_to_qpos(s.get_joints())[:5]
    tip = s.jaw_tip_positions()[0]
    r = k.solve_R(tip + np.array([0, 0, 0.025]), s.gripper_mat(), q_init=q5)     # same jaw orientation, 25 mm higher
    assert r.converged, r
    for i in range(1, 31):                       # 25 mm straight up in 1 s (joint-space interpolation)
        cmd = release.copy()
        cmd[:5] = np.degrees(q5 + (r.q5 - q5) * i / 30)
        s.set_ctrl(cmd)
        s.settle(1.0 / 30)
    s.settle(1.0)
    p2 = s.piece_pos("w_king_1")
    print(f"    release: tilt {tilt_open:.2f} deg right after opening (base z {p_open[2]*1000:.1f} mm), after the 25 mm vertical "
          f"retreat tilt {s.piece_tilt_deg('w_king_1'):.2f} deg, xy shift {np.linalg.norm(p2[:2]-p0[:2])*1000:.2f} mm, "
          f"base z {p2[2]*1000:.1f} mm (board top {s.cfg.board_thickness*1000:.0f})")
    assert tilt_open < 5.0 and np.linalg.norm(p_open[:2] - p0[:2]) < 0.003, (tilt_open, p_open)
    assert s.piece_tilt_deg("w_king_1") < 2.0 and np.linalg.norm(p2[:2] - p0[:2]) < 0.003 and abs(p2[2] - s.cfg.board_thickness) < 0.001
    s.reset(seed=0, randomise=False)


# ---------------------------------------------------------------- (k) control rate: exact 30 Hz grid
def test_k_control_rate():
    s = scene()
    s.reset(seed=0, randomise=False, arm_pose=None, settle_seconds=0)
    assert s.control_ticks == 0 and s.data.time == 0.0
    sizes = set()
    for k in range(1, 301):
        t_before = s.data.time
        s.control_step()
        sizes.add(int(round((s.data.time - t_before) / s.model.opt.timestep)))
        assert abs(s.data.time - k / C.CONTROL_HZ) <= s.model.opt.timestep + 1e-9, (k, s.data.time)
    assert abs(s.data.time - 10.0) < 1e-6 and s.control_ticks == 300
    assert sizes == {16, 17}, sizes
    print(f"    300 control steps: sim time {s.data.time:.6f} s (exact 10.0), substeps used {sorted(sizes)}")
    # the default reset settles 0.2 s of physics: the grid is anchored there, so the first control steps are still
    # full periods (they used to be six 1-step ticks: 6 actions applied in 12 ms); likewise after settle() /
    # set_position(settle_steps=..) / a raw step()
    s.reset(seed=0, randomise=False)
    t0 = s.data.time
    assert s.control_ticks == 0 and abs(s.grid_t0 - t0) < 1e-12 and abs(t0 - 0.2) < 1e-9, (s.control_ticks, s.grid_t0, t0)
    first = []
    for k in range(1, 31):
        t_before = s.data.time
        s.control_step()
        first.append(int(round((s.data.time - t_before) / s.model.opt.timestep)))
        assert abs(s.data.time - t0 - k / C.CONTROL_HZ) <= s.model.opt.timestep + 1e-9, (k, s.data.time)
    assert set(first) == {16, 17}, first
    s.step(7)                                   # physics advanced outside the grid: the next control step re-anchors
    t_before = s.data.time
    s.control_step()
    n = int(round((s.data.time - t_before) / s.model.opt.timestep))
    assert n in (16, 17) and s.control_ticks == 1 and abs(s.grid_t0 - t_before) < 1e-12, (n, s.control_ticks)
    s.set_position(chess.Board(), settle_steps=50)
    assert s.control_ticks == 0
    print(f"    after the default reset (settled {t0:.3f} s): first substeps {first[:8]} (all 16/17); a raw step(7) re-anchors "
          f"the grid ({n} substeps next)")
    s.reset(seed=0, randomise=False)


# ---------------------------------------------------------------- (l) board pose (offset + yaw): square <-> world round trips
def test_l_board_pose_roundtrip():
    s = scene()
    x0, x1, y0, y1, yaw = C.Randomisation().board_range()
    poses = [(0.0, 0.0, 0.0), (x1, y1, yaw), (x0, y0, -yaw), (x1, y0, -yaw), (x0, y1, yaw), (0.020, -0.020, 5.0), (-0.020, 0.020, -5.0),
             (0.0, 0.0, 10.0)]
    fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1"
    rng = np.random.default_rng(3)
    worst = 0.0
    for dx, dy, a in poses:
        s.set_board_pose(dx, dy, a)
        assert np.allclose(s.board_pose(), (dx, dy, a), atol=1e-9), (s.board_pose(), (dx, dy, a))
        # square_xyz -> nearest_square for every square, also with an in-square offset along the BOARD axes
        R2 = np.array([[np.cos(np.radians(a)), -np.sin(np.radians(a))], [np.sin(np.radians(a)), np.cos(np.radians(a))]])
        for sq in [f + r for f in "abcdefgh" for r in "12345678"]:
            p = s.square_xyz(sq)
            n, d = s.nearest_square(p)
            assert n == sq and d < 1e-9, (sq, n, d)
            off = R2 @ rng.uniform(-0.0125, 0.0125, 2)
            n2, d2 = s.nearest_square(p[:2] + off)
            assert n2 == sq and abs(d2 - np.linalg.norm(off)) < 1e-9, (sq, n2)
            assert np.allclose(s.world_to_board(s.board_to_world(p[:2])), p[:2])
        # the square pitch / layout is rigid: a8-h8 along the rotated file axis, a8-a1 along the rank axis
        v = s.square_xyz("h8") - s.square_xyz("a8")
        assert abs(np.linalg.norm(v[:2]) - 7 * C.SQUARE) < 1e-9 and abs(np.degrees(np.arctan2(v[1], v[0])) - (90.0 + a)) < 1e-6
        # the analytic corners == the corner sites of the (moved) board body; the board centre moved by (dx, dy)
        site = np.array([s.data.site_xpos[i] for i in s.corner_sites.values()])
        worst = max(worst, float(np.abs(site - s.board_corners()).max()))
        assert np.allclose(site, s.board_corners(), atol=1e-9)
        assert np.allclose(s.board_corners()[:, :2].mean(axis=0), s.cfg.board_centre[:2] + (dx, dy))
        assert np.allclose(s.board_corners(), s.board_corners_at(dx, dy, a, (0.0, 1.0)))
        # inside_board follows the rotated edges
        a1c = s.board_corners()[2, :2]
        outward = s.board_to_world(s.cfg.board_centre[:2] + np.array([C.BOARD_SIZE / 2 + 0.008, 0.0]))
        assert not s.inside_board(outward) and s.inside_board(s.square_xyz("a1")[:2]) and s.inside_board(a1c)
        # a FEN placed on the moved board reads back (physics settle included), pieces rotated with the board
        mapping = s.set_position(fen, settle_steps=100)
        assert s.to_chess_board().board_fen() == chess.Board(fen).board_fen(), (dx, dy, a)
        for sq, n in mapping.items():
            q = s.piece_quat(n)
            yaw_p = np.degrees(2 * np.arctan2(q[3], q[0]))
            assert abs((yaw_p - a + 180) % 360 - 180) < 1.0, (sq, n, yaw_p, a)
            assert np.linalg.norm(s.piece_pos(n)[:2] - s.square_xyz(sq)[:2]) < 0.002
        assert len(s.board_state()) == len(mapping)
    # the drop-zone / spare pads stay on the table: no piece standing in a cell (20 mm foot, pad jitter extremes)
    # ever overlaps the board box (playing surface + border) at any board pose of the 'wide' range
    from chess_sim.feasibility import extreme_poses
    half = C.BOARD_SIZE / 2 + C.BOARD_BORDER
    jz = C.Randomisation().zone_jitter_m
    cells = [np.array(s.cfg.drop_cell_xy(i, j)) for i in range(4) for j in range(4)] + [np.array(s.cfg.spare_cell_xy(i)) for i in range(2)]
    min_clear = 1.0
    for dx, dy, a_ in extreme_poses(C.Randomisation().board_range()):
        s.set_board_pose(dx, dy, a_, forward=False)
        for c in cells:
            for ox in (-jz, jz):
                for oy in (-jz, jz):
                    loc = s.world_to_board(c + (ox, oy)) - s.cfg.board_centre[:2]
                    q = np.maximum(np.abs(loc) - half, 0.0)
                    min_clear = min(min_clear, float(np.linalg.norm(q)) - 0.010)
    assert min_clear > 0.0, min_clear
    # restore_nominal (any reset) puts the board back
    s.reset(seed=0, randomise=False)
    assert np.allclose(s.board_pose(), (0, 0, 0))
    # a randomised reset draws offset + yaw inside the 'wide' range; the drop zone / spare pads jitter on their own
    R = s.cfg.randomisation
    assert R.name == "wide"
    yaws, offs, zones = [], [], []
    for seed in range(40):
        s.reset(seed=seed, randomise=True, settle_seconds=0)
        dx, dy, a = s.board_pose()
        assert x0 - 1e-9 <= dx <= x1 + 1e-9 and y0 - 1e-9 <= dy <= y1 + 1e-9 and abs(a) <= yaw + 1e-9, (seed, dx, dy, a)
        assert np.all(np.abs(s.drop_zone_offset[:2]) <= R.zone_jitter_m + 1e-9) and np.all(np.abs(s.spare_zone_offset[:2]) <= R.zone_jitter_m + 1e-9)
        assert np.allclose(s.drop_cell_xyz(0, 0)[:2], np.array(s.cfg.drop_cell_xy(0, 0)) + s.drop_zone_offset[:2])
        assert np.allclose(s.spare_xyz("w")[:2], np.array(s.cfg.spare_cell_xy(0)) + s.spare_zone_offset[:2])
        assert np.allclose(s.piece_pos("w_queen_spare_1")[:2], s.spare_xyz("w")[:2], atol=1e-9)
        yaws.append(a)
        offs.append((dx, dy))
        zones.append(np.linalg.norm(s.drop_zone_offset[:2]))
        assert s.to_chess_board().board_fen() == chess.Board().board_fen(), seed
    print(f"    {len(poses)} board poses (up to +-20 mm, +-10 deg): 64 squares round-trip (nearest_square / world_to_board), FEN read back "
          f"after a 0.2 s settle, pieces rotated with the board, corner sites == analytic corners ({worst*1e9:.1f} nm); 40 randomised "
          f"resets: yaw {min(yaws):+.1f}..{max(yaws):+.1f} deg, dx {min(o[0] for o in offs)*1000:+.1f}..{max(o[0] for o in offs)*1000:+.1f} mm, "
          f"dy {min(o[1] for o in offs)*1000:+.1f}..{max(o[1] for o in offs)*1000:+.1f} mm, drop-zone jitter <= {max(zones)*1000:.1f} mm; "
          f"drop / spare pieces clear the board box by >= {min_clear*1000:.1f} mm at every extreme pose")
    s.reset(seed=0, randomise=False)


# ---------------------------------------------------------------- (m) piece jitter: in the square, never touching a neighbour
def test_m_piece_jitter_no_overlap():
    s = scene()
    R = s.cfg.randomisation
    rng = random.Random(5)
    fens = [START_FEN]
    b = chess.Board()
    for k in range(30):                           # crowded mid-game positions from random play
        b.push(rng.choice(list(b.legal_moves)))
        if b.is_game_over():
            break
        if k % 6 == 5:
            fens.append(b.fen())
    n_pairs = n_resets = 0
    worst_gap, worst_jit = 1.0, 0.0
    max_jit_axis = 0.0
    contacts = 0
    piece_bodies = set(s.piece_body_id.values())
    for i in range(120):
        fen = fens[i % len(fens)]
        mapping = s.reset(seed=1000 + i, randomise=True, fen=fen, settle_seconds=0)
        n_resets += 1
        names = list(mapping.items())
        for k1, (sq1, n1) in enumerate(names):
            p1 = s.piece_pos(n1)
            local = s.world_to_board(p1[:2]) - s.world_to_board(s.square_xyz(sq1)[:2])     # jitter along the board axes
            max_jit_axis = max(max_jit_axis, float(np.abs(local).max()))
            assert np.all(np.abs(local) <= R.piece_jitter_m + 1e-9), (sq1, local)
            for sq2, n2 in names[k1 + 1:]:
                d = float(np.linalg.norm(p1[:2] - s.piece_pos(n2)[:2]))
                need = s.piece_min_distance(s.piece_type[n1], s.piece_type[n2])
                if d < 0.045:
                    n_pairs += 1
                    worst_gap = min(worst_gap, d - need)
                assert d >= need + R.piece_min_gap_m - 1e-9, (fen, sq1, sq2, d, need)
        # no piece-piece contact in the physics right after the reset
        mujoco.mj_forward(s.model, s.data)
        for c in range(s.data.ncon):
            con = s.data.contact[c]
            b1, b2 = s.model.geom_bodyid[con.geom1], s.model.geom_bodyid[con.geom2]
            if b1 in piece_bodies and b2 in piece_bodies and con.dist < 0:
                contacts += 1
    # the jitter does reach the full +-5 mm and the envelope check is exact: two kings 21.2 mm apart would touch
    assert max_jit_axis > 0.9 * R.piece_jitter_m, max_jit_axis
    assert abs(s.piece_min_distance("king", "king") - 2 * max(r for _, _, r in C.PIECE_ENVELOPE["king"])) < 0.001
    assert contacts == 0, contacts
    print(f"    {n_resets} 'wide' resets ({len(fens)} positions incl. random mid-games): jitter up to {max_jit_axis*1000:.2f} mm per board axis, "
          f"{n_pairs} neighbour pairs, smallest footprint gap {worst_gap*1000:.2f} mm (>= {R.piece_min_gap_m*1000:.1f} required; mesh "
          f"envelopes at every height), 0 piece-piece contacts")
    s.reset(seed=0, randomise=False)


# ---------------------------------------------------------------- (n) multi-view scan: all 64 squares + the drop zone for every board pose
def test_n_scan_coverage():
    s = scene()
    s.reset(seed=0, randomise=False)
    views = s.scan_poses()                     # strict: every view feasible (cached in assets/watch_pose_scan.json)
    assert len(views) == len(C.SCAN_REGIONS) and np.allclose(views[0], s.watch_pose())
    W, H = s.cfg.render_w, s.cfg.render_h
    u0, u1, v0, v1 = C.WRIST_IMAGE_WINDOW
    rows = []
    for i, reg in enumerate(C.SCAN_REGIONS):
        r = s.scan_metrics(i)
        assert r["feasible"] and r["occluded"] is False and r["collision"] is False, (reg, r["extent"], r["occluded"])
        rows.append((reg, (1 - r["extent"]) / 2, r["area_frac"]))
        assert (1 - r["extent"]) / 2 >= C.SCAN_MARGIN - 1e-9
    # the board tiles of the views cover every square (tiles lie on square boundaries)
    tiles = [ChessScene._region4(r) for r in C.SCAN_REGIONS if not ChessScene._is_drop(r)]
    for f in range(8):
        for rk in range(1, 9):
            d0, d1 = (8 - rk) / 8, (9 - rk) / 8          # depth band of the square (0 = rank 8 edge)
            f0, f1 = f / 8, (f + 1) / 8
            assert any(t[0] - 1e-9 <= d0 and d1 <= t[1] + 1e-9 and t[2] - 1e-9 <= f0 and f1 <= t[3] + 1e-9 for t in tiles), (f, rk)
    assert any(ChessScene._is_drop(r) for r in C.SCAN_REGIONS)
    # random board poses / pad jitters inside the range, the live model moved there and the arm at each view:
    # every square's 4 corners inside the window of at least one view (independent of the tile bookkeeping)
    x0, x1, y0, y1, yaw = s.cfg.scan_board_range
    rng = np.random.default_rng(11)
    worst = 1.0
    sq_names = [fl + str(rk) for fl in "abcdefgh" for rk in range(1, 9)]
    d = s.data
    for k in range(60):
        pose = (rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(-yaw, yaw))
        s.set_board_pose(*pose, forward=False)
        imgs = []
        for v in views:
            d.qpos[s.qpos_adr] = C.lerobot_to_qpos(v)
            mujoco.mj_kinematics(s.model, d)
            mujoco.mj_camlight(s.model, d)
            imgs.append(v)
            for sq in list(sq_names):
                c = s.square_xyz(sq)
                R2 = s._board_rot()
                pts = np.array([np.r_[c[:2] + R2 @ np.array(o) * C.SQUARE / 2, c[2]] for o in ((-1, -1), (-1, 1), (1, -1), (1, 1))])
                px = s.project(pts, C.WRIST_CAM)
                if not np.any(np.isnan(px)):
                    m_ = min(px[:, 0].min() / W - u0, u1 - px[:, 0].max() / W, px[:, 1].min() / H - v0, v1 - px[:, 1].max() / H)
                    if m_ >= 0:
                        sq_names.remove(sq)
        assert not sq_names, (pose, sq_names)
        sq_names = [fl + str(rk) for fl in "abcdefgh" for rk in range(1, 9)]
    s.reset(seed=0, randomise=False)
    descr = "; ".join(f"{('drop zone' if isinstance(rg, str) else rg)}: margin {m*100:.1f}% area {a*100:.1f}%" for rg, m, a in rows)
    print(f"    {len(views)} scan views (fovy {s.cfg.wrist_fovy:.0f} deg, {W}x{H}, roll {s.cfg.wrist_roll_deg:+.0f} deg, window u <= "
          f"{u1:.3f}): {descr}; worst over the 9 board poses / 5 pad jitters of the range, rays at every pose; 60 random board "
          f"poses: every square inside the window of >= 1 view")
    for i, v in enumerate(views):
        print(f"      view {i}: {np.round(v, 1).tolist()}")


# ---------------------------------------------------------------- (o) render sanity under the 'wide' randomisation
def test_o_wide_render_sanity():
    s = scene()
    R = s.cfg.randomisation
    assert R.name == "wide"
    s.reset(seed=0, randomise=False)
    nominal = s.render(C.WRIST_CAM)
    nominal_black = float((nominal.max(axis=2) < 25).mean())
    nominal_lum = float(nominal.mean())
    c = s.cam_ids[C.WRIST_CAM]
    nom_pos, nom_quat = s._nominal["cam_pos"][c], s._nominal["cam_quat"][c]
    Rn = np.zeros(9)
    mujoco.mju_quat2Mat(Rn, nom_quat)
    Rn = Rn.reshape(3, 3)
    blacks, lums, fracs, angs, redraws = [], [], [], [], 0
    n_out = 0
    table_changed = 0
    for seed in range(60):
        s.reset(seed=seed, randomise=True, settle_seconds=0)
        img = s.render(C.WRIST_CAM)
        black = float((img.max(axis=2) < 25).mean())
        blacks.append(black)
        lums.append(float(img.mean()))
        assert black <= nominal_black + 0.06, (seed, black, nominal_black)
        assert img.std() > 15 and 0.35 * nominal_lum <= img.mean() <= 1.9 * nominal_lum, (seed, img.mean(), nominal_lum)
        if not corners_inside(s):
            n_out += 1
        # the board is visible: it covers a large share of the board quad's pixels (colour-independent test: the
        # quad region differs from the table around it)
        px = s.project(s.target_corners_at(*s.board_pose()), C.WRIST_CAM)       # the watch region of the live board
        quad = px[[0, 1, 3, 2]]
        area = 0.5 * abs(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))) / (s.cfg.render_w * s.cfg.render_h)
        fracs.append(area)
        back = float((Rn.T @ (s.model.cam_pos[c] - nom_pos))[2])
        assert back <= R.wrist_cam_back_m + 1e-9 and np.linalg.norm(s.model.cam_pos[c] - nom_pos) <= R.wrist_cam_m + 1e-9
        qd = np.zeros(4)
        mujoco.mju_mulQuat(qd, np.array([nom_quat[0], -nom_quat[1], -nom_quat[2], -nom_quat[3]]), s.model.cam_quat[c])
        ang = float(np.degrees(2 * np.arccos(np.clip(abs(qd[0]), -1, 1))))
        assert ang <= R.wrist_cam_deg + 1e-6
        angs.append(ang)
        redraws += s.last_cam_redraws
        table_changed += not np.allclose(s.model.geom_rgba[s.table_geom_id], s._nominal["geom_rgba"][s.table_geom_id])
        assert 0.6 * s._nominal["head_ambient"][0] - 1e-9 <= s.model.vis.headlight.ambient[0] <= 1.4 * s._nominal["head_ambient"][0] + 1e-9
    assert n_out == 0, n_out
    assert table_changed == 60
    s.reset(seed=0, randomise=False)
    assert np.allclose(s.model.vis.headlight.ambient, s._nominal["head_ambient"]) and np.allclose(s.model.light_dir, s._nominal["light_dir"])
    assert np.allclose(s.render(C.WRIST_CAM), nominal)
    print(f"    60 'wide' resets: near-black wrist pixels <= {max(blacks)*100:.1f} % (nominal {nominal_black*100:.1f} %), mean level "
          f"{min(lums):.0f}..{max(lums):.0f} (nominal {nominal_lum:.0f}), watch-region quad {min(fracs)*100:.1f}..{max(fracs)*100:.1f} % "
          f"of the image, its corners always inside the window, camera rotation <= {max(angs):.2f} deg, {redraws} camera re-draws (keep_board_visible) in 60 resets")


TESTS = [test_a_dimensions, test_a2_piece_models, test_a3_piece_model_switches, test_b_texture_orientation, test_c_stability, test_d_board_state_roundtrip,
         test_e_watch_pose, test_f_renders, test_g_randomised_reset, test_h_config_consistency, test_i_watch_search,
         test_j_gripper_model, test_k_control_rate, test_l_board_pose_roundtrip, test_m_piece_jitter_no_overlap,
         test_n_scan_coverage, test_o_wide_render_sanity]


def main() -> int:
    """Runs every test; the overall progress bar goes to chess_sim/data/progress.txt (progress_view.command),
    the inner watch-pose searches write theirs to a temp file so the shared bar never shows a deliberately
    infeasible search as a failure.  The temp directory is removed at the end."""
    failed = 0
    pb = ProgressBar(len(TESTS))
    pb.update(0, "test_scene: starting", force=True)
    try:
        for i, t in enumerate(TESTS):
            pb.update(i, f"test_scene: {t.__name__}", force=True)
            t0 = time.time()
            try:
                t()
                print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
            except Exception:
                failed += 1
                print(f"FAIL {t.__name__} ({time.time() - t0:.1f}s)")
                traceback.print_exc()
        print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
        pb.done(f"test_scene: {len(TESTS) - failed}/{len(TESTS)} PASS" if not failed else f"test_scene: {failed} FAILED")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
