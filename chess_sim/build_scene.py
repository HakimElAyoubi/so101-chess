"""Generate the chess scene XML (+ the board texture PNG) from config.

Run:  python -m chess_sim.build_scene  [--board-gap 0.09] [--board-thickness 0.01] [--out PATH]
  or  python chess_sim/build_scene.py ...   (same; the package root is added to sys.path)
The default output is cfg.scene_xml_path(): scene/chess_scene.xml for the default config, otherwise
scene/chess_scene_<hash>.xml, so a tuned config never overwrites the default scene.  The config the
file was built from is embedded as <custom><text name="chess_config"> and checked by ChessScene.
The SO-101 (camera-mount variant) is inlined so meshdir/texturedir resolve relative to the
scene file regardless of the CWD.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
from xml.dom import minidom

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):          # run as a plain script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chess_sim import config as C
else:
    from . import config as C


# --------------------------------------------------------------------------- small helpers
def _f(x) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.atleast_1d(x))


def quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def lookat_xyaxes(pos, target, up=(0, 0, 1)) -> str:
    """MuJoCo camera xyaxes for a camera at pos looking at target (camera looks along -z)."""
    pos, target, up = map(lambda v: np.asarray(v, float), (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(-fwd, x)          # z_cam = -fwd ; y = z x x
    return _f(np.concatenate([x, y]))


# --------------------------------------------------------------------------- board texture
def make_board_texture(path: str = C.BOARD_TEXTURE_PNG, px: int = 96) -> str:
    """Checker texture with a border carrying file letters / rank numbers.

    Texture mapping (verified by rendering): image columns run along world +x (rank 8 -> rank 1),
    image rows run along world -y as rows go down (top row = +y = file h, bottom row = file a).
    """
    border = int(round(px * C.BOARD_BORDER / C.SQUARE))
    size = 8 * px + 2 * border
    img = Image.new("RGB", (size, size), tuple(int(255 * v) for v in C.BORDER_RGB))
    d = ImageDraw.Draw(img)
    dark = tuple(int(255 * v) for v in C.DARK_RGB)
    light = tuple(int(255 * v) for v in C.LIGHT_RGB)
    for col in range(8):            # col 0 = rank 8 (near the robot, small x)
        rank = 8 - col
        for row in range(8):        # row 0 (top) = file h (+y) ... row 7 = file a (-y)
            file_idx = 7 - row
            colour = dark if C.square_is_dark(file_idx, rank) else light
            x0, y0 = border + col * px, border + row * px
            d.rectangle([x0, y0, x0 + px - 1, y0 + px - 1], fill=colour)
    # letters (readable from the human side, i.e. looking toward -x: image rotated 90 deg CW)
    try:
        font = ImageFont.load_default(size=int(border * 0.8))
    except TypeError:               # very old Pillow
        font = ImageFont.load_default()
    txt_rgb = (235, 225, 205)

    def put(text, cx, cy):
        w, h = 4 * border, 4 * border
        t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        td = ImageDraw.Draw(t)
        bbox = td.textbbox((0, 0), text, font=font)
        td.text(((w - bbox[2] - bbox[0]) / 2, (h - bbox[3] - bbox[1]) / 2), text, font=font, fill=txt_rgb + (255,))
        t = t.rotate(-90, expand=False)  # +x is "up" for the human -> rotate so text reads along +y
        img.paste(t, (int(cx - w / 2), int(cy - h / 2)), t)

    for row in range(8):            # file letters on both x-borders
        file_idx = 7 - row
        cy = border + row * px + px / 2
        put("abcdefgh"[file_idx], border / 2, cy)
        put("abcdefgh"[file_idx], size - border / 2, cy)
    for col in range(8):            # rank numbers on both y-borders
        rank = 8 - col
        cx = border + col * px + px / 2
        put(str(rank), cx, border / 2)
        put(str(rank), cx, size - border / 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


# --------------------------------------------------------------------------- SO-101 arm
def wrist_camera_frame(cfg: C.ChessConfig):
    """(pos, quat) of the wrist camera in the gripper body frame.

    The camera sits at the wrist_camera body of the camera-mount model, looks along the jaw
    (gripperframe +x) direction tilted `wrist_tilt_deg` toward the jaw axis so the fingertips
    are at the bottom of the image.  MuJoCo cameras look along -z with +y up.
    """
    if cfg.wrist_cam_calibrated and C.HAND_EYE is not None:  # measured on the real arm (calibration/hand_eye.py)
        return np.array(C.HAND_EYE["mujoco_cam_pos"], float), np.array(C.HAND_EYE["mujoco_cam_quat_wxyz"], float)
    m = mujoco.MjModel.from_xml_path(C.SO101_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = m.body("gripper").id
    Rg = d.xmat[g].reshape(3, 3)
    pg = d.xpos[g]
    cam_world = d.xpos[m.body("wrist_camera").id]
    site = m.site("gripperframe").id
    s_world = d.site_xpos[site]
    jaw_world = d.site_xmat[site].reshape(3, 3)[:, 0]
    # express in the gripper frame
    c = Rg.T @ (cam_world - pg)
    s = Rg.T @ (s_world - pg)
    j = Rg.T @ jaw_world
    j /= np.linalg.norm(j)
    lat = (s - c) - np.dot(s - c, j) * j      # from the camera toward the jaw axis, perpendicular to the jaw
    lat /= np.linalg.norm(lat)
    t = np.deg2rad(cfg.wrist_tilt_deg)
    optical = np.cos(t) * j + np.sin(t) * lat  # tilted toward the fingertips
    up = np.sin(t) * j - np.cos(t) * lat       # image up points away from the fingertips
    z = -optical
    y = up / np.linalg.norm(up)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    # roll about the optical axis (cfg.wrist_roll_deg, CCW about the camera's +z): the real module is mounted rolled
    a = np.deg2rad(cfg.wrist_roll_deg)
    R = R @ np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    # the lens sits on the front face of the 32x32 mm module, not at its centre
    return c + cfg.wrist_cam_forward * optical, quat_from_mat(R)


def load_arm(cfg: C.ChessConfig):
    """Return (defaults, assets, base_body, actuator) elements of the SO-101 with the wrist camera added."""
    tree = ET.parse(C.SO101_XML)
    root = tree.getroot()
    defaults = [copy.deepcopy(e) for e in root.findall("default")]
    asset = copy.deepcopy(root.find("asset"))
    actuator = copy.deepcopy(root.find("actuator"))
    base = copy.deepcopy(root.find("worldbody").find("body"))
    # collision masks + box fingers on the jaws (see prepare_arm_collisions)
    prepare_arm_collisions(defaults, base, measure_jaw_geometry()["fingers"])
    # wrist camera
    pos, quat = wrist_camera_frame(cfg)
    gripper = None
    for b in base.iter("body"):
        if b.get("name") == "gripper":
            gripper = b
    assert gripper is not None
    cam = ET.SubElement(gripper, "camera")
    cam.set("name", C.WRIST_CAM)
    cam.set("pos", _f(pos))
    cam.set("quat", _f(quat))
    cam.set("fovy", _f(cfg.wrist_fovy))
    return defaults, asset, base, actuator


# --------------------------------------------------------------------------- arm collision masks / jaw fingers
# Collision bits: pieces and the jaw finger boxes use contype=1 conaffinity=1; the convex hulls of the two jaw meshes
# use 2/2 (they collide with the table, the board and the other arm links, all 1/3, but NOT with the pieces or the
# finger boxes): MuJoCo collides meshes as convex hulls, and the hull of the curved fingers (+ gripper housing)
# bulges into the jaw gap with a slanted face that squeezes a piece stem out instead of holding it (measured: the
# hull touched a king stem 27 mm above the fingertips).  The box fingers reproduce the real inner-face profile.
MASK_PIECE = ("1", "1")
MASK_ARM = ("1", "3")
MASK_HULL = ("2", "2")
JAW_BODIES = {"fixed": "gripper", "moving": "moving_jaw_so101_v1"}
FINGER_BINS = ((0.0, 0.010), (0.010, 0.020), (0.020, 0.040))   # box segments, distance back from the fingertip (m)


def jaw_finger_boxes(model: mujoco.MjModel, tips: dict | None = None, bins=FINGER_BINS) -> list:
    """Box collision geoms approximating each jaw finger, derived from the jaw meshes at the closed joint limit:
    one box per `bins` segment (along the jaw, measured back from the fingertip) whose inner face lies at the
    innermost mesh point of the segment (the lip), outer face at the outermost point and width at the widest.
    Returns [dict(name, body, pos, quat, size)] in each body's local frame."""
    tips = tips or jaw_tip_vertices(model)
    data = mujoco.MjData(model)
    jid = model.joint(C.GRIPPER_JOINT).id
    data.qpos[model.jnt_qposadr[jid]] = model.jnt_range[jid, 0]
    mujoco.mj_forward(model, data)
    with np.errstate(all="ignore"):
        jaw = data.site_xmat[model.site(C.ARM_SITE).id].reshape(3, 3)[:, 0]
        axis = data.xaxis[jid].copy()
        lat = np.cross(axis, jaw)
        lat /= np.linalg.norm(lat)
        verts = {k: _mesh_world_vertices(model, data, tips["geom_" + k]) for k in ("fixed", "moving")}
        if (verts["moving"].mean(0) - verts["fixed"].mean(0)) @ lat < 0:
            lat = -lat                                  # lat: from the fixed jaw toward the moving jaw
        boxes = []
        for key, sign in (("fixed", 1.0), ("moving", -1.0)):
            tip = verts[key][tips["tip_" + key]]
            inner_dir = sign * lat                      # toward the other jaw
            R = np.column_stack([jaw, inner_dir, np.cross(jaw, inner_dir)])
            bid = model.body(JAW_BODIES[key]).id
            Rb, pb = data.xmat[bid].reshape(3, 3), data.xpos[bid]
            rel = verts[key] - tip
            along, inner, wid = -(rel @ jaw), rel @ inner_dir, rel @ R[:, 2]
            for i, (a0, a1) in enumerate(bins):
                sel = (along >= a0) & (along < a1)
                assert sel.any(), (key, a0, a1)
                c_along, c_inner, c_w = -(a0 + a1) / 2, (inner[sel].max() + inner[sel].min()) / 2, (wid[sel].max() + wid[sel].min()) / 2
                size = ((a1 - a0) / 2, (inner[sel].max() - inner[sel].min()) / 2, (wid[sel].max() - wid[sel].min()) / 2)
                centre = tip + R @ np.array([c_along, c_inner, c_w])
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, (Rb.T @ R).reshape(9))
                boxes.append(dict(name=f"{key}_finger_{i}", body=JAW_BODIES[key], pos=(Rb.T @ (centre - pb)).tolist(),
                                  quat=q.tolist(), size=[float(v) for v in size]))
    return boxes


def prepare_arm_collisions(defaults: list, base: ET.Element, fingers: list) -> None:
    """Apply the collision masks to the copied arm (class 'collision' -> 1/3, jaw hulls -> 2/2) and add the finger
    boxes (1/1) to the jaw bodies.  Works on the ElementTree copy produced by load_arm (or the raw arm XML)."""
    for d in defaults:
        for cls in d.iter("default"):
            if cls.get("class") == "collision":
                for g in cls.findall("geom"):
                    g.set("contype", MASK_ARM[0])
                    g.set("conaffinity", MASK_ARM[1])
    bodies = {b.get("name"): b for b in base.iter("body")}
    for key, bname in JAW_BODIES.items():
        k = 0
        for g in bodies[bname].findall("geom"):
            if g.get("class") == "collision":
                g.set("contype", MASK_HULL[0])
                g.set("conaffinity", MASK_HULL[1])
                g.set("name", f"{key}_hull_{k}")
                k += 1
    for f in fingers:
        ET.SubElement(bodies[f["body"]], "geom", name=f["name"], type="box", pos=_f(f["pos"]), quat=_f(f["quat"]),
                      size=_f(f["size"]), group="3", contype=MASK_PIECE[0], conaffinity=MASK_PIECE[1],
                      friction=f"{C.PIECE_FRICTION} 0.005 0.0001", condim="4", solref="0.004 1", solimp="0.95 0.99 0.001",
                      rgba="0.2 0.8 0.2 0.4", mass="0")


def finger_geom_ids(model: mujoco.MjModel) -> Tuple[list, list]:
    """Geom ids of the fixed-jaw and moving-jaw finger boxes in a model built with prepare_arm_collisions."""
    fixed = [g for g in range(model.ngeom) if (model.geom(g).name or "").startswith("fixed_finger_")]
    moving = [g for g in range(model.ngeom) if (model.geom(g).name or "").startswith("moving_finger_")]
    return fixed, moving


def jaw_clearance_from_boxes(model: mujoco.MjModel, data: mujoco.MjData, ids: Tuple[list, list] | None = None) -> float:
    """Minimum distance (m) between the fixed and moving finger boxes at the current data (what a stem feels in
    the simulation); negative when they penetrate."""
    fixed, moving = ids or finger_geom_ids(model)
    assert fixed and moving, "model has no finger boxes"
    best = np.inf
    for a in fixed:
        for b in moving:
            best = min(best, mujoco.mj_geomDistance(model, data, a, b, 0.5, None))
    return float(best)


def arm_model_with_fingers(fingers: list) -> mujoco.MjModel:
    """Compile the standalone arm with the collision masks + finger boxes (temporary XML next to the arm file so
    meshdir resolves)."""
    tree = ET.parse(C.SO101_XML)
    root = tree.getroot()
    prepare_arm_collisions(root.findall("default"), root.find("worldbody").find("body"), fingers)
    tmp = os.path.join(os.path.dirname(C.SO101_XML), "_jaw_measure_tmp.xml")
    tree.write(tmp)
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# --------------------------------------------------------------------------- jaw geometry (measured from the meshes)
JAW_MODEL_NUMERIC = "jaw_model"              # <numeric>: r_fixed r_moving theta0 dz q_closed q_open
JAW_CLEARANCE_NUMERIC = "jaw_clearance"      # <numeric>: q_0 c_0 q_1 c_1 ... (clearance table)


def _mesh_world_vertices(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    mid = model.geom_dataid[geom]
    v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = np.ascontiguousarray(model.mesh_vert[v0:v0 + nv], dtype=np.float64)
    R = np.ascontiguousarray(data.geom_xmat[geom].reshape(3, 3), dtype=np.float64)
    with np.errstate(all="ignore"):            # Accelerate's matmul on these views raises spurious fp warnings
        return np.asarray(data.geom_xpos[geom]) + np.dot(verts, R.T)


def jaw_geoms(model: mujoco.MjModel) -> Tuple[int, int]:
    """(fixed-jaw visual mesh geom on body 'gripper', moving-jaw visual mesh geom on body 'moving_jaw_so101_v1')."""
    with np.errstate(all="ignore"):
        return _jaw_geoms(model)


def _jaw_geoms(model: mujoco.MjModel) -> Tuple[int, int]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jaw_dir = data.site_xmat[model.site(C.ARM_SITE).id].reshape(3, 3)[:, 0]

    def pick(body_name):
        bid = model.body(body_name).id
        cands = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
        assert cands, body_name
        # the jaw is the visual mesh that reaches furthest along the jaw (pointing) direction
        return max(cands, key=lambda g: float((_mesh_world_vertices(model, data, g) @ jaw_dir).max()))
    return pick("gripper"), pick("moving_jaw_so101_v1")


def jaw_tip_vertices(model: mujoco.MjModel, tip_region: float = 0.012, inner_band: float = 0.004) -> dict:
    """Vertex indices (into each jaw mesh) of the fingertips: 'tip_fixed'/'tip_moving' = the most forward vertex
    along the jaw direction at the closed joint limit; 'region_fixed'/'region_moving' = all vertices within
    tip_region (m) of the forward extent (the part of the jaws that holds a piece stem) and on the inner faces."""
    with np.errstate(all="ignore"):
        return _jaw_tip_vertices(model, tip_region, inner_band)


def _jaw_tip_vertices(model: mujoco.MjModel, tip_region: float, inner_band: float) -> dict:
    gf, gm = jaw_geoms(model)
    data = mujoco.MjData(model)
    jid = model.joint(C.GRIPPER_JOINT).id
    data.qpos[model.jnt_qposadr[jid]] = model.jnt_range[jid, 0]
    mujoco.mj_forward(model, data)
    jaw_dir = data.site_xmat[model.site(C.ARM_SITE).id].reshape(3, 3)[:, 0]
    lat = np.cross(data.xaxis[jid], jaw_dir)          # jaw opening direction (perpendicular to hinge and jaw)
    lat /= np.linalg.norm(lat)
    out = dict(geom_fixed=gf, geom_moving=gm)
    verts = {key: _mesh_world_vertices(model, data, g) for key, g in (("fixed", gf), ("moving", gm))}
    # sign of `lat` pointing from the fixed jaw toward the moving jaw
    if (verts["moving"].mean(0) - verts["fixed"].mean(0)) @ lat < 0:
        lat = -lat
    for key, sign in (("fixed", 1.0), ("moving", -1.0)):
        p = verts[key] @ jaw_dir
        out["tip_" + key] = int(np.argmax(p))
        region = np.where(p > p.max() - tip_region)[0]
        # keep the inner faces only (within inner_band of the extent toward the other jaw); outer faces can never
        # be the closest pair of a rigid jaw, and this cuts the clearance search ~4x
        inner = (verts[key][region] @ lat) * sign
        region = region[inner > inner.max() - inner_band]
        out["region_" + key] = region
    return out


def jaw_gap_from_mesh(model: mujoco.MjModel, data: mujoco.MjData, tips: dict | None = None) -> float:
    """Fingertip distance (m) between the most forward points of the two jaws at the current data.qpos."""
    tips = tips or jaw_tip_vertices(model)
    a = _mesh_world_vertices(model, data, tips["geom_fixed"])[tips["tip_fixed"]]
    b = _mesh_world_vertices(model, data, tips["geom_moving"])[tips["tip_moving"]]
    return float(np.linalg.norm(a - b))


def jaw_clearance_from_mesh(model: mujoco.MjModel, data: mujoco.MjData, tips: dict | None = None) -> float:
    """Minimum distance (m) between the two jaws in the fingertip region at the current data.qpos: the widest
    stem that passes between the jaws (the inner lips are ~4 mm closer than the fingertips when closed)."""
    tips = tips or jaw_tip_vertices(model)
    vf = _mesh_world_vertices(model, data, tips["geom_fixed"])[tips["region_fixed"]]
    vm = _mesh_world_vertices(model, data, tips["geom_moving"])[tips["region_moving"]]
    best = np.inf
    for i in range(0, len(vf), 512):            # chunked to bound memory
        d = np.linalg.norm(vf[i:i + 512, None, :] - vm[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


JAW_GEOMETRY_JSON = os.path.join(C.ASSETS_DIR, "jaw_geometry.json")


def _arm_files_key() -> dict:
    """Identity of the arm model files (XML text hash + mesh sizes): the jaw-geometry cache key."""
    import hashlib
    with open(C.SO101_XML, "rb") as f:
        xml_hash = hashlib.sha1(f.read()).hexdigest()[:12]
    meshes = {n: os.path.getsize(os.path.join(C.SO101_MESH_DIR, n)) for n in sorted(os.listdir(C.SO101_MESH_DIR))
              if n.lower().endswith(".stl")}
    return dict(xml=xml_hash, meshes=meshes)


def measure_jaw_geometry(model: mujoco.MjModel | None = None, n_table: int = 41, cache: bool = True) -> dict:
    """Measure the gripper's jaw kinematics from the meshes of the arm model (default: C.SO101_XML).

    Fingertip gap: both fingertips lie at fixed radii from the gripper hinge, so their distance obeys the law of
    cosines  gap(q)^2 = dz^2 + r_f^2 + r_m^2 - 2 r_f r_m cos(theta0 + q - q_closed)  (exact to < 0.01 mm).
    Clearance: a (q, clearance) table of the minimum distance between the box fingers that the scene uses as the
    jaws' collision geometry (jaw_finger_boxes, derived from the meshes), i.e. the widest stem that passes between
    the simulated jaws / the angle the jaws stall at on a stem.  'fingers' holds the box specs.
    Returns dict(r_fixed, r_moving, theta0, dz, q_closed, q_open, table_q, table_clearance, gap_closed, gap_open,
    table_clearance_mesh, fingers).
    For the default arm model (model=None) the result is cached in assets/jaw_geometry.json keyed by the arm files."""
    key = None
    if model is None:
        key = _arm_files_key()
        if cache and os.path.exists(JAW_GEOMETRY_JSON):
            try:
                with open(JAW_GEOMETRY_JSON) as f:
                    rec = json.load(f)
                if (rec.get("arm_files") == key and rec.get("n_table") == n_table and "fingers" in rec.get("geometry", {})
                        and "table_levels" in rec.get("geometry", {})):
                    return rec["geometry"]
            except (OSError, ValueError, KeyError):
                pass
        model = mujoco.MjModel.from_xml_path(C.SO101_XML)
    data = mujoco.MjData(model)
    jid = model.joint(C.GRIPPER_JOINT).id
    qadr = model.jnt_qposadr[jid]
    q_closed, q_open = (float(v) for v in model.jnt_range[jid])
    tips = jaw_tip_vertices(model)

    def at(q):
        data.qpos[:] = 0.0
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)

    at(q_closed)
    axis, anchor = data.xaxis[jid].copy(), data.xanchor[jid].copy()
    a = _mesh_world_vertices(model, data, tips["geom_fixed"])[tips["tip_fixed"]]
    b = _mesh_world_vertices(model, data, tips["geom_moving"])[tips["tip_moving"]]
    ra, rb = a - anchor, b - anchor
    za, zb = float(ra @ axis), float(rb @ axis)
    pa, pb = ra - za * axis, rb - zb * axis
    r_fixed, r_moving = float(np.linalg.norm(pa)), float(np.linalg.norm(pb))
    theta0 = float(np.arccos(np.clip(pa @ pb / (r_fixed * r_moving), -1.0, 1.0)))
    # the hinge angle must increase theta (otherwise the sign convention of the joint axis is flipped)
    at(q_closed + 0.2)
    b2 = _mesh_world_vertices(model, data, tips["geom_moving"])[tips["tip_moving"]] - anchor
    pb2 = b2 - float(b2 @ axis) * axis
    th2 = float(np.arccos(np.clip(pa @ pb2 / (r_fixed * np.linalg.norm(pb2)), -1.0, 1.0)))
    assert abs((th2 - theta0) - 0.2) < 1e-3, "jaw hinge does not open the tips with +q"
    table_q = np.linspace(q_closed, q_open, n_table)
    table_mesh = []
    for q in table_q:
        at(q)
        table_mesh.append(jaw_clearance_from_mesh(model, data, tips))
    at(q_closed)
    gap_closed = jaw_gap_from_mesh(model, data, tips)
    at(q_open)
    gap_open = jaw_gap_from_mesh(model, data, tips)
    # finger boxes and the clearance between them (the simulation's jaw clearance)
    fingers = jaw_finger_boxes(model, tips)
    fm = arm_model_with_fingers(fingers)
    fd = mujoco.MjData(fm)
    fids = finger_geom_ids(fm)
    fq = fm.jnt_qposadr[fm.joint(C.GRIPPER_JOINT).id]
    table_c = []
    table_levels = []
    for q in table_q:
        fd.qpos[:] = 0.0
        fd.qpos[fq] = q
        mujoco.mj_forward(fm, fd)
        table_c.append(jaw_clearance_from_boxes(fm, fd, fids))
        # per finger level (box 0 = pads, 1, 2): the distance between the fixed and moving box of that level (the
        # gap a piece feels at each height above the tips; the grasp rule needs how much wider boxes 1/2 are)
        table_levels.append([float(mujoco.mj_geomDistance(fm, fd, fids[0][i], fids[1][i], 0.5, None)) for i in range(len(fids[0]))])
    geom = dict(r_fixed=r_fixed, r_moving=r_moving, theta0=theta0, dz=za - zb, q_closed=q_closed, q_open=q_open,
                table_q=[float(v) for v in table_q], table_clearance=[float(v) for v in table_c],
                gap_closed=gap_closed, gap_open=gap_open, table_clearance_mesh=[float(v) for v in table_mesh],
                fingers=fingers, table_levels=table_levels)
    if key is not None and cache:
        os.makedirs(os.path.dirname(JAW_GEOMETRY_JSON), exist_ok=True)
        with open(JAW_GEOMETRY_JSON, "w") as f:
            json.dump(dict(arm_files=key, n_table=n_table, geometry=geom), f, indent=1)
    return geom


def embedded_numeric(model: mujoco.MjModel, name: str) -> np.ndarray | None:
    """A <custom><numeric> array recorded in a compiled model (None if absent)."""
    for i in range(model.nnumeric):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, i) == name:
            adr, size = model.numeric_adr[i], model.numeric_size[i]
            return np.array(model.numeric_data[adr:adr + size], float)
    return None


def embedded_jaw_geometry(model: mujoco.MjModel) -> dict | None:
    """The jaw geometry recorded in a scene XML by build_xml (same keys as measure_jaw_geometry, no gap_*)."""
    jm, tab = embedded_numeric(model, JAW_MODEL_NUMERIC), embedded_numeric(model, JAW_CLEARANCE_NUMERIC)
    if jm is None or tab is None:
        return None
    return dict(r_fixed=jm[0], r_moving=jm[1], theta0=jm[2], dz=jm[3], q_closed=jm[4], q_open=jm[5],
                table_q=tab[0::2].tolist(), table_clearance=tab[1::2].tolist())


def jaw_geometry_agrees(a: dict, b: dict, tol_m: float = 2e-4) -> bool:
    """True if two jaw-geometry records give the same fingertip gap and clearance (within tol_m) over the joint range."""
    q = np.linspace(a["q_closed"], a["q_open"], 61)

    def gap(g):
        th = g["theta0"] + q - g["q_closed"]
        return np.sqrt(g["dz"] ** 2 + g["r_fixed"] ** 2 + g["r_moving"] ** 2 - 2 * g["r_fixed"] * g["r_moving"] * np.cos(th))

    def clr(g):
        return np.interp(q, g["table_q"], g["table_clearance"])
    return bool(np.max(np.abs(gap(a) - gap(b))) <= tol_m and np.max(np.abs(clr(a) - clr(b))) <= tol_m)


# --------------------------------------------------------------------------- pieces
def piece_visual_geoms(ptype: str, H: float, D: float):
    """List of (type, size, pos) tuples describing the visual assembly of a piece (z from the base)."""
    r = D / 2
    g = []
    g.append(("cylinder", (r, 0.06 * H), (0, 0, 0.06 * H)))                 # base disc
    g.append(("cylinder", (0.42 * D, 0.04 * H), (0, 0, 0.16 * H)))          # collar
    stem_top = {"pawn": 0.72, "rook": 0.78, "knight": 0.55, "bishop": 0.68, "queen": 0.68, "king": 0.78}[ptype] * H
    g.append(("cylinder", (0.27 * D, (stem_top - 0.20 * H) / 2), (0, 0, (stem_top + 0.20 * H) / 2)))
    if ptype == "pawn":
        rs = 0.28 * D
        g.append(("sphere", (rs,), (0, 0, H - rs)))
    elif ptype == "rook":
        g.append(("cylinder", (0.45 * D, (H - stem_top) / 2), (0, 0, (H + stem_top) / 2)))
    elif ptype == "knight":
        g.append(("box", (0.18 * D, 0.2 * D, 0.14 * H), (0.05 * D, 0, stem_top + 0.14 * H)))    # neck
        g.append(("box", (0.36 * D, 0.2 * D, 0.11 * H), (0.22 * D, 0, H - 0.11 * H)))          # head (offset +x)
    elif ptype == "bishop":
        rs = 0.32 * D
        g.append(("sphere", (rs,), (0, 0, H - 0.1 * D - rs)))
        g.append(("sphere", (0.1 * D,), (0, 0, H - 0.1 * D)))
    elif ptype == "queen":
        rs = 0.33 * D
        g.append(("sphere", (rs,), (0, 0, H - 0.07 * H - rs)))
        g.append(("cylinder", (0.42 * D, 0.03 * H), (0, 0, H - 0.03 * H)))                     # crown ring
    elif ptype == "king":
        g.append(("cylinder", (0.38 * D, 0.03 * H), (0, 0, stem_top + 0.03 * H)))
        g.append(("box", (0.045 * D, 0.045 * D, 0.08 * H), (0, 0, H - 0.08 * H)))              # cross vertical
        g.append(("box", (0.2 * D, 0.045 * D, 0.045 * D), (0, 0, H - 0.1 * H)))                # cross horizontal
    return g


# --------------------------------------------------------------------------- piece meshes: profiles, bands, hulls
def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices (N,3) m, triangles (M,3) int) of a triangulated OBJ (positions only)."""
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):          # fan-triangulate polygons
                    F.append([idx[0], idx[k], idx[k + 1]])
    return np.array(V, float), np.array(F, int)


def piece_mesh_path(ptype: str) -> str:
    return os.path.join(C.PIECE_MESH_DIR, f"{ptype}.obj")


def section_points(V: np.ndarray, F: np.ndarray, z: float) -> np.ndarray:
    """Points (K,3) where the mesh's triangle edges cross the plane z = const (the exact planar cross-section)."""
    P = V[F]
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = P[:, i], P[:, j]
        da, db = a[:, 2] - z, b[:, 2] - z
        m = (da * db) < 0
        if m.any():
            t = da[m] / (da[m] - db[m])
            pts.append(a[m] + (b[m] - a[m]) * t[:, None])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def caliper_width(xy: np.ndarray) -> float:
    """Maximum pairwise distance of 2-D points (the manifest's 'max_xy_distance')."""
    if len(xy) < 2:
        return 0.0
    hull = np.ascontiguousarray(xy, dtype=float)
    if len(xy) > 400:               # keep the extreme points along many directions (enough for the diameter)
        ang = np.linspace(0, np.pi, 90, endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        with np.errstate(all="ignore"):     # Accelerate's matmul raises spurious fp flags on some views
            proj = np.dot(hull, dirs.T)
        hull = hull[np.unique(np.concatenate([proj.argmax(0), proj.argmin(0)]))]
    d = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)
    return float(d.max())


def width_profile(V: np.ndarray, F: np.ndarray, step: float = 0.0001, axis_deg: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """(z, width) of the mesh every `step` m (planar cuts): the caliper width, or with axis_deg the extent of the
    section along that direction (deg from +x in the piece frame)."""
    H = float(V[:, 2].max())
    zs = np.arange(step / 2, H, step)
    if axis_deg is None:
        ws = np.array([caliper_width(section_points(V, F, z)[:, :2]) for z in zs])
    else:
        d = np.array([math.cos(math.radians(axis_deg)), math.sin(math.radians(axis_deg))])
        ws = []
        for z in zs:
            s = section_points(V, F, z)[:, :2]
            p = s @ d if len(s) else np.zeros(1)
            ws.append(float(p.max() - p.min()) if len(s) else 0.0)
        ws = np.array(ws)
    return zs, ws


def measure_axis_radii(ptype: str, step: float = 0.0001) -> Tuple[float, ...]:
    """config.PIECE_AXIS_RADIUS_MM for one type: per manifest slice the largest distance (mm, rounded up to
    0.01) from the piece axis over the slice's mesh vertices AND over exact planar cuts every `step` m (a slice
    without a vertex at its widest cut, e.g. a long cone edge, is under-reported by the vertices alone)."""
    V, F = load_obj(piece_mesh_path(ptype))
    out = []
    for z0, z1, _ in C.PIECE_PROFILE[ptype]["slices"]:
        sel = V[(V[:, 2] >= z0 - 1e-9) & (V[:, 2] <= z1 + 1e-9)]
        r = float(np.linalg.norm(sel[:, :2], axis=1).max()) if len(sel) else 0.0
        for z in np.arange(z0 + step / 2, z1, step):
            s = section_points(V, F, z)
            if len(s):
                r = max(r, float(np.linalg.norm(s[:, :2], axis=1).max()))
        out.append(math.ceil(r * 1e5 - 1e-6) / 100.0)
    return tuple(out)


def convex_hull_mujoco(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of the convex hull of a point cloud, computed by MuJoCo (qhull) from a vertex= mesh asset:
    a mesh without faces is replaced by its hull at compile time."""
    pts = np.asarray(points, float)
    c = pts.mean(0)
    vtxt = " ".join(f"{v:.7g}" for v in (pts - c).ravel())
    xml = (f'<mujoco><asset><mesh name="m" vertex="{vtxt}"/></asset>'
           f'<worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    nv, nf = m.mesh_vertnum[0], m.mesh_facenum[0]
    verts = np.array(m.mesh_vert[m.mesh_vertadr[0]:m.mesh_vertadr[0] + nv], float)
    faces = np.array(m.mesh_face[m.mesh_faceadr[0]:m.mesh_faceadr[0] + nf], int)
    # the compiler expresses the vertices in the mesh's inertial frame: map them back to the input frame
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.mesh_quat[0])
    with np.errstate(all="ignore"):
        verts = np.dot(np.ascontiguousarray(verts), np.ascontiguousarray(R.reshape(3, 3)).T) + m.mesh_pos[0] + c
    return verts, faces


def hull_band_error(V: np.ndarray, F: np.ndarray, zs: np.ndarray, ws: np.ndarray, z0: float, z1: float):
    """(max hull-minus-profile width error, height of the worst sample, hull vertices, hull faces) of the convex
    hull of the mesh band [z0, z1], measured on the actual hull (planar cuts of the hull at the profile samples)."""
    hv, hf = convex_hull_mujoco(band_vertices(V, F, z0, z1))
    m = (zs >= z0 - 1e-9) & (zs <= z1 + 1e-9)
    if not m.any():
        return 0.0, None, hv, hf
    # reference: the exact real width at each height, or the widest section within BAND_STEP_TOL ABOVE it when
    # that is wider (an undercut: the knight's muzzle underside, the crown points, the king's cross arms, the
    # bishop's mitre).  A convex hull is at least as wide as the widest section above, so an undercut cannot be
    # followed by any band stack; a concave slope (the rook's cove between foot and stem) can and must be.
    k = max(1, int(round(BAND_STEP_TOL / (zs[1] - zs[0]))))
    ref = np.array([ws[i:i + k + 1].max() for i in np.where(m)[0]])
    err = np.array([caliper_width(section_points(hv, hf, z)[:, :2]) for z in zs[m]]) - ref
    i = int(err.argmax())
    return float(err[i]), float(zs[m][i]), hv, hf


BAND_STEP_TOL = 0.0005        # m, upward window of the band accuracy reference (undercuts), see hull_band_error


def write_obj(path: str, V: np.ndarray, F: np.ndarray, comment: str = "") -> None:
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        for v in V:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for t in F:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def hull_width_profile(zs: np.ndarray, ws: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Width of the convex hull of the profile band [z0, z1] at the sample heights inside it: the upper concave
    envelope of (z, w) over the band (the hull of a stack of discs is the revolution of the 2-D hull)."""
    m = (zs >= z0 - 1e-9) & (zs <= z1 + 1e-9)
    z, w = zs[m], ws[m]
    if len(z) < 2:
        return w
    # upper hull of the points (z, w) by a monotone chain
    pts = list(zip(z, w))
    up = []
    for p in pts:
        while len(up) >= 2 and (up[-1][0] - up[-2][0]) * (p[1] - up[-2][1]) - (up[-1][1] - up[-2][1]) * (p[0] - up[-2][0]) >= 0:
            up.pop()
        up.append(p)
    ux, uy = np.array([p[0] for p in up]), np.array([p[1] for p in up])
    return np.interp(z, ux, uy)


def slice_bands(V: np.ndarray, F: np.ndarray, zs: np.ndarray, ws: np.ndarray, H: float, tol: float = C.BAND_TOL,
                min_h: float = C.BAND_MIN_HEIGHT, max_n: int = C.BAND_MAX_COUNT) -> Tuple[List[float], dict]:
    """Cut heights [0, ..., H] of the collision bands of a mesh: start with the local minima of the (lightly
    smoothed) width profile (a hull would bridge a neck), then split the band with the largest hull-vs-profile
    width error (measured on the real convex hull) at its worst height until every band is within tol (or max_n
    bands); cuts closer than min_h are merged.  Returns (cuts, {(z0, z1): (error, hull verts, hull faces)})."""
    k = 3
    sm = np.convolve(np.pad(ws, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    cuts = [0.0, H]
    for i in range(1, len(zs) - 1):
        if sm[i] < sm[i - 1] - 1e-6 and sm[i] <= sm[i + 1] + 1e-6 and (ws[i] < ws[max(0, i - 8):i + 9].max() - 0.0005):
            cuts.append(float(zs[i]))

    def merged(cs):
        cs = sorted(set(cs))
        out = [cs[0]]
        for c in cs[1:]:
            if c - out[-1] < min_h:
                if c == H:
                    out[-1] = c
                continue
            out.append(c)
        if out[-1] != H:
            out.append(H)
        return out
    cuts = merged(cuts)
    cache: dict = {}
    while True:
        worst = (0.0, None)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if (a, b) not in cache:
                cache[(a, b)] = hull_band_error(V, F, zs, ws, a, b)
            err, z_bad, _, _ = cache[(a, b)]
            if z_bad is not None and err > worst[0]:
                worst = (err, z_bad)
        if worst[0] <= tol or worst[1] is None or len(cuts) - 1 >= max_n:
            break
        new = merged(cuts + [worst[1]])
        if new == cuts:                 # the split point is within min_h of an existing cut: accept the residual
            break
        cuts = new
    return cuts, cache


def band_vertices(V: np.ndarray, F: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Vertices of the mesh inside [z0, z1] plus the exact cross-sections at both cuts (so the hull caps are exact)."""
    inside = V[(V[:, 2] >= z0 - 1e-9) & (V[:, 2] <= z1 + 1e-9)]
    parts = [inside]
    for z in (z0, z1):
        s = section_points(V, F, z)
        if len(s):
            parts.append(s)
    pts = np.concatenate(parts)
    pts[:, 2] = np.clip(pts[:, 2], z0, z1)
    return pts


def build_piece_bands(out_dir: str = C.PIECE_BAND_DIR, types=C.PIECE_TYPES, verbose: bool = True) -> dict:
    """Slice every piece mesh into horizontal bands and write one convex-hull OBJ per band into out_dir (+
    bands.json: cuts, hull vertex/face counts, the max hull-vs-profile width error).  Returns the record."""
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(tol_m=C.BAND_TOL, pieces={})
    for t in types:
        for old in os.listdir(out_dir):                 # stale hulls of an earlier slicing (fewer / more bands)
            if old.startswith(f"{t}_band") and old.endswith(".obj"):
                os.remove(os.path.join(out_dir, old))
        V, F = load_obj(piece_mesh_path(t))
        H = float(V[:, 2].max())
        zs, ws = width_profile(V, F)
        cuts, cache = slice_bands(V, F, zs, ws, H)
        bands = []
        worst = 0.0
        for i, (z0, z1) in enumerate(zip(cuts[:-1], cuts[1:])):
            err, _, hv, hf = cache[(z0, z1)]
            path = os.path.join(out_dir, f"{t}_band{i}.obj")
            write_obj(path, hv, hf, f"{t} collision band {i}: convex hull of the mesh between z = {z0*1000:.2f} and {z1*1000:.2f} mm (metres)")
            m = (zs >= z0 - 1e-9) & (zs <= z1 + 1e-9)
            worst = max(worst, err)
            bands.append(dict(index=i, z0=z0, z1=z1, file=os.path.basename(path), hull_vertices=int(len(hv)), hull_faces=int(len(hf)),
                              width_error_m=err, width_max_m=float(ws[m].max()) if m.any() else 0.0))
        rec["pieces"][t] = dict(height=H, cuts=cuts, bands=bands, max_width_error_m=worst, n_bands=len(bands))
        if verbose:
            print(f"  {t:7s}: {len(bands)} bands, cuts {[round(c*1000, 2) for c in cuts]} mm, max hull-vs-profile width error {worst*1000:.2f} mm "
                  f"(per band: {[round(b['width_error_m']*1000, 2) for b in bands]})")
    with open(os.path.join(out_dir, "bands.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def load_piece_bands(out_dir: str = C.PIECE_BAND_DIR) -> dict | None:
    try:
        with open(os.path.join(out_dir, "bands.json")) as f:
            rec = json.load(f)
        if abs(rec.get("tol_m", -1) - C.BAND_TOL) > 1e-12:
            return None
        for t, p in rec["pieces"].items():
            for b in p["bands"]:
                if not os.path.exists(os.path.join(out_dir, b["file"])):
                    return None
        return rec
    except (OSError, ValueError, KeyError):
        return None


def ensure_piece_bands(verbose: bool = False) -> dict:
    rec = load_piece_bands()
    if rec is None or set(rec["pieces"]) != set(C.PIECE_TYPES):
        rec = build_piece_bands(verbose=verbose)
    return rec


# --------------------------------------------------------------------------- grasp table ('bands' collision model)
PAD_HEIGHT = FINGER_BINS[0][1]            # the fingertip pads: finger box 0 spans tip .. tip + 10 mm along the jaw
GRASP_PAD_SLOPE_MAX = 0.5                 # max radius/height slope under a pad EDGE (friction cone: tan < mu = 1, 2x margin)
GRASP_RING_MIN = 0.0010                   # a stall feature thinner than this (height with width >= max - 1 mm) is a thin ring
GRASP_LEVEL_MARGIN = 0.0010               # finger boxes 1 / 2 must clear the piece above the pads by this
GRASP_CONTACT_TOL = 0.0012                # sections within this of the stall width count as pad contact (the soft contacts
                                          # sink ~0.9 mm per side at the squeeze force: measured stall gap = width - 1.8 mm)
GRASP_CONTACT_MIN = 0.0040                # the contact must span this height (two features or a cylinder): rigid hold
GRASP_STOP_BELOW = 0.0008                 # a wider section this far below the tips still stops the swing (bishop collar:
                                          # 0.65 mm below the tips holds it at < 3 deg, 0.85 mm lets it swing 5 deg, 1.5 mm 9 deg)
# reach caps (m) measured with the IK (test_motion a): above these the far corners a1/h1 need more than the 8 deg lean
GRASP_T_MAX_ABS = {"king": 0.0294}
GRASP_ZONE_WIDTH_TOL = 0.00025            # tip heights whose stall width is within this of the narrowest form the zone
GRASP_FEATURE_JUMP = 0.006                # a jump of the stall height larger than this (> a manifest slice) is another feature
GRASP_T_MIN_FRAC = 0.30                   # lowest tip height considered (above the foot)
GRASP_T_MAX_FRAC = 0.70                   # highest tip height considered (the top knobs / the king's cross are no grasp features)
GRASP_T_MAX_BELOW_TOP = 0.008             # the pads must overlap the piece by at least this (no pinch on a top knob)
# pieces that are not solids of revolution are grasped across their narrow axis: {type: opening direction in the
# piece frame (deg from +x)}; the width profile then uses the slices' extent along that axis
GRASP_AXIS_LOCAL_DEG = {"knight": 90.0}   # the knight's head is 19.7 mm long (+x, muzzle) but 10 mm across (y)


def manifest_width_fn(ptype: str):
    """w(z): width (m) of a piece at height z from the manifest: the 1 %-step stem profile over 45..75 % (linear
    interpolation; the caliper width), the slice maximum elsewhere (conservative), 0 outside the piece.  For a
    type in GRASP_AXIS_LOCAL_DEG the slices' extent along the grasp axis is used everywhere (the stem profile only
    has the caliper width, which for the knight is its head length)."""
    p = C.PIECE_PROFILE[ptype]
    sz = np.array([z for z, _ in p["stem"]])
    sw = np.array([w for _, w in p["stem"]])
    H = p["height"]
    axis = GRASP_AXIS_LOCAL_DEG.get(ptype)
    if axis is None:
        slices = p["slices"]
    else:
        key = "slices_y" if abs((axis % 180.0) - 90.0) < 45.0 else "slices_x"
        slices = p[key]

    def w(z):
        z = float(z)
        if z < 0.0 or z > H:
            return 0.0
        if axis is None and sz[0] - 1e-9 <= z <= sz[-1] + 1e-9:
            return float(np.interp(z, sz, sw))
        for z0, z1, ww in slices:
            if z <= z1 + 1e-9:
                return ww
        return slices[-1][2]
    return w


def finger_level_extra(jg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pad gap, box-1 gap - pad gap, box-2 gap - pad gap) tables from the measured jaw geometry 'levels'."""
    lv = np.array(jg["table_levels"], float)      # (n, 3): box-box distance per finger level at table_q
    return lv[:, 0], lv[:, 1] - lv[:, 0], lv[:, 2] - lv[:, 0]


def grasp_table(types=C.PIECE_TYPES, jg: dict | None = None, step: float = 0.00025, verbose: bool = False,
                use_mesh: bool = True) -> Dict[str, Dict[str, float]]:
    """Derive GRASP_TABLE_BANDS from the piece width profile and the measured finger-box levels.  The profile is the
    mesh's own planar cuts (0.1 mm; the geometry the manifest's stem profile and 10 slices were measured from,
    which are too coarse above 75 % of the height for the shape tests) or, with use_mesh=False, the manifest's
    stem profile / slice maxima (conservative).  For every tip height t (the tool point = the fixed fingertip) the
    pads cover [t, t + PAD_HEIGHT]; the stall width is the widest section there.  A candidate is rejected when (a)
    the stall is at a pad edge on a slope steeper than GRASP_PAD_SLOPE_MAX (a flare: the squeeze pushes the piece
    out), (b) the stall feature is a thin ring, (b2) the pads would touch a single narrow feature only (the piece
    swings: contact extent < GRASP_CONTACT_MIN), (c) finger box 1 (tip + 10..20 mm) or box 2 (tip + 20..40 mm),
    which at the stall are wider than the pads by the measured level extras, would touch the piece (collar / head)
    above.  The narrowest accepted width defines the zone (all accepted t within GRASP_ZONE_WIDTH_TOL of it that
    stall on the same feature); the lowest such zone wins and the table height is its centre.  A type in
    GRASP_AXIS_LOCAL_DEG is measured along that axis (the knight across its head).
    Returns {type: dict(height, width, envelope, zone_lo, zone_hi, axis_deg)} in metres."""
    jg = jg or measure_jaw_geometry()
    gaps, ex1, ex2 = finger_level_extra(jg)
    out = {}
    for t in types:
        p = C.PIECE_PROFILE[t]
        H = p["height"]
        axis = GRASP_AXIS_LOCAL_DEG.get(t)
        if use_mesh:
            V, F = load_obj(piece_mesh_path(t))
            zs, ws = width_profile(V, F, axis_deg=axis)
            step = float(zs[1] - zs[0])
            wfn = lambda z, zs=zs, ws=ws, H=H: float(np.interp(z, zs, ws)) if 0.0 <= z <= H else 0.0
        else:
            wfn = manifest_width_fn(t)
            zs = np.arange(step / 2, H, step)
            ws = np.array([wfn(z) for z in zs])
        cands = []
        notes = {}
        t_max = min(GRASP_T_MAX_FRAC * H, H - GRASP_T_MAX_BELOW_TOP, GRASP_T_MAX_ABS.get(t, 1.0))
        for tip in np.arange(GRASP_T_MIN_FRAC * H, t_max + 1e-9, step):
            win = (zs >= tip - 1e-9) & (zs <= tip + PAD_HEIGHT + 1e-9)
            if not win.any():
                continue
            w = float(ws[win].max())
            zstar = float(zs[win][int(ws[win].argmax())])
            why = None
            # (a) rigidity: a single narrow feature (a ring, a ball's equator, a rim, a flare at a pad edge) between
            # flat pads lets the piece swing like a pendulum or slide (measured 6..12 deg of tilt while carried); the
            # pads must touch over >= GRASP_CONTACT_MIN of height: a cylinder, or two features (the king's collar +
            # crown rim, the pawn's collar + head, the queen's ring + crown cone) that lock each other, counting a
            # wider feature just below the tips (it stops the swing at the pad bottom edge)
            zone_m = (zs >= tip - GRASP_STOP_BELOW - 1e-9) & (zs <= tip + PAD_HEIGHT + 1e-9) & (ws >= w - GRASP_CONTACT_TOL)
            ext = float(zs[zone_m].max() - zs[zone_m].min()) + step if zone_m.any() else 0.0
            if ext < GRASP_CONTACT_MIN:
                why = f"single-feature pinch (contact extent {ext*1000:.1f} mm)"
            # (b) a two-feature hold with the lower feature on the friction-limited flare of the foot (the pads would
            # ride up the cone until the upper feature locks: the piece ends higher in the jaws than planned)
            if why is None and zstar - tip < 0.001:
                below = wfn(max(0.0, tip - 0.001))
                if (below - w) / 2.0 / 0.001 > GRASP_PAD_SLOPE_MAX:
                    why = f"flare at the pad bottom edge (slope {(below - w) / 2.0 / 0.001:.2f})"
            # (c) room for finger boxes 1 and 2 above the pads
            if why is None:
                e1 = float(np.interp(w, gaps, ex1))
                e2 = float(np.interp(w, gaps, ex2))
                m1 = (zs > tip + PAD_HEIGHT) & (zs <= tip + FINGER_BINS[1][1])
                m2 = (zs > tip + FINGER_BINS[1][1]) & (zs <= tip + FINGER_BINS[2][1])
                if m1.any() and ws[m1].max() > w + e1 - GRASP_LEVEL_MARGIN:
                    why = f"box 1 would touch the piece above the pads ({ws[m1].max()*1000:.1f} > {w*1000:.1f} + {e1*1000:.1f} - 1 mm)"
                elif m2.any() and ws[m2].max() > w + e2 - GRASP_LEVEL_MARGIN:
                    why = f"box 2 would touch the piece above the pads ({ws[m2].max()*1000:.1f} > {w*1000:.1f} + {e2*1000:.1f} - 1 mm)"
            if why is None:
                cands.append((float(tip), w, zstar))
            else:
                notes[round(float(tip) * 1000, 2)] = why
        if not cands:
            raise RuntimeError(f"grasp_table: no admissible tip height for the {t}: {notes}")
        w_min = min(c[1] for c in cands)
        zone = [c for c in cands if c[1] <= w_min + GRASP_ZONE_WIDTH_TOL]
        # zones: runs of contiguous tip heights that stall on the SAME feature (z* continuous); the king's collar
        # and crown rim have the same width but are different features.  The lowest zone wins (reach near the base)
        runs: List[List[tuple]] = [[zone[0]]]
        for c in zone[1:]:
            if c[0] - runs[-1][-1][0] <= step * 1.5 and abs(c[2] - runs[-1][-1][2]) <= GRASP_FEATURE_JUMP:
                runs[-1].append(c)
            else:
                runs.append([c])
        best = runs[0]
        z_lo, z_hi = best[0][0], best[-1][0]
        height = round((z_lo + z_hi) / 2.0 / 0.0001) * 0.0001
        win = (zs >= height - 1e-9) & (zs <= height + PAD_HEIGHT + 1e-9)
        width = float(ws[win].max())
        if axis is None:
            envelope = 2.0 * C.piece_max_radius(t, height, "bands")
        else:
            # the fixed finger descends on the grasp axis side: the extent along that axis above the tips
            envelope = max([float(ws[zs >= height - 1e-9].max()), width])
        out[t] = dict(height=round(height, 5), width=round(width, 5), envelope=round(envelope, 5), zone_lo=round(z_lo, 5),
                      zone_hi=round(z_hi, 5), axis_deg=axis)
        if axis is not None:
            # directional extents (mm, per 10 deg of the direction theta in the piece frame): how far the piece
            # reaches from its axis toward theta inside the pad window (the stall side) and anywhere above the tips
            # (what the open fixed finger must clear while descending) -- a non-symmetric piece grasped along its
            # head has its axis at the fixed-side extent, not at half the width
            V, F = load_obj(piece_mesh_path(t))
            win = np.ascontiguousarray(V[(V[:, 2] >= height - 1e-9) & (V[:, 2] <= height + PAD_HEIGHT + 1e-9)][:, :2])
            above = np.ascontiguousarray(V[V[:, 2] >= height - 1e-9][:, :2])
            ext = []
            for th in range(0, 360, 10):
                d = np.array([math.cos(math.radians(th)), math.sin(math.radians(th))])
                with np.errstate(all="ignore"):
                    ext.append((round(float(np.dot(win, d).max()) * 1000, 2), round(float(np.dot(above, d).max()) * 1000, 2)))
            out[t]["extent_mm"] = ext
        if verbose:
            print(f"  {t:7s}: tip {height*1000:5.2f} mm ({height / H * 100:4.1f} %) stall width {width*1000:5.2f} mm  envelope {envelope*1000:5.2f} mm  "
                  f"zone {z_lo*1000:.2f}..{z_hi*1000:.2f} mm; {len(cands)} admissible tip heights"
                  + (f"; across the piece's {axis:.0f} deg axis" if axis is not None else ""))
    return out


def add_piece_body(worldbody: ET.Element, name: str, colour: str, ptype: str, xyz, cfg: C.ChessConfig, bands: dict | None = None):
    H, D, mass = C.PIECE_SPECS[ptype]
    mass *= cfg.piece_mass_scale
    body = ET.SubElement(worldbody, "body", name=name, pos=_f(xyz))
    ET.SubElement(body, "freejoint", name=f"{name}_joint")
    # explicit mass (the spec's) and the inertia of a solid cylinder r = 0.4 D, height H about its centre: the
    # collision / visual geoms carry no mass in either model
    rr = 0.4 * D
    ixx = mass * (3 * rr * rr + H * H) / 12.0
    izz = mass * rr * rr / 2.0
    ET.SubElement(body, "inertial", pos=_f((0, 0, 0.42 * H)), mass=_f(mass), diaginertia=_f((ixx, ixx, izz)))
    rgba = C.WHITE_RGBA if colour == "w" else C.BLACK_RGBA
    material = "piece_white" if colour == "w" else "piece_black"
    # visuals (group 1, no collision): the mesh skin or the primitive assembly
    if cfg.piece_visual == "mesh":
        ET.SubElement(body, "geom", {"class": "piece_vis"}, name=f"{name}_vis_mesh", type="mesh", mesh=f"piece_{ptype}",
                      material=material, rgba=_f(rgba))
    else:
        for i, (gt, size, pos) in enumerate(piece_visual_geoms(ptype, H, D)):
            ET.SubElement(body, "geom", {"class": "piece_vis"}, name=f"{name}_vis{i}", type=gt,
                          size=_f(size), pos=_f(pos), material=material, rgba=_f(rgba))
    if cfg.piece_collision == "bands":
        # one convex hull per horizontal band of the mesh (assets/pieces/bands, build_piece_bands)
        assert bands is not None
        for b in bands["pieces"][ptype]["bands"]:
            ET.SubElement(body, "geom", {"class": "piece_col"}, name=f"{name}_col_band{b['index']}", type="mesh",
                          mesh=f"band_{ptype}_{b['index']}")
    else:
        # collision: base cylinder (30 % of the height) + stem cylinder (60 % of the base diameter)
        hb = C.COLLISION_BASE_HEIGHT_FRAC * H
        ET.SubElement(body, "geom", {"class": "piece_col"}, name=f"{name}_col_base",
                      size=_f((D / 2, hb / 2)), pos=_f((0, 0, hb / 2)))
        ET.SubElement(body, "geom", {"class": "piece_col"}, name=f"{name}_col_stem",
                      size=_f((C.COLLISION_STEM_DIA_FRAC * D / 2, (H - hb) / 2)), pos=_f((0, 0, (H + hb) / 2)))
    return body


START_SQUARES = {
    "w": {"pawn": [f"{f}2" for f in "abcdefgh"], "rook": ["a1", "h1"], "knight": ["b1", "g1"],
          "bishop": ["c1", "f1"], "queen": ["d1"], "king": ["e1"]},
    "b": {"pawn": [f"{f}7" for f in "abcdefgh"], "rook": ["a8", "h8"], "knight": ["b8", "g8"],
          "bishop": ["c8", "f8"], "queen": ["d8"], "king": ["e8"]},
}


# --------------------------------------------------------------------------- scene
def build_xml(cfg: C.ChessConfig = C.DEFAULT_CONFIG, path: str = C.SCENE_XML) -> ET.Element:
    scene_dir = os.path.dirname(os.path.abspath(path))
    # meshdir = the assets directory: arm meshes under so101/assets, piece meshes under pieces/ (+ pieces/bands)
    meshdir = os.path.relpath(C.ASSETS_DIR, scene_dir)
    arm_mesh_prefix = os.path.relpath(C.SO101_MESH_DIR, C.ASSETS_DIR).replace(os.sep, "/")
    piece_prefix = os.path.relpath(C.PIECE_MESH_DIR, C.ASSETS_DIR).replace(os.sep, "/")
    band_prefix = os.path.relpath(C.PIECE_BAND_DIR, C.ASSETS_DIR).replace(os.sep, "/")
    texdir = os.path.relpath(os.path.dirname(C.BOARD_TEXTURE_PNG), scene_dir)
    t = cfg.board_thickness
    bands = ensure_piece_bands() if cfg.piece_collision == "bands" else None

    root = ET.Element("mujoco", model="so101_chess")
    ET.SubElement(root, "compiler", angle="radian", meshdir=meshdir, texturedir=texdir, autolimits="true")
    # record the config this file was generated from (ChessScene refuses a mismatching config)
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name=C.CONFIG_TEXT_NAME, data=json.dumps(cfg.scene_key(), sort_keys=True))
    ET.SubElement(custom, "numeric", name="board_centre", data=_f(cfg.board_centre))
    # jaw kinematics measured from the arm meshes (ChessScene checks config.JAW_MODEL / clearance table against it)
    jg = measure_jaw_geometry()
    ET.SubElement(custom, "numeric", name=JAW_MODEL_NUMERIC,
                  data=" ".join(f"{jg[k]:.7g}" for k in ("r_fixed", "r_moving", "theta0", "dz", "q_closed", "q_open")))
    ET.SubElement(custom, "numeric", name=JAW_CLEARANCE_NUMERIC,
                  data=" ".join(f"{q:.7g} {c:.7g}" for q, c in zip(jg["table_q"], jg["table_clearance"])))
    opt = ET.SubElement(root, "option", timestep=_f(cfg.timestep), integrator="implicitfast",
                        cone="pyramidal", noslip_iterations="2")
    ET.SubElement(root, "statistic", extent="1.0", center=_f((cfg.board_gap + 0.1, 0, 0.1)))
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1920", offheight="1440", azimuth="150", elevation="-25")
    ET.SubElement(vis, "headlight", ambient="0.25 0.25 0.25", diffuse="0.30 0.30 0.30", specular="0.1 0.1 0.1")
    ET.SubElement(vis, "map", znear="0.002", zfar="40")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "rgba", haze="0.6 0.65 0.7 1")

    defaults, arm_asset, arm_base, arm_actuator = load_arm(cfg)
    for d in defaults:
        root.append(d)
    mydef = ET.SubElement(root, "default")
    pv = ET.SubElement(mydef, "default", {"class": "piece_vis"})
    ET.SubElement(pv, "geom", group="1", contype="0", conaffinity="0", mass="0")
    pc = ET.SubElement(mydef, "default", {"class": "piece_col"})
    ET.SubElement(pc, "geom", group="3", type="cylinder", friction=f"{C.PIECE_FRICTION} 0.005 0.0001",
                  condim="4", rgba="1 0 0 0.3", solref="0.004 1", solimp="0.95 0.99 0.001")

    asset = ET.SubElement(root, "asset")
    for e in list(arm_asset):
        if e.tag == "mesh" and e.get("file"):
            e.set("file", f"{arm_mesh_prefix}/{e.get('file')}")
        asset.append(e)
    # piece materials (colour from the geom rgba, which the randomiser scales) and meshes
    ET.SubElement(asset, "material", name="piece_white", rgba=_f(C.WHITE_RGBA), specular="0.35", shininess="0.4", reflectance="0.0")
    ET.SubElement(asset, "material", name="piece_black", rgba=_f(C.BLACK_RGBA), specular="0.45", shininess="0.5", reflectance="0.0")
    if cfg.piece_visual == "mesh":
        for ptype in C.PIECE_TYPES:
            ET.SubElement(asset, "mesh", name=f"piece_{ptype}", file=f"{piece_prefix}/{ptype}.obj")
    if bands is not None:
        for ptype in C.PIECE_TYPES:
            for b in bands["pieces"][ptype]["bands"]:
                ET.SubElement(asset, "mesh", name=f"band_{ptype}_{b['index']}", file=f"{band_prefix}/{b['file']}")
    ET.SubElement(asset, "texture", name="board_tex", type="2d", file=os.path.basename(C.BOARD_TEXTURE_PNG))
    ET.SubElement(asset, "material", name="board_mat", texture="board_tex", texuniform="false",
                  texrepeat="1 1", specular="0.2", shininess="0.2", reflectance="0.0")
    ET.SubElement(asset, "texture", name="skybox", type="skybox", builtin="gradient",
                  rgb1="0.75 0.8 0.85", rgb2="0.35 0.4 0.5", width="256", height="1536")

    wb = ET.SubElement(root, "worldbody")
    # lights
    ET.SubElement(wb, "light", name="light_main", pos=_f((cfg.board_gap + 0.1, 0.2, 1.0)), dir="0 -0.2 -1",
                  diffuse="0.50 0.50 0.50", specular="0.15 0.15 0.15", castshadow="true", directional="false")
    ET.SubElement(wb, "light", name="light_fill", pos=_f((cfg.board_gap - 0.3, -0.4, 0.8)), dir="0.4 0.4 -1",
                  diffuse="0.20 0.20 0.20", specular="0.05 0.05 0.05", castshadow="false", directional="false")
    # table: top surface at z = 0 (the arm base sits on it)
    ET.SubElement(wb, "geom", name="table", type="plane", size="1.5 1.5 0.05", pos="0 0 0",
                  friction=f"{C.BOARD_FRICTION} 0.005 0.0001", rgba=_f(C.TABLE_RGB + (1.0,)), group="0",
                  contype=MASK_ARM[0], conaffinity=MASK_ARM[1])

    # board body (jittered as a whole by the randomiser)
    board = ET.SubElement(wb, "body", name="board", pos=_f(cfg.board_centre))
    half = C.BOARD_SIZE / 2 + C.BOARD_BORDER
    ET.SubElement(board, "geom", name="board_box", type="box", size=_f((half, half, t / 2)), pos=_f((0, 0, t / 2)),
                  friction=f"{C.BOARD_FRICTION} 0.005 0.0001", rgba=_f(C.BORDER_RGB + (1.0,)), group="0",
                  condim="4", solref="0.004 1", solimp="0.95 0.99 0.001", contype=MASK_ARM[0], conaffinity=MASK_ARM[1])
    # visual-only textured top (plane geoms render finite; contype 0 so it never collides)
    ET.SubElement(board, "geom", name="board_top", type="plane", size=_f((half, half, 0.01)),
                  pos=_f((0, 0, t + 0.0004)), material="board_mat", contype="0", conaffinity="0", group="0",
                  rgba="1 1 1 1")
    # board corner sites (playing surface corners) for projection tests
    for nm, (sx, sy) in {"corner_a8": (-1, -1), "corner_h8": (-1, 1), "corner_a1": (1, -1), "corner_h1": (1, 1)}.items():
        ET.SubElement(board, "site", name=nm, pos=_f((sx * C.BOARD_SIZE / 2, sy * C.BOARD_SIZE / 2, t)),
                      size="0.002", rgba="0 1 0 0", group="4")

    # capture drop zone pad (robot's right) and spare-queen pad (robot's left)
    dz = ET.SubElement(wb, "body", name="drop_zone", pos="0 0 0")
    dx = cfg.board_gap + C.DROP_X_SHIFT + C.DROP_CELLS * C.SQUARE / 2
    dy = -(C.BOARD_SIZE / 2 + C.DROP_GAP + C.DROP_CELLS * C.SQUARE / 2)
    ET.SubElement(dz, "geom", name="drop_pad", type="box", pos=_f((dx, dy, 0.0005)),
                  size=_f((C.DROP_CELLS * C.SQUARE / 2 + 0.003, C.DROP_CELLS * C.SQUARE / 2 + 0.003, 0.0005)),
                  rgba=_f(C.DROP_PAD_RGBA), contype="0", conaffinity="0", group="0")
    for i in range(C.DROP_CELLS):
        for j in range(C.DROP_CELLS):
            x, y = cfg.drop_cell_xy(i, j)
            shade = 0.9 if (i + j) % 2 else 1.1
            ET.SubElement(dz, "geom", name=f"drop_cell_{i}{j}", type="box", pos=_f((x, y, 0.0011)),
                          size=_f((C.SQUARE / 2 - 0.001, C.SQUARE / 2 - 0.001, 0.0004)),
                          rgba=_f(tuple(min(1, v * shade) for v in C.DROP_PAD_RGBA[:3]) + (1.0,)),
                          contype="0", conaffinity="0", group="0")
    sp = ET.SubElement(wb, "body", name="spare_zone", pos="0 0 0")
    for i in range(2):
        x, y = cfg.spare_cell_xy(i)
        ET.SubElement(sp, "geom", name=f"spare_cell_{i}", type="box", pos=_f((x, y, 0.0005)),
                      size=_f((C.SQUARE / 2, C.SQUARE / 2, 0.0005)), rgba=_f(C.SPARE_PAD_RGBA),
                      contype="0", conaffinity="0", group="0")

    # pieces at the start position + spare queens
    for colour in ("w", "b"):
        for ptype, n in C.PIECE_COUNTS.items():
            for k in range(n):
                sq = START_SQUARES[colour][ptype][k]
                f, r = C.square_index(sq)
                x, y = cfg.square_xy(f, r)
                add_piece_body(wb, f"{colour}_{ptype}_{k + 1}", colour, ptype, (x, y, t), cfg, bands)
        x, y = cfg.spare_cell_xy(0 if colour == "w" else 1)
        add_piece_body(wb, f"{colour}_queen_spare_1", colour, "queen", (x, y, 0.0), cfg, bands)

    # privileged cameras
    bc = cfg.board_centre
    ET.SubElement(wb, "camera", name=C.OVERHEAD_CAM, pos=_f((bc[0], 0, t + cfg.overhead_height)),
                  xyaxes="0 -1 0 1 0 0", fovy=_f(cfg.overhead_fovy))
    spos = (-0.22, 0.42, 0.33)
    ET.SubElement(wb, "camera", name=C.SPECTATOR_CAM, pos=_f(spos),
                  xyaxes=lookat_xyaxes(spos, (bc[0] - 0.05, 0.0, 0.05)), fovy=_f(cfg.spectator_fovy))

    # the arm
    wb.append(arm_base)
    root.append(arm_actuator)
    return root


def embedded_config(model: mujoco.MjModel) -> dict | None:
    """The scene_key() dict recorded in a compiled model (None for an XML without it)."""
    for i in range(model.ntext):
        adr, size = model.text_adr[i], model.text_size[i]
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXT, i) == C.CONFIG_TEXT_NAME:
            raw = bytes(model.text_data[adr:adr + size]).decode().rstrip("\x00")
            return json.loads(raw)
    return None


def write_scene(cfg: C.ChessConfig = C.DEFAULT_CONFIG, path: str | None = None) -> str:
    """Write the scene XML for cfg (default: cfg.scene_xml_path()) and compile it once; returns the path."""
    path = path or cfg.scene_xml_path()
    make_board_texture(C.BOARD_TEXTURE_PNG)
    root = build_xml(cfg, path)
    xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    xml = "\n".join(line for line in xml.splitlines() if line.strip())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(xml + "\n")
    # compile once to fail early and verify the embedded record
    m = mujoco.MjModel.from_xml_path(path)
    assert embedded_config(m) == cfg.scene_key(), "embedded config record does not round-trip"
    return path


def add_config_args(ap: argparse.ArgumentParser) -> None:
    d = C.DEFAULT_CONFIG
    ap.add_argument("--board-gap", type=float, default=d.board_gap, help="x of the board edge nearest the robot (m)")
    ap.add_argument("--board-thickness", type=float, default=d.board_thickness)
    ap.add_argument("--piece-mass-scale", type=float, default=d.piece_mass_scale)
    ap.add_argument("--wrist-fovy", type=float, default=d.wrist_fovy)
    ap.add_argument("--wrist-tilt", type=float, default=d.wrist_tilt_deg)
    ap.add_argument("--wrist-roll", type=float, default=d.wrist_roll_deg, help="wrist camera roll about its optical axis (deg)")
    ap.add_argument("--wrist-cam-forward", type=float, default=d.wrist_cam_forward)
    ap.add_argument("--render", default=f"{d.render_w}x{d.render_h}", help="wrist image size WxH")
    ap.add_argument("--piece-visual", choices=C.PIECE_VISUAL_MODES, default=d.piece_visual, help="piece visuals: mesh skins or primitives")
    ap.add_argument("--piece-collision", choices=C.PIECE_COLLISION_MODES, default=d.piece_collision,
                    help="piece collision model: convex hull bands of the mesh or base + stem cylinders")
    ap.add_argument("--randomisation", choices=C.RANDOMISATION_PRESETS, default="wide",
                    help="domain-randomisation preset used by --randomise / randomised resets ('wide' default, 'narrow' = first build)")


def config_from_args(a: argparse.Namespace) -> C.ChessConfig:
    w, h = (int(v) for v in a.render.lower().split("x"))
    return C.ChessConfig(board_gap=a.board_gap, board_thickness=a.board_thickness, piece_mass_scale=a.piece_mass_scale,
                         wrist_fovy=a.wrist_fovy, wrist_tilt_deg=a.wrist_tilt, wrist_roll_deg=a.wrist_roll, wrist_cam_forward=a.wrist_cam_forward,
                         render_w=w, render_h=h, piece_visual=a.piece_visual, piece_collision=a.piece_collision,
                         randomisation=C.Randomisation.preset(getattr(a, "randomisation", "wide")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--out", default=None, help="output XML (default: cfg.scene_xml_path())")
    ap.add_argument("--print-jaw", action="store_true",
                    help="measure the jaw kinematics from the arm meshes and print the config.py constants")
    ap.add_argument("--print-grasp", action="store_true",
                    help="derive the 'bands' grasp table from the piece manifest + finger boxes and print config.GRASP_TABLE_BANDS")
    ap.add_argument("--bands", action="store_true", help="(re)build the collision band hulls in assets/pieces/bands and exit")
    ap.add_argument("--print-envelope", action="store_true",
                    help="measure the per-slice axis radii of the piece meshes (vertices + exact cuts) and print config.PIECE_AXIS_RADIUS_MM")
    a = ap.parse_args(argv)
    if a.print_envelope:
        print("PIECE_AXIS_RADIUS_MM = {")
        for t in C.PIECE_TYPES:
            r = measure_axis_radii(t)
            print(f'    "{t}": ({", ".join(f"{v:.2f}" for v in r)}),')
        print("}")
        return
    if a.bands:
        rec = build_piece_bands(verbose=True)
        print(f"wrote {C.PIECE_BANDS_JSON}: {sum(p['n_bands'] for p in rec['pieces'].values())} band hulls")
        return
    if a.print_grasp:
        tab = grasp_table(verbose=True)
        print("GRASP_TABLE_BANDS = {")
        for t, r in tab.items():
            extra = ""
            if "extent_mm" in r:
                extra = ",\n            extent_mm=[" + ", ".join(f"({a:.2f}, {b:.2f})" for a, b in r["extent_mm"]) + "]"
            print(f'    "{t}": dict(height={r["height"]:.5f}, width={r["width"]:.5f}, envelope={r["envelope"]:.5f}, '
                  f'zone_lo={r["zone_lo"]:.5f}, zone_hi={r["zone_hi"]:.5f}, axis_deg={r["axis_deg"]}{extra}),')
        print("}")
        return
    if a.print_jaw:
        g = measure_jaw_geometry(cache=False)
        print(f"JAW_MODEL = dict(r_fixed={g['r_fixed']:.7g}, r_moving={g['r_moving']:.7g}, theta0={g['theta0']:.7g}, "
              f"dz={g['dz']:.7g},\n                 q_closed=GRIPPER_Q_CLOSED, q_open=GRIPPER_Q_OPEN)")
        print(f"JAW_CLEARANCE_Q = tuple(np.linspace(GRIPPER_Q_CLOSED, GRIPPER_Q_OPEN, {len(g['table_q'])}).tolist())")
        print("JAW_CLEARANCE = (" + ", ".join(f"{c:.6g}" for c in g["table_clearance"]) + ")")
        print(f"# fingertip gap at the joint limits: {g['gap_closed']*1000:.2f} .. {g['gap_open']*1000:.1f} mm")
        print("# mesh (lip) clearance for reference: " + ", ".join(f"{c*1000:.2f}" for c in g["table_clearance_mesh"]) + " mm")
        for f in g["fingers"]:
            print(f"# {f['name']}: size {np.round(np.array(f['size'])*1000, 2).tolist()} mm")
        return
    cfg = config_from_args(a)
    p = write_scene(cfg, a.out)
    m = mujoco.MjModel.from_xml_path(p)
    print(f"wrote {p}: nbody={m.nbody} ngeom={m.ngeom} ncam={m.ncam} nq={m.nq} config={cfg.scene_key()}")


if __name__ == "__main__":
    main()
