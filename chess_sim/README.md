# chess_sim: MuJoCo digital twin of the SO-101 chess setup

A simulation environment to train (and script) a policy that executes **one chess move** with the SO-101 arm:
pick a piece from square A, place it on square B (or a captured piece in the drop zone), return to a watching
pose, with a Stockfish game loop around it.  Everything here was verified by running code (see *Test results*).

```
chess_sim/
  config.py        constants + ChessConfig (dimensions, layout, cameras, speeds, randomisation)
  build_scene.py   generates scene/chess_scene.xml (board texture PNG, pieces, cameras) from config
  scene.py         ChessScene: load model, square <-> xyz, FEN -> pieces, reset (+randomisation), render, watch pose
  kinematics.py    FK / damped-least-squares IK for the gripper (vertical-jaw constraint), reach profile
  expert.py        scripted pick-and-place planner + executor (capture, castling, en passant, promotion)
  chess_env.py     gymnasium env  gym_so101_chess/SO101ChessMove-v0  (LeRobot-style obs / action, move one-hot)
  record_demos.py  expert demos -> LeRobotDataset (video), progress bar
  run_policy.py    runs a trained LeRobot checkpoint in the env (PolicyExecutor: obs adapter + pre/post processors)
  board_reader.py  'arm looks at the board': BoardReader interface, privileged reader (sim ground truth behind a wrist
                   render), vision reader stub, infer_move() (opponent move from the observed position)
  play_game.py     Stockfish game loop: viewer window (mjpython), --no-viewer (terminal) or --headless; the arm's
                   moves by the scripted expert or a policy checkpoint (--policy-path); board reader; progress bar
  view_scene.py    opens the scene in the MuJoCo viewer (mjpython); --snapshot / --previews headless renders
  feasibility.py   reach (IK, every square x type) and wrist-view analysis at extreme board poses (process pool)
  expert_stats.py  expert success statistics under a randomisation preset (replies to 1.e4 / random moves, pool)
  progress.py / progress_view.command   one-line progress bar file + self-refreshing Terminal viewer
  assets/          so101/ (arm model copy, camera-mount variant), board_texture.png, watch_pose.json (+ _scan.json: the
                   7-view board scan), jaw_geometry.json,
                   pieces/ (Staunton meshes {type}.obj/.stl + convex hulls, manifest.json, LICENSE.txt; bands/ = the
                   generated collision band hulls + bands.json), preview/*.png
  scene/           generated chess_scene.xml
  tests/           test_scene.py, test_motion.py, test_env.py, run_all.py (plain asserts, no pytest needed)
```

Interpreter: `python` from the project's environment (MuJoCo 3.14, gymnasium, python-chess, lerobot).
Viewers need `mjpython` (macOS `launch_passive` requirement).
Stockfish: `stockfish` on PATH (`brew install stockfish`), or `stockfish_path` in config.yaml.  All commands run from
the repository root (`chess_sim` is a package).

## Layout and conventions

World frame: **x forward** from the robot toward the human, **y = robot's LEFT**, **z up**; the arm base centre is
at the origin and the table top is z = 0.  The robot plays **Black** and sits at rank 8 (nearest rank); file a is on
its RIGHT (negative y), file h on its left.  The board is 224 mm (8 x 28 mm squares), its near edge at
x = `board_gap` (0.100 m, tuned from 0.090 for reachability), playing surface at z = `board_thickness` (0.010 m).

```
                 human (White)                       y  (robot's left)
     x = 0.324   +---------------------------+       ^
                 | a1 b1 c1 d1 e1 f1 g1 h1   | rank 1 |
                 | a2 ...                    |        |
                 |          board            |        +-----> x (forward)
                 | ...                       |
     x = 0.100   | a8 b8 c8 d8 e8 f8 g8 h8   | rank 8     a8 at y = -0.098, h8 at y = +0.098
                 +---------------------------+
   drop zone 4x4 |D|D|D|D|   (y < -0.158, x 0.090..0.202)  spare queens (y > +0.158): w (0,0) / b (1,0) cells
                 |D|D|D|D|   captured pieces,
                 ...         cell (0,0) first
                        [ SO-101 base at (0,0) ]      wrist_cam on the gripper, overhead_cam, spectator
                 parking area behind the robot (x < -0.3): pieces not in play (never in any camera)
```

Square centre of (file f = 0..7 for a..h, rank r = 1..8): `x = board_gap + (8 - r + 0.5) * 0.028`,
`y = (f - 3.5) * 0.028`, then the board pose (offset + yaw about the board centre, see *Domain randomisation*) is
applied: `ChessScene.square_xyz / nearest_square / board_to_world / world_to_board`.  Drop cell (i, j):
`x = board_gap - 0.010 + (i + 0.5) * 0.028`, `y = -(0.112 + 0.046 + (j + 0.5) * 0.028)`; spare cell i:
`x = board_gap + (i + 0.5) * 0.028`, `y = +(0.112 + 0.046 + 0.014)` (`DROP_GAP` 46 mm, `DROP_X_SHIFT` -10 mm: the pads
were moved out of the way of the randomised board, see *Domain randomisation*; first build: 10 mm gap, no shift).
Colours: a1 dark, dark RGB (0.42, 0.27, 0.17), light (0.87, 0.78, 0.62); file/rank letters on the border.

Joint convention (LeRobot): a 6-vector `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]` in
**degrees** (0 = calibration mid pose = MJCF zero) + **gripper %** (0 = 7 mm fingertip gap, 100 = 80 mm,
`GRIPPER_PERCENT_MODE = "measured_gap"`).  Helpers: `config.lerobot_to_qpos / qpos_to_lerobot`,
`deg2rad / rad2deg`, `gripper_percent_to_frac / gripper_frac_to_percent`, `gripper_gap(q)`, `gripper_ctrl_for_gap`.

## Dimensions

| piece  | height | base dia | mass | 'bands' grasp: tips / stall width | 'cylinders' grasp: height / stem |
|--------|-------:|---------:|-----:|----------------------------------:|---------------------------------:|
| pawn   | 25 mm  | 15 mm    | 6 g  | 16.5 mm (66 %) / 8.7 mm           | 15.5 mm (62 %) / 9 mm            |
| rook   | 28 mm  | 20 mm    | 10 g | 17.9 mm (64 %) / 15.7 mm          | 17.4 mm / 12 mm                  |
| knight | 31 mm  | 17 mm    | 10 g | 16.6 mm (54 %) / 10.1 mm (across the head) | 19.2 mm / 10.2 mm       |
| bishop | 36 mm  | 20 mm    | 10 g | 21.7 mm (60 %) / 12.9 mm          | 22.3 mm / 12 mm                  |
| queen  | 40 mm  | 20 mm    | 12 g | 23.1 mm (58 %) / 13.1 mm          | 24.8 mm / 12 mm                  |
| king   | 50 mm  | 20 mm    | 15 g | 29.1 mm (58 %) / 11.4 mm          | 31 mm (expert: 25 mm) / 12 mm    |

Height and base (widest foot) diameter are the user's dimensions; the meshes are scaled to them exactly (see *Piece
models*).  Masses are the spec's (set explicitly on each body; `piece_mass_scale`).  Two collision models exist
(`ChessConfig.piece_collision`): **'cylinders'** (default) = base cylinder (base dia, 30 % of the height) + stem
cylinder (60 % of the base dia, 30..100 % of the height); **'bands'** = one convex hull per horizontal band of the
mesh, following the real profile (complete and tested, not the default: it measured worse in the expert's hands,
see *Piece models*).  Friction 1.0, board/table 0.6.  Visuals (`piece_visual`): the mesh skins
(default) or the old primitive assemblies, both in geom group 1 with contype 0.  Physics 2 ms (500 Hz); control
30 Hz (`CONTROL_HZ`, 16/17 substeps so the ticks stay on the exact 1/30 s grid).  Jaw gap: 7 mm closed .. 80 mm
open at 0..100 % (the MJCF joint range opens to 141 mm); approach opening 45 mm; close command = stall width - 2 mm
(the 'bands' stall width is the widest section inside the fingertip pads, `config.stem_diameter`; 0.6 x base dia
for 'cylinders').

## Piece models

* **Source**: "Chess Set" by Riley Queen, Poly Haven (https://polyhaven.com/a/chess_set), **CC0 1.0** (public
  domain; `assets/pieces/LICENSE.txt`).  Geometry only (the glTF textures were not used); one mesh per type
  (`assets/pieces/{type}.obj` + `.stl`, `{type}_hull.obj` = its convex hull), single watertight component,
  1640 (pawn) .. 4172 (king) triangles.  `chess_sim_assets_staging/build_pieces.py` regenerates them from the
  glTF; nothing imports from the staging directory at runtime.
* **Scaling**: Y-up -> Z-up, base centre at the origin, z_min = 0, then scaled NON-uniformly: z so the height
  equals the spec, xy so the widest foot diameter equals the spec base diameter (the two factors differ by up to
  32 %: bishop 0.618 xy / 0.420 z).  The knight's muzzle faces +x; its head is 19.7 mm long (wider than its 17 mm
  foot, accepted) but 10 mm across.  `assets/pieces/manifest.json` records the transforms, a 10-slice diameter
  profile, the 1 %-step stem profile over 45..75 % of the height and the narrowest stem of each piece.
* **Visuals** (`piece_visual='mesh'`, default): each of the 34 bodies carries its type's mesh as one visual geom
  (`<name>_vis_mesh`, group 1, contype = conaffinity = 0) with the `piece_white` / `piece_black` material (the
  randomiser still scales the geom rgba).  `'primitive'` keeps the cylinder/sphere/box assemblies.  34 mesh geoms
  in the compiled default model; the overhead render differs from the primitive scene on 2.2 % of the pixels by
  more than 40 levels (`test_scene a3`).  Previews: `assets/preview/` (wrist_cam at the watch pose, overhead,
  spectator, wrist_cam_corners, wrist_cam_randomised), regenerated with the meshes.
* **Collision 'bands'** (`ChessConfig(piece_collision='bands')`, its own `scene/chess_scene_<hash>.xml`; NOT the
  default, see *Measured difference* below): `build_scene.build_piece_bands` cuts every mesh into horizontal bands (cuts at
  every local minimum of the 0.1 mm width profile, then the band with the largest hull-vs-profile error is split
  at its worst height until all bands are within `BAND_TOL` 0.8 mm, bands >= 0.5 mm tall, at most 14 per piece)
  and writes each band's vertices + exact cut sections as its own OBJ (`assets/pieces/bands/{type}_band{i}.obj`,
  one `<mesh>` asset each; MuJoCo compiles the convex hull): pawn 6, rook 6, knight 6, bishop 10, queen 14,
  king 12 bands (54 hulls, 264 collision geoms over the 34 bodies; `python -m chess_sim.build_scene --bands`
  rebuilds them, then the scene XML).  Accuracy (`test_scene a2`, against the mesh's
  exact planar cuts): at the 60 manifest slice centres the hull is at most 0.97 mm wider than the real cut; every
  0.5 mm it is at most 0.66 mm wider (0.78 mm at 0.1 mm steps) than the real width or, at an undercut, than the
  widest section 0.5 mm above (a convex hull can never be narrower than what is above it: the knight's muzzle
  underside (up to 5 mm), the queen's crown points, the king's cross arms and the bishop's mitre are the only
  places where the hull exceeds the exact cut by more than 1 mm); it never undercuts the mesh.  NOTE the
  manifest's 10-slice widths are pairwise VERTEX distances: 8 of the 60 slices under-report the exact cuts of a
  sparse cone by more than 0.5 mm, up to 2.1 mm (rook 8.4..11.2 mm: 13.4 vs 15.6; king 10..15 mm: 9.9 vs 12.1;
  queen 16..20 mm: 8.0 vs 9.9), and the knight's values pair vertices at different heights (mane / muzzle), so
  the analytic envelope uses `PIECE_AXIS_RADIUS_MM` (vertices + 0.1 mm cuts, `--print-envelope`) instead.  Contact parameters as before (solref 0.004, solimp 0.95/0.99, condim 4, friction
  1.0); the soft contacts let the finger pads sink ~0.75 mm per side into a hull band at the squeeze force (the
  two-cylinder model: < 0.1 mm), which the grasp rule accounts for.  `'cylinders'` keeps the two-cylinder model.
* **Analytic envelope** (planning): `config.piece_radius_at / piece_height_within / piece_max_radius` give the
  collision radius as a function of height from the 10 slices (+ the knight's head extent through the axis
  radii); the expert's obstacle floor, carried-piece offsets, reach-ceiling checks and opening candidates use it
  for 'bands', the two cylinders for 'cylinders'.  The exact finger-box / piece clearances at every knot
  (`jaw_piece_clearance`, `carried_piece_clearance`) are measured with `mj_geomDistance` on the hull bands
  themselves.
* **Grasp table** (`config.GRASP_TABLE_BANDS`, derived by `build_scene.grasp_table` from the mesh width profile and
  the measured finger-box levels, `python -m chess_sim.build_scene --print-grasp`; `test_scene a2` checks the
  table against a fresh derivation).  The fingertips (= the tool point) are put at `height` above the piece base;
  the pads (finger box 0, tip .. tip + 10 mm) stall on `width`; the open fixed finger descends `envelope`/2 +
  2.5 mm from the axis (the widest section above the tips).  Rule: the pads must touch the piece over >= 4 mm of
  height (a cylinder, or two features that lock each other; a single ring / ball equator / rim between flat pads
  swings 6..12 deg while carried, measured), no flare at the pad bottom edge, finger boxes 1 and 2 (tip + 10..20 /
  20..40 mm, 2..3.5 / 6..11 mm wider than the pads at the stall) must clear the collar / head above by 1 mm, tips
  between 30 % and 70 % of the height and below the IK reach cap (king 29.4 mm); the narrowest admissible stall
  width wins and its zone is centred.  The manifest's narrowest stems (pawn 4.2 mm @ 17.2 mm, rook 10.2 @ 16.0,
  knight 12.6 @ 16.4 (neck), bishop 7.0 @ 16.2, queen 7.2 @ 23.2, king 6.3 @ 27.0) are therefore NOT where the
  pieces are held: the 10 mm pads need 10 mm of vertical room, which the thin stems do not have under their
  collars / heads.  Chosen heights (mm; physics probe `test_motion c2`: deviation after the lift and tilt while carried):

  | type   | tips | % height | stall width | descent envelope | zone | what the pads hold |
  |--------|-----:|---------:|------------:|-----------------:|-----:|--------------------|
  | pawn   | 16.5 | 66 | 8.7 mm  | 9.5 mm  | 16.0..17.0 | head ball + the 9.2 mm collar 0.75 mm under the tips as the stop (the 4.6..5.7 mm stem has no 10 mm of room) |
  | rook   | 17.9 | 64 | 15.7 mm | 16.2 mm | 16.3..19.6 | the 15.7 mm battlement fills the pad top (the 10.2 mm stem is 4 mm tall under it) |
  | knight | 16.6 | 54 | 10.1 mm | 10.4 mm | 16.4..16.9 | across the head (9.8..10.9 mm over 15..27 mm, `axis_deg` 90: the jaws must be within 25 deg of the narrow axis; a knight squeezed further off turns in the jaws) |
  | bishop | 21.7 | 60 | 12.9 mm | 13.1 mm | 21.2..22.2 | mitre belly + the 14 mm collar 0.7 mm under the tips as the stop |
  | queen  | 23.1 | 58 | 13.1 mm | 14.2 mm | 23.1..23.2 | the 12.1 mm ring + the crown cone at the pad top edge lock each other |
  | king   | 29.1 | 58 | 11.4 mm | 11.7 mm | 28.9..29.4 | collar ring + the crown rim at the pad top edge (the collar alone swings 10 deg; 29.4 mm = the a1/h1 reach limit at 8 deg of lean) |

  `config.grasp_height_candidates`: the table height, then the lower end of the zone when it is >= 2 mm lower
  (rook, knight: a lower hold lets a carried piece pass under a low reach ceiling); `kinematics.grasp_tool_site /
  place_tool_site` use the envelope / stall extents (direction-dependent for the knight, `extent_mm`).
* **Switches**: `ChessConfig(piece_visual=..., piece_collision=...)` (`--piece-visual`, `--piece-collision` on
  `build_scene`); both are part of `scene_key`, so each combination gets its own XML; the watch pose is shared
  (the occlusion test only sees the table, board and arm).  `python -m chess_sim.build_scene --bands` rebuilds the
  band hulls (then rebuild the scene XML).
* **Measured difference, 'bands' vs 'cylinders'** (why the default stayed 'cylinders'; same expert, same seeds):
  `test_motion d` (60 random moves, seeds 7/11/42) 54/60 with 5 physical failures (bystander rooks displaced
  5.3 / 7.0 / 10.5 mm by the fingers passing their 15.7 mm battlements, the knight toppling at its own release
  (b1c3, 99 deg) and a knight release knocking a pawn (g3e2)) vs 58/60 with 0 physical failures; `test_motion f`
  (all black moves of three openings) 69/83 vs 79/83; fresh seeds 3/5/9 (random play 0..40 plies, 60 moves):
  51/60 with 4 physical failures (two rank-8 side approaches knocking the neighbour / the king while backing away,
  two rooks displaced 5.9 / 6.2 mm) vs 56/60 with 1 (a knight release knocking a pawn) and 3 clean PlanErrors.
  Causes: the pads sink ~0.75 mm per side into a hull band, the knight's 19.7 mm head and the rook's 15.7 mm top
  stand where the 'cylinders' planner expects a 10..12 mm stem, and the fixed 45 mm opening / straight-up release
  leave no margin between mesh neighbours; the 'bands' grasp zones are narrow (queen 0.1 mm), so a grasp outside
  the zone (the old +3 mm fallback) pinches a single feature: the spare queen grasped 3 mm high swung 28 deg and
  toppled on a1 (`test_env h`), fixed by `config.grasp_dz_candidates` (offsets inside the zone first, +3 mm as a
  last resort).  The Crowded phase's adaptive opening and release are the prerequisites for promoting 'bands'.

Cameras: `wrist_cam` on the gripper (camera-mount variant of the arm, tilted 15 deg toward the jaws), `overhead_cam`
(privileged, above the board centre), `spectator` (3/4 view).  The wrist camera matches the REAL camera (Innomaker
U20CAM, 1920x1080): **vertical fovy 47 deg, 16:9 (640 x 360), rolled -90 deg about its optical axis**
(`wrist_roll_deg`; at the same joint pose the real frame is the first build's render rotated 90 deg counter-clockwise:
white at the top, the fixed jaw on the RIGHT).  Intrinsics of the real camera (OpenCV fisheye, flat iPad target,
13 images, RMS 0.19 px: fx 1238.8, fy 1239.1, cx 956.2, cy 538.2, k1..k4 -0.1257 / 0.0075 / -0.0051 / -0.0014) are
stored in `config.WRIST_CAMERA_INTRINSICS` (source `calibration/output/wrist_camera_intrinsics_FINAL.json`; raw fisheye FOV 88.8 x 49.9 deg): the sim
renders the undistorted pinhole view (f = 1239 px -> 47.0 x 75.5 deg), so real frames must be undistorted with
that model before being compared with / fed to a policy.  The real jaw covers the right ~37.5 % of the frame:
`WRIST_IMAGE_WINDOW` = the left 62.5 % is the usable window for every visibility requirement.  With 47 deg no single
pose sees the whole board (the user confirmed this on the real arm), so:
* **Watch pose** (the arm's rest pose between moves, LeRobot deg/%) `[24.9, -11.6, -10.9, 92.0, 40.4, 0]`: sees the far
  a-d quarter (ranks 1-4, files a-d) inside the window with >= 5.0 % margin, unoccluded, for every board pose of the
  'wide' range (12.4 % at the nominal board), 19.4 % of the image.
* **Board scan** (`ChessScene.scan_poses()`, `board_reader.render_scan()`; `SCAN_REGIONS`): 7 views = the watch pose,
  the far e-h quarter, four near views of 2 files x ranks 5-8 each, and the drop zone; every view sees its tile
  inside the window with >= 2 % margin (`SCAN_MARGIN`), unoccluded (rays at every pose), at all 9 extreme / nominal
  board poses (drop zone: 5 pad-jitter poses), so the union is all 64 squares + the drop zone for every randomised
  board (`test_scene n`, also checked square by square on 60 random poses).  `assets/preview/wrist_scan_views.png`.

## How to run

```bash
cd so101-chess
PY=python
MJ=mjpython

# scene: regenerate the XML (only needed after editing config.py), print the config / open decisions
$PY -m chess_sim.build_scene            # writes chess_sim/scene/chess_scene.xml + assets/board_texture.png
$PY -m chess_sim.config                 # gripper mapping, control rate, OPEN_DECISIONS
$PY -m chess_sim.scene --recompute-watch   # re-run the watch-pose search (progress bar in data/progress.txt)
$PY -c "from chess_sim.scene import ChessScene; ChessScene().scan_poses(recompute=True)"   # the 7-view scan (~25 min)
$PY -m chess_sim.feasibility                 # reach at the 11 extreme board poses + drop zone (6 processes, ~6 min)
$PY -m chess_sim.feasibility --x=-15,-10,0,5,10 --no-view   # scan the x offset (mm) at the y / yaw extremes
$PY -m chess_sim.expert_stats --preset wide  # expert success, 120 randomised moves (6 processes, ~10 min); --preset narrow
                                             # --set piece_jitter_m=0.003 overrides a Randomisation field

# look at it (native MuJoCo window, physics in real time)
$MJ chess_sim/view_scene.py                         # start position, arm at the watch pose
$MJ chess_sim/view_scene.py --fen "<FEN>" --zero-pose --randomise --seed 3
$MJ chess_sim/view_scene.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" --move e7e5 --loop
$PY chess_sim/view_scene.py --snapshot out.png --camera overhead_cam    # headless render instead of a window
$PY chess_sim/view_scene.py --previews chess_sim/assets/preview           # regenerate the preview PNGs (headless)

# one expert move in the physics (prints plan, success, metrics)
$PY chess_sim/expert.py e2e4

# play against Stockfish (you are White, type moves in the terminal; the arm plays Black in the window)
$MJ chess_sim/play_game.py --skill 5 --time 0.1
$MJ chess_sim/play_game.py --auto --plies 20           # Stockfish both sides, White teleported, Black by the arm
$MJ chess_sim/play_game.py --auto --arm-white           # the arm executes both colours
$PY chess_sim/play_game.py --headless --plies 6         # no window; per-move report, exit 0 iff all arm moves OK
$PY chess_sim/play_game.py --no-viewer                  # the interactive terminal game without a window (plain
                                                        # python; moves can be piped on stdin, EOF = quit)
                                                        # every mode writes the progress bar (ply, move, expert phase)
                                                        # to chess_sim/data/progress.txt (--progress-path) about
                                                        # once a second; the final line reads <plies played>/<plies played>
$PY chess_sim/play_game.py --headless --plies 6 --policy-path outputs/train/so101_chess/checkpoints/last/pretrained_model \
    [--policy-device mps] [--policy-max-steps 900]     # a trained policy executes Black's moves instead of the expert
                                                        # (--reader privileged is the board reader; 'vision' is the TODO stub)

# gymnasium env + expert demos -> LeRobotDataset
$PY chess_sim/chess_env.py 0                            # one expert episode through env.step (seed 0)
$PY chess_sim/record_demos.py --episodes 20 --out data/demos_20 --repo-id user/so101_chess_sim [--randomise] \
    [--progress-path chess_sim/data/record_progress.txt]
                                                        # --out is relative to the cwd: run from the project root
                                                        # it writes "data/demos_20"
                                                        # (use --out chess_sim/data/demos_20 to keep it in chess_sim);
                                                        # --progress-path keeps the bar apart from a game / test running
                                                        # at the same time (they all default to data/progress.txt)
open chess_sim/progress_view.command                    # live progress bar of any long job (Terminal window)

# train (LeRobot 0.6.2, verified on a recorded dataset: 'End of training', checkpoints/<step> + last written)
lerobot-train --policy.type=act \
    --dataset.repo_id=user/so101_chess_sim --dataset.root="data/demos_20" \
    --output_dir=outputs/train/so101_chess --steps=20000 --batch_size=8 --num_workers=0 \
    --policy.device=mps --policy.push_to_hub=false --wandb.enable=false        # --policy.device=cpu also works
                                                        # (there is no --eval_freq option in lerobot 0.6.2; the env is
                                                        # not a registered lerobot env, evaluate with run_policy.py)
$PY chess_sim/run_policy.py --policy-path outputs/train/so101_chess/checkpoints/last/pretrained_model --episodes 10 \
    [--device mps] [--fen FEN --move e7e5] [--randomise]   # success rate of the checkpoint in SO101ChessMove-v0

# tests
$PY chess_sim/tests/test_scene.py        # ~1 min
$PY chess_sim/tests/test_motion.py       # 14-22 min
$PY chess_sim/tests/test_env.py          # ~2.5 min (140 s measured)
$PY chess_sim/tests/run_all.py           # all three, PASS/FAIL table (17-19 min measured); --only scene,env  --skip motion
```

`play_game.py` interactive commands: a move as UCI (`e2e4`) or SAN (`Nf3`), `hint`, `auto` (Stockfish plays your
white move), `fen`, `quit` (also EOF on stdin).  The same loop runs without a window under plain python with
`--no-viewer` (that is how `test_env.py` exercises it, piping moves on stdin).  An explicit `--stockfish-path` that
does not exist is an error (no silent fallback to the default binary).  White's moves are applied by teleporting the piece (the human moves it on the real
board); captures by the arm go to the next free drop cell; promotion = pawn to the drop zone, then the spare
queen of that colour to the promotion square; castling moves the king then the rook (rook first when needed);
en passant removes the pawn behind the target square.  If the best engine move cannot be planned the next
`--multipv` candidates are tried; if none can, or the execution fails, the failure is reported, the pieces are
re-synchronised to the logical position (teleport; captured pieces then leave the drop zone for the parking
area) and the game goes on.  A teleported promotion whose spare queen is already on the board uses the regular
queen body of that colour when it is off the board (captured); a position with three queens of a colour cannot
be represented and is reported as such.  The progress bar (`chess_sim/data/progress.txt`, `--progress-path`)
counts plies (total = `--plies`, else 80 for an open-ended game) and shows the current move and expert phase;
`open chess_sim/progress_view.command` views it.

## The gymnasium environment

```python
import gymnasium as gym, chess_sim.chess_env          # registers the id
env = gym.make("gym_so101_chess/SO101ChessMove-v0")   # or SO101ChessMoveEnv(randomise=..., dense_reward=..., max_steps=1500)
obs, info = env.reset(seed=0, options={"fen": "...", "move": "e7e5"})   # options optional: random mid-game position + random move
obs["pixels"]["wrist"]  # (480, 640, 3) uint8      obs["agent_pos"]  # float32[6] deg + %      obs["task"]  # 'move the piece from e7 to e5'
obs["environment_state"]  # float32[128] one-hot: from-square [0:64] + to-square [64:128]  (encode_move / decode_move)
obs, r, terminated, truncated, info = env.step(action)   # action float32[6] targets (deg, %), <= 3 deg per joint per step
a = env.unwrapped.expert_action(obs)                     # scripted expert action for a demo
```

* **How the policy is told which move to make**: `obs["environment_state"]` is the one-hot move (a1 = index 0 ..
  h8 = 63 for the from-square, +64 for the to-square; a promotion is implied, castling is the king's move).
  LeRobot's `preprocess_observation` maps it to `observation.environment_state`, which ACT (`env_state_feature`)
  and diffusion consume, and `record_demos.py` stores the same vector, so an ACT trained on the demos is
  move-conditioned (verified: a `lerobot-train --policy.type=act` checkpoint lists `observation.environment_state
  [128]` in its `input_features`).  The text `task` is only used by language-conditioned policies (smolvla, pi0).
* reset without options: random legal play from the start position for 1..29 plies (Black to move), then a random
  legal move the expert can plan (`plan_tries` candidates).  The pieces captured during that random game stand in
  the drop zone in capture order (`replay_captures`; a 17th capture and beyond is parked), as after a real game; a
  position given as `fen` has no history, so its captured pieces are parked behind the robot.
  `options={"randomise": True}` applies the domain randomisation (`cfg.randomisation`, default preset 'wide': board
  pose x -10..+5 mm / y +-20 mm / yaw +-5 deg, pieces +-5 mm + yaw, pads +-5 mm, light / headlight / colours, wrist
  camera +-3 deg / 5 mm; see *Domain randomisation*).
  `options={"fen": ..., "move": ..., "keep_scene": True}` starts an episode on the scene as it is (no scene reset;
  the physical board must match the fen): this is how `play_game.py --policy-path` runs a policy move by move.
* reward 1.0 on success (sparse); `dense_reward=True` adds `-0.05 * (fingertip-to-stem + piece-to-target)` [m].
* terminated: success, or `info["failure"]` in `toppled:<piece>` (> 20 deg; a piece swinging in the jaws is
  exempt), `knocked:<piece>` (> 15 mm), `dropped:<piece>` (carried, then resting away from source and target),
  `off_table:<piece>`.  truncated at `max_steps` (1500 = 50 s).
* success: every moved piece within 8 mm of its target, tilt < 10 deg, resting, no other piece displaced > 5 mm,
  the gripper retreated to the lift height (both fingertips >= the episode's retreat height above the board with no
  arm-piece contact, latched once reached after the pieces are in place; `info["retreat_height"]`,
  `info["retreated"]`) and (`require_watch`) the arm back at the watch pose (<= 3 deg).  The retreat height is
  the spec's 90 mm (`LIFT_CLEARANCE`) wherever the arm reaches it above the last placed piece (ranks 1-6, the
  drop zone), else that square's reach ceiling - 5 mm (rank 7 ~49 mm, c8-f8 ~25 mm; the expert's own lowered
  lift height, `kinematics.SPEC_DEVIATIONS`); `SO101ChessMoveEnv(retreat_height=0.06)` fixes it instead.
  `info["phase"]`: reach / grasp / carry / place / return / done / failed.
* guards: `step()` raises `ValueError` on a NaN/inf action (the actuators keep their last finite target) and
  `RuntimeError` when called before `reset()`; `reset(options={"move": ...})` without `"fen"` raises `ValueError`
  (the start position has White to move).
* info also carries the privileged state: `board_pose` {dx, dy, yaw_deg}, `board_corners`, `drop_zone_offset`,
  `spare_zone_offset`, `fen` (before the move), `move`, `targets` {piece: xyz}, `piece_err_mm`,
  `pieces` {name: [x y z qw qx qy qz]} for all 34 bodies, `board` (square -> symbol read from the physics),
  `steps`, `sim_time`, `expert` (current expert waypoint label).

Dataset (`record_demos.py`, LeRobot v3): fps 30, features `observation.images.wrist` (video AV1 640x480, or
`--images` PNG), `observation.state` float32[6], `observation.environment_state` float32[128] (the move one-hot),
`action` float32[6], `task` string; `<out>/episodes.json` lists fen / move / success / steps per episode.  Failed
episodes are discarded and re-drawn (`--keep-failed` keeps them).  `verify()` decodes the move back from the stored
`observation.environment_state` of the first frame.

Running a checkpoint (`run_policy.py`): `PolicyExecutor(path, device)` loads any LeRobot policy type from a
`pretrained_model` directory with its pre/post processors (normalisation, device) and turns an env observation into
the policy batch with LeRobot's own `preprocess_observation` (+ the task string); `act(obs, task)` returns the
LeRobot action, `run_episode(env, ...)` a summary like `run_expert_episode`.  `play_game.py --policy-path` uses it
to execute Black's moves inside the game loop (one env episode per move on the live scene, judged by the env's
success definition; a failed move is re-synchronised by teleport like an expert failure).

The game loop (`play_game.py`) is  watch pose -> **board reader** (`board_reader.py`) -> `infer_move()` recovers
the opponent's move from the observed position -> Stockfish -> arm (expert or policy) -> watch pose.  Only the
privileged reader exists in the digital twin: it renders the wrist camera at the watch pose and returns the
simulator's `board_state_symbols()`; `VisionBoardReader.read()` raises `NotImplementedError` (plan in its
docstring).  Every opponent move is checked through the reader (`SUMMARY ... board reads k/n`).

## Domain randomisation

`ChessScene.reset(randomise=True)` draws from `cfg.randomisation` (`config.Randomisation`).  Two presets:

| | 'narrow' (first build) | 'wide' (default) |
|---|---|---|
| board pose | x, y +-5 mm, no yaw | x -10..+5 mm (toward / away from the arm), y +-20 mm, yaw +-5 deg about the board centre |
| pieces | +-3 mm xy, random yaw | +-5 mm along the board axes, random yaw, re-drawn so neighbours keep >= 1 mm between their mesh envelopes (`piece_min_gap_m`) |
| drop zone / spare pads | fixed | each +-5 mm (they stay on the table, independent of the board) |
| lights | position +-15 cm, diffuse +-30 % | position +-30 cm (re-aimed at the board), diffuse +-40 %, headlight ambient / diffuse +-40 % |
| colours | pieces / board +-10 % | pieces / board / table +-15 %, table tint +-10 % per channel |
| wrist camera mount | 2 deg, 5 mm ball (backward <= 0.5 mm) | 3 deg, 5 mm ball (backward <= 0.5 mm) |

**Board pose end to end.**  The board body carries a yaw quaternion; everything in board coordinates follows:
`square_xyz`, `nearest_square`, `board_state` / `pieces_on_board` (inside-board test in board coordinates),
`set_position` (pieces placed and rotated with the board), `board_corners`, the textured top and corner sites (children
of the body), the watch / scan visibility tests, the expert (it plans from `square_xyz` / `piece_pos`; its
board-aligned opening directions and its A* grid follow the board pose) and the env info (`board_pose`,
`board_corners`).  Decision for the drop zone and the spare-queen cells: they are separate pads on the table, so they
do NOT follow the board; they get their own +-5 mm jitter (`drop_zone_offset`, `spare_zone_offset`).  They were moved
out of reach of the randomised board (`DROP_GAP` 10 -> 46 mm, `DROP_X_SHIFT` -10 mm toward the arm so the far cells
stay reachable: 490/490 drop / spare targets with the pads at their +-5 mm corners; -40 mm, tried first, sent the
expert's routes from the drop zone to rank 8 over the c8 bishop (Qd8xd5 regression)): a piece in a cell clears the
board box by >= 3.0 mm at every extreme pose (`test_scene l`).  The expert's A* grid keeps the first build's lattice
anchor and is only extended by whole cells (to cover the moved pads and the board pose), so routes on the nominal
layout keep their knots (a shifted lattice changed the Kxf7 / O-O routes and toppled the king).  On the
real table: keep the drop zone >= 4.6 cm beside the board's playing surface.

**Reach analysis** (`feasibility.py`; every square x every type at its grasp height, the test_motion a logic; 384
targets per pose).  With the first build's 8 deg lean cap the far corners a1/h1 have ~0 mm of slack at the nominal
pose (radius 0.325 m reachable, 0.327 m not), so any sideways shift or yaw loses them:

| board pose (dx mm, dy mm, yaw deg) | 8 deg cap | + 12.5 deg far tier |
|---|---|---|
| (0, 0, 0) | 384/384 | 384/384 |
| (-20 / -15, any, any) | 372-378 (d8 / e8 / f8 / c8: too close to the pan axis) | not used |
| (-10, +-20, +-5) | 381-384 (a1 or h1 queen/king; f8/c8 king) | 384/384 at all 4 |
| (-10, 0, 0) | 384/384 | 384/384 |
| (-5, +-20, +-5) | 379-384 (a1 or h1) | 384/384 at all 4 |
| (0, +-20, +-5) | 378-382 (a1 or h1) | 384/384 at all 4 |
| (+2.5 / +5, +-20, +-5) | 375-376 at +5 | 384/384 at all 8 |
| (+7.5, +-20, +-5) | - | 381/384 (a1 or h1) |
| (+10, +-20, +-5) | - | 378/384 (a1 or h1) |
| drop zone + spare cells, pads at the +-5 mm corners | 490/490 | 490/490 |

Hence `expert.GRASP_MAX_LEAN_FAR` = 12.5 deg: a separate, later tier of `grasp_pose` used only when no pose within
8 deg exists (wherever 8 deg suffices the expert's choice is unchanged; a lean of 12.5 deg reaches ~12 mm further),
and the asymmetric x range -10..+5 mm.  **Real-world tolerance** to tell the user: the board's near edge must be
100 mm from the arm base axis (`board_gap`) within **-1.0 cm (toward the arm) .. +0.5 cm (away)**, its centre line
within **+-2 cm sideways**, and its edges parallel to the arm's forward axis within **+-5 deg**; beyond -1.5 cm the
d8 / e8 squares get too close to the pan axis, beyond +0.75 cm (with a sideways shift / yaw) a1 or h1 are lost.
`test_motion k` asserts 100 % reach at the 11 poses (nominal, the 8 corners of the box, and x_min / x_max with y = yaw
= 0, where the near squares are closest).

**Wrist view** (`watch_metrics` / `scan_metrics`, worst case over the nominal + 8 extreme board poses, rays at every
pose, targets inside `WRIST_IMAGE_WINDOW`).  With the first build's camera (70 deg, 4:3, unrolled) the old watch pose
dropped from 5.0 % margin to -8.2 % (x +-10 mm) / -15.9 % (+-20 mm) with jaw occlusion, and a re-searched single
pose reached at best -2.2 %.  With the real camera (47 deg, 16:9, roll -90, jaw window) the best single pose for the
whole board is far outside (-14 % for ranks 1-5 across all files, -38 % for ranks 7-8 across all files), so option (b) was
implemented: the 7-view scan above.  Measured per tile (best margin found, 'wide' range): ranks 1-4 x files a-d / e-h
5.0 % / 2.0 % (feasible at 5 %); ranks 5-8 x 3 files 0.9 % / 3.6 % / -0.8 % (infeasible), x 4 files -6 %, x 2 files
2.0 % each (feasible at 2 %); drop zone 2.0 % (5 % reachable in a separate search).  Watch pose: 5.0 % worst / 12.4 %
nominal, 19.4 % of the image.  `keep_board_visible` now only re-draws the CAMERA jitter, and only when the watch
region of the drawn board is visible with the nominal camera (0 re-draws in 60 'wide' resets, `test_scene o`).

**Expert statistics** (`expert_stats.py`, randomise=True on every move; 'random' = 30 positions x 2 moves, each move
re-randomised; first build: test_motion g 60/60, test_motion d 58/60 without randomisation):

| sample | 'narrow' | 'wide' | 'wide' with +-3 mm pieces |
|---|---|---|---|
| Black's 20 replies to 1.e4 x 3 seeds | 60/60 | 54/60 | 59/60 |
| random positions, 60 moves | 56/60 | 56/60 | - |

The 'wide' replies lose 6 moves to the +-5 mm piece jitter in the crowded start ranks, not to the board pose or yaw:
two knocked pawns (the jaws closing on the b8 knight / carrying f7, neighbours down to the 1 mm footprint gap), three
PlanErrors of the g8 knight (lift clearance to the f8 bishop < 2 mm, or no carry over g7) and one e7 pre-grasp
without room; with +-3 mm pieces and every other 'wide' range 59/60.  A larger `piece_min_gap_m` (3 mm) did not help
(56/60).  The random-position failures are the known rank-8 PlanErrors (d7e8, d7d8, d8g5; they fail with 'narrow' too, and in
'wide' the two king moves drew dx = -9.6 mm, where the release above d8 / e8 with a side retreat is also at the edge
of the reach: the reach test covers top-down grasp poses, not the retreat legs) + one castling rook (h1 at dy +17.7
mm, lift clearance 1.6 mm).

## Config parameters (`config.py`)

| name | default | meaning |
|------|---------|---------|
| `ChessConfig.board_gap` | 0.100 m | x of the board edge nearest the robot (0.080..0.110; 0.090 leaves d8/e8 kings ungraspable) |
| `board_thickness` | 0.010 m | playing surface height |
| `piece_mass_scale` | 1.0 | scales all piece masses |
| `piece_visual` / `piece_collision` | 'mesh' / 'cylinders' | piece models (see *Piece models*): mesh skins or primitive assemblies; convex hull bands of the mesh or base + stem cylinders |
| `BAND_TOL`, `BAND_MIN_HEIGHT`, `BAND_MAX_COUNT` | 0.8 mm, 0.5 mm, 14 | band slicing: hull-vs-profile width tolerance, thinnest band, bands per piece |
| `GRASP_TABLE_BANDS`, `PIECE_AXIS_RADIUS_MM`, `ENVELOPE_PAD` | derived | 'bands' grasp heights / stall widths / descent envelopes; per-slice axis radii of the analytic envelope (+0.1 mm) |
| `wrist_fovy` | 47 deg | wrist camera vertical FOV (real camera after undistortion; first build 70 deg) |
| `wrist_roll_deg` | -90 deg | wrist camera roll about its optical axis (real mount; 0 = first build) |
| `WRIST_CAMERA_INTRINSICS` | fisheye fit | real camera K + fisheye k1..k4 (undistort real frames) |
| `WRIST_IMAGE_WINDOW` | (0, 0.625, 0, 1) | usable image part (the real jaw covers the right ~third) |
| `watch_region`, `watch_board_range` | far a-d quarter, 'wide' range | what the watch pose must see, for which board poses |
| `SCAN_REGIONS`, `SCAN_MARGIN`, `scan_board_range` | 7 views, 2 %, 'wide' range | multi-view board scan |
| `DROP_GAP`, `DROP_X_SHIFT` | 46 mm, -10 mm | drop zone / spare cells clear of the randomised board |
| `wrist_tilt_deg` | 15 deg | optical axis tilt toward the jaws (the printed mount is 25.4 deg) |
| `render_w x render_h` | 640 x 360 | wrist image size (16:9 like the real camera) |
| `Randomisation.*` | preset 'wide' | see *Domain randomisation* (`Randomisation.preset('narrow')` = the first build's ranges; `--randomisation narrow` on the CLIs) |
| `expert.GRASP_MAX_LEAN_FAR` | 12.5 deg | far-reach lean tier, only when no pose within 8 deg exists (a1/h1 on a shifted / rotated board) |
| `CONTROL_HZ` / `TIMESTEP` | 30 / 0.002 | control and physics rates |
| `JOINT_SPEED_DEG_S` | 90 | per-joint speed limit (3 deg per control step) |
| `GRASP_HEIGHT_FRAC`, `LIFT_CLEARANCE` | 0.62, 0.090 m | grasp height fraction, lift height above the board |
| `APPROACH_GAP`, `SQUEEZE` | 0.045 m, 0.002 m | jaw opening for the approach; close gap = 0.6 x base - 2 mm |
| `SUCCESS_POS_TOL`, `SUCCESS_TILT_DEG`, `DISTURB_TOL` | 8 mm, 10 deg, 5 mm | success criteria |
| `GRIPPER_GAP_CLOSED/OPEN`, `GRIPPER_PERCENT_MODE` | 7 / 80 mm, measured_gap | percent <-> jaw gap mapping |
| `WATCH_MARGIN`, `WATCH_POSE_PRIOR`, `WATCH_SEARCH` | 5 %, ... | watch-pose requirement, warm start, search budget |
| `chess_env.RETREAT_HEIGHT`, `RETREAT_MIN`, `MAX_PLIES_DEFAULT` | 0.090 m (= `LIFT_CLEARANCE`), 0.020 m, 30 | env success retreat height (rank dependent, see below), floor, random-position depth |
| `expert.LANDING_TILT_MAX`, `ADAPT_MAX` | 4 deg, 15 mm | release candidates / grasps ranked so the rigidly held piece lands within this tilt; largest in-jaw offset corrected at the place (larger ones applied capped) |

A non-default `ChessConfig` gets its own `scene/chess_scene_<hash>.xml` and `assets/watch_pose_<hash>.json`
(regenerated automatically), so tuning never clobbers the default scene.  `python -m chess_sim.config` prints the
`OPEN_DECISIONS` (assumptions awaiting the user: fovy, gripper 0/100 % gaps, mount tilt, image roll).

## Test results (this build)

* `test_scene.py` 13/13 PASS (60 s): dimensions read back from the compiled model (28 mm squares, 224 mm board,
  piece heights/bases, the spec masses on all 34 bodies), (a2) piece models: 6 meshes with the manifest's triangle
  counts (1640..4172) / heights / foot diameters, the analytic envelope contains every vertex and planar cut
  (excess 0.0 um), the 'bands' grasp table equals a fresh derivation, 54 band hulls within 0.97 mm of the real
  width at the 60 slice centres and 0.66 mm every 0.5 mm (never undercutting the mesh); (a3) the model switches
  build their own scenes (primitive/cylinders ngeom 282, mesh/cylinders 164 = the default, mesh/bands 360) and the
  mesh skins render differently from the primitives (2.2 % of the overhead pixels differ by > 40 levels); texture
  orientation (a1 dark), stability of the 34 pieces for 2 s on both collision models (drift 0.53 / 0.58 mm, tilt
  0.002 / 0.054 deg), FEN round trip, watch pose (margin 5.0 %, no occlusion), renders, randomised reset (200
  seeds: camera displacement <= 5.00 mm with <= 0.43 mm backward, rotation <= 1.99 deg, all corners in the image;
  near-black wrist pixels <= 10.2 % over 100 seeds, nominal 8.1 %), config consistency, jaw-gap model (max error
  0.000 mm vs the meshes), a physical squeeze / lift / release of the king (released with the expert's open-then-
  retreat sequence: 0.00 deg after the retreat), 30 Hz grid (300 steps = 10.000000 s).
* `test_motion.py` 11/11 PASS (1004 s): 482/482 IK targets; grasp + 1 s hold of every type from the crowded start
  squares (e7/b8/a8/c8/d8/e8, deviation <= 3.6 mm); (c2) the mesh 'bands' scene: every type grasped by the
  expert's plan at the table heights (knight at four yaws), lifted 41..54 mm and held 2 s: deviation 0.5..2.9 mm,
  tilt 0.9..6.2 deg, measured jaw clearance within 2.5 mm of the table's stall width; 60 random moves 58/60 (both
  failures clean PlanErrors: d7e8 / d8g5 rank-8 diagonal crossings, 0 physical failures; mean 12.8 s sim per
  move, plan 2.0 s); 79/83 black moves from three openings (95 %: the four d8-queen diagonal crossings of 1.e4 e5
  2.Nf3 fail to plan); randomised 1.e4 replies 60/60; 4 promotions + 3 castlings OK; verifier regressions 11/11
  incl. Kg1-h1 (1.8 mm, landing 3.1 deg after the corner-release fix below), Rb1-a1 (1.3 mm), Kxf7 after 5.Bxf7+
  (0.9 mm); max placement error of the regressions 6.8 mm (c8d8).
* `test_env.py` 11/11 PASS (205 s): spaces match incl. the one-hot move (`environment_state` 52->36 = e7e5),
  fen/move resets (random resets 4.8 s incl. planning; captured pieces of the random game in the drop zone: (2, 2),
  (4, 4) ...), expert rollouts through `env.step` 5/5 random moves (placement error 0.6-1.2 mm, <= 2.99 deg per
  action step), a 2-episode dataset loads back (2 episodes, 534 frames, fps 30, image (3, 480, 640), state/action
  (6,), environment_state (128,) decoding to the move), `play_game.py --headless --plies 6` 3/3 arm moves + 3/3
  board reads + its progress bar ending in DONE, termination on a toppled piece / truncation at max_steps, NaN/inf
  action and 'move'-without-'fen' rejection, rank-dependent retreat heights (e5/f6/d6 90 mm, d7 49 mm, c8 24.5 mm)
  with d8d7 and e7e5 rollouts succeeding, a second promotion teleported with the regular queen body, `step()`
  before `reset()` rejected, the interactive terminal loop (`--no-viewer`, moves piped on stdin), a bad
  `--stockfish-path` exits 1, `--plies 0` finishes `DONE`; (j) the privileged board reader + `infer_move` (Nf3,
  a7a8q), the vision stub raising, `keep_scene` resets, a malformed `--fen` exiting 1 in 0.3 s; (k) an untrained
  ACT (11.4 M params, inputs wrist image + state + environment_state) saved as a checkpoint, loaded by
  `PolicyExecutor`, 12 steps in the env and one policy ply inside `play_game.Game` (fails as expected, resynced).
* `run_all.py`: 35 tests (scene 13, motion 11, env 11).  Last full run 35/35 PASS in 20.3 min (scene 60 s, motion
  1004 s, env 157 s) on the final build (mesh visuals, 'cylinders' collision).
* Verification runs of the Meshes phase (not in the suite): the same 60 random moves and openings on the 'bands'
  scene (see *Piece models*, measured difference) and fresh random seeds 3/5/9 (0..40 plies, 60 moves) on both
  collision models: 'cylinders' 56/60 (1 physical failure: a knight release knocking a pawn at b1c3; 3 clean
  PlanErrors in crowded rank-8 / castling positions), 'bands' 51/60 (4 physical failures).
* Verified outside the suite: `lerobot-train --policy.type=act` on a 1-episode dataset (3 steps, cpu): 'End of
  training', checkpoint whose `config.json` lists `observation.environment_state [128]` among the inputs;
  `run_policy.py --policy-path <that checkpoint> --episodes 1 --max-steps 30`: runs (0/1 success, 3 training steps).
* Extra runs: 8/8 random env episodes (seeds 0-7, 257-482 steps each, 2-5 s wall without rendering);
  `play_game.py --headless --plies 24 --skill 1`: 12/12 arm moves incl. a capture (dxe3 to the drop zone) and
  O-O (king slid from e8 with a side approach, rook routed in 3 legs), 156 s of simulation in 56 s wall;
  `--plies 30 --skill 1 --time 0.05 --seed 4`: 15/15 arm moves, 187 s of simulation in 87 s wall, the progress
  bar refreshed with 31 distinct states sampled every 2 s.

## Known limits

* **Reach band**: measured with vertical jaws 8..31 cm from the base (6..34 cm with <= 15 deg lean).  Over ranks
  7-8 (within ~8 cm of the pan axis) the open jaws cannot rise above ~46 mm, so c8..f8 pieces (and the king on
  rank 7) are approached from the side and dragged, not lifted; the lift height there is the reach ceiling
  (36..51 mm) instead of 90 mm; the king is grasped at 50 % of its height (25 mm) instead of 62 %; leans up to
  8 deg at a1/h1 (spec 5).  Rank-8 queen/king moves that must cross diagonally between two occupied squares
  (e.g. d8->e7/f6/g5/h4 with e8/d7 occupied) and castling with d7/e7/f7 all occupied raise `PlanError`
  (`kinematics.SPEC_DEVIATIONS`); `play_game` then tries the next engine candidates.
* **Corner releases / side releases**: a lifted piece is held rigidly, so it lands with the tilt between the grasp
  and release jaw orientations.  a1/h1 leave a single reachable release orientation; the grasp candidates are
  therefore ranked by release compatibility (`Expert.rank_grasps_for_release`, `LANDING_TILT_MAX` 4 deg) and the
  release candidates by predicted landing tilt.  Kg1-h1 (1.5 mm), Rb1-a1 (1.3 mm) and Kxf7 after 5.Bxf7+ (3.3 mm,
  the king side-approached on rank 7/8 with a 10 mm in-jaw offset, corrected at the place: `ADAPT_MAX` 15 mm,
  larger offsets applied capped) are regression cases of `test_motion j`.  Pieces that are side-approached or slid
  (rank 8) still land with up to ~13 deg of predicted tilt because they stand on their base while dragged (c8d8:
  6.8 mm placement error, within the 8 mm tolerance but the largest of the suite).  The release search at a top
  destination stops at the first lean level with a good clearance; for a grasp that itself needed more than the
  5 deg spec lean (the corners) it now searches every lean level (`Expert._pick_place_once`), because the
  landing-compatible release can sit one level further out (Kg1-h1: the +8.0 deg level, landing 3.1 deg instead
  of 10.6 and a toppled king).  A promotion to a1/h1 still lands the spare queen with the corner's 7.6 deg
  (grasped at 0 deg on the spare cell, where no leaning grasp exists): it stands (1.6 mm error).
* **Wrist camera**: 47 deg / 16:9 / roll -90 deg and the fisheye intrinsics come from the real camera; the sim renders
  an ideal pinhole (no distortion, no vignetting, no motion blur): undistort real frames with
  `WRIST_CAMERA_INTRINSICS` first.  The usable window (left 62.5 %) is the real jaw's footprint, approximated; the
  sim's own jaw meshes occupy a similar region.  The real mount tilt is 25.4 deg vs the scene's 15 deg (`wrist_tilt_deg`).
  The real gripper's 0 % / 100 % openings (7 / 80 mm) are still to confirm with calipers.  The wrist-camera jitter is
  a 5 mm ball except backward (<= 0.5 mm, mirrored forward otherwise: the camera module mesh starts 1 mm behind the lens).
* **No single view of the whole board**: the watch pose sees ranks 1-4 x files a-d; reading the board needs the 7-view
  scan (`render_scan`), i.e. ~7 arm motions per read on the real robot; the near views have only 2 % margin (the
  near ranks are 10-22 cm from the pan axis).  A policy trained on the watch-pose image does not see the near ranks
  at the start of an episode.
* **Board placement tolerance** is set by the arm's reach, not by the randomisation: -10..+5 mm in x, +-20 mm, +-5 deg;
  the far corners need up to 12.5 deg of jaw lean (`GRASP_MAX_LEAN_FAR`, spec 5 deg) on a shifted / rotated board.
* **Piece jitter +-5 mm** costs the expert 6/60 of the 1.e4 replies (crowded start ranks, see *Domain
  randomisation*): demos recorded with `--randomise` discard more failed episodes.
* **Board reader**: only the privileged reader exists (simulator ground truth behind a wrist render); the vision
  reader is a stub (`NotImplementedError`), so the loop 'arm looks at the board' is exercised structurally, not
  perceptually.  The human's move is applied by teleport and then *detected* through the reader (`infer_move`).
* **Policies**: `run_policy.py` / `play_game.py --policy-path` run any checkpoint, but no trained policy ships
  here; the verified checkpoints are a 3-step ACT (which fails the move, as expected) and the untrained ACT of
  `test_env k`.  Success of a policy in the loop is judged by the env (`SO101ChessMove-v0` success definition).
* **Piece meshes**: the collision band hulls cannot follow undercuts (knight muzzle underside, crown points, king's
  cross arms, bishop's mitre: up to 5 mm wider than the exact cut right under the knight's muzzle, see *Piece
  models*); the real pieces' finish, exact profile and masses are not measured (the meshes are a CC0 Staunton set
  scaled to the user's height / base diameter only).  The grasp heights of the 'bands' model are NOT the narrowest
  stems (no vertical room for the 10 mm pads there); a knight is only held reliably within 25 deg of its narrow
  axis.  Right after a release the piece can still touch the fixed pad (the soft pads sink ~0.5 mm into a hull band
  and the position-controlled arm comes back ~1 mm higher than it left): the king leans 2.2 deg until the jaw
  retreats, then stands at 0.02 deg (`test_scene j`); the expert always retreats straight up after opening.
* **Promotion** always produces a queen via the spare queen (one per colour; a second promotion of the same colour
  cannot be executed by the arm: `play_game` teleports it with the regular queen body if that one was captured,
  otherwise the position is not representable and the physical/logical mismatch is reported).  Captured pieces of a *random* env position start in the parking area, not in the drop zone.
* The `dropped` detection needs the piece to come to rest away from both squares; a piece that falls back onto its
  source square is not a failure until `max_steps`.
* Timing: an expert move is 8-23 s of simulation (250-700 control steps); recording with rendering runs at
  ~20 frames/s of wall time (AV1 encoding included).
* **Viewer windows are untested here**: the mjpython entry points (`view_scene.py`, `play_game.py` with a window)
  cannot be run from the headless build session (`launch_passive` never returns there, `--exit-after 5` produced no
  output in 90 s), so only their plain-python paths are covered by the tests (`--headless`, `--no-viewer`,
  `--snapshot`, `--previews`).  Try `$MJ chess_sim/view_scene.py --exit-after 5` and `$MJ chess_sim/play_game.py
  --plies 2` once from a real Terminal and note the result here.
* **Control grid**: `ChessScene.control_step()` keeps `data.time` on a 30 Hz grid anchored at the last reset /
  `settle()` / `set_position(settle_steps=..)`; physics advanced by a raw `step()` re-anchors it at the next control
  step.  The first control steps after `reset()` are full 16/17-step periods (they used to be six 1-step ticks
  because the 0.2 s settle was not on the grid).

## Next steps

1. Crowded positions (PlanErrors on castling with d7/e7/f7 occupied, rank-8 diagonal crossings, randomised Qxd5):
   adaptive jaw opening along the real piece profile, 360 deg wrist-roll search, king-then-rook castling.
2. Vision board reader: detect the game state from the wrist (and overhead) image at the watch pose, replacing
   the privileged `board_state()`; the 7-view scan (`render_scan`) guarantees every square inside the usable window of
   one view (2 % margin) for every board pose of the 'wide' range; merge the per-view detections in board coordinates.
3. Policy training from `record_demos.py` datasets (ACT / diffusion in LeRobot, see *Train* above) on
   `SO101ChessMove-v0`, with `--randomise` for domain randomisation; evaluate with `run_policy.py` and in the loop
   with `play_game.py --policy-path`; then sim2real: confirm fovy / gripper gaps / mount tilt, calibrate the board
   pose relative to the arm, implement `VisionBoardReader` (the real board reader) and fill
   `observation.environment_state` from the game loop on the real robot.
4. Spec items still open: watch-pose roll (rank 1 at the top?), 90 mm lift height on ranks 7-8 (needs a longer
   gripper or a larger `board_gap`; the env's success retreat height follows the same cap there).
