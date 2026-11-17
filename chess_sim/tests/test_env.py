"""Environment tests (plain asserts).  Run from the project root:
    python chess_sim/tests/test_env.py [a b c ...]
(a) gym registration, observation / action spaces match the observations, speed limit of the commanded target
(b) reset with a given fen + move (UCI and SAN), random mid-game resets have Black to move and a legal move
(c) expert rollouts through env.step: >= 4/5 random moves succeed (reward 1.0, phase 'done', <= 3 deg/step)
(d) a 2-episode record_demos run writes a LeRobotDataset that loads back with the right shapes / fps / count
(e) play_game.py --headless --plies 6 completes with every arm move successful (failures are listed) and writes
    its progress bar (--progress-path) in the spec's one-line format, ending with DONE
(f) termination on a toppled piece and truncation at max_steps
(g) guards: non-finite actions raise ValueError (ctrl stays finite), 'move' without 'fen' raises ValueError;
    rank-dependent retreat height (90 mm on ranks 1-6, the reach ceiling - 5 mm on ranks 7-8, fixed override)
    and a rank-7 expert rollout succeeds with it
(h) play_game.Game API: a second promotion of the same colour (spare queen already used) is teleported with the
    regular queen body and the physical board stays consistent with the logical one
(i) play_game.py --no-viewer (the interactive terminal loop) driven over stdin: e2e4 teleported, the arm answers,
    hint / fen / illegal move / quit handled, bar ends at 2/2 DONE; a bad --stockfish-path is an error; --plies 0
(j) board reader + loop plumbing: the privileged reader renders the wrist image and infer_move recovers a teleported
    move (and None for a non-move); the vision reader raises NotImplementedError; reset(keep_scene) runs on the live
    scene and rejects a mismatching fen; play_game --fen with a malformed FEN exits 1 at once (no engine hang)
(k) policy executor: an untrained ACT built from the (d) dataset's metadata (inputs incl. observation.environment_state),
    saved like a lerobot-train checkpoint, is loaded by run_policy.PolicyExecutor, drives SO101ChessMove-v0 through the
    LeRobot observation adapter and executes an arm ply inside play_game.Game (executor 'policy')
Progress bar: chess_sim/data/progress.txt (progress_view.command).  ~4 minutes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import chess
import gymnasium as gym
import numpy as np

from chess_sim import config as C
from chess_sim.progress import ProgressBar
from chess_sim.chess_env import (SO101ChessMoveEnv, ENV_ID, STEP_LIMIT_DEG, RETREAT_HEIGHT, run_expert_episode,
                                 random_position, move_targets, task_string, encode_move, decode_move, MOVE_ENCODING_DIM)

PY = sys.executable
ENV: SO101ChessMoveEnv | None = None
PB: ProgressBar | None = None
RESULTS: dict = {}
TMP = tempfile.mkdtemp(prefix="chess_sim_test_env_")
FEN_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def env() -> SO101ChessMoveEnv:
    global ENV
    if ENV is None:
        ENV = SO101ChessMoveEnv()
    return ENV


def bar(k: int, state: str):
    if PB is not None:
        PB.update(k, state, force=True)


# ---------------------------------------------------------------- (a) spaces
def test_a_spaces():
    e = gym.make(ENV_ID, scene=env().scene)          # registration + gymnasium's passive checker on reset/step
    obs, info = e.reset(seed=0, options={"fen": FEN_AFTER_E4, "move": "e7e5"})
    assert e.observation_space.contains(obs), "reset observation not in observation_space"
    assert e.action_space.shape == (6,) and e.action_space.dtype == np.float32
    img = obs["pixels"]["wrist"]
    assert img.shape == (C.DEFAULT_CONFIG.render_h, C.DEFAULT_CONFIG.render_w, 3) and img.dtype == np.uint8, img.shape
    assert img.std() > 5, "wrist image is flat"
    assert obs["agent_pos"].shape == (6,) and obs["agent_pos"].dtype == np.float32
    assert obs["task"] == "move the piece from e7 to e5", obs["task"]
    # the move-conditioning channel: one-hot from/to squares (what ACT / diffusion consume as observation.environment_state)
    ms = obs["environment_state"]
    assert ms.shape == (MOVE_ENCODING_DIM,) and ms.dtype == np.float32 and ms.sum() == 2.0, (ms.shape, ms.dtype, ms.sum())
    assert decode_move(ms).uci() == "e7e5" and np.array_equal(ms, encode_move("e7e5"))
    assert decode_move(encode_move("a7a8q"), chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")).uci() == "a7a8q"
    a = e.action_space.sample()
    obs2, r, term, trunc, info2 = e.step(a)
    assert e.observation_space.contains(obs2) and isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    # speed limit: the commanded target moves at most STEP_LIMIT_DEG per joint per step
    ctrl0 = env().scene.get_ctrl()
    far = np.array([80.0, 80.0, -80.0, 80.0, 120.0, 100.0], np.float32)
    e.step(far)
    ctrl1 = env().scene.get_ctrl()
    dmax = float(np.max(np.abs(ctrl1[:5] - ctrl0[:5])))
    assert dmax <= STEP_LIMIT_DEG + 1e-6, dmax
    assert abs(ctrl1[5] - 100.0) < 1e-6
    print(f"    obs: pixels {img.shape} {img.dtype}, agent_pos {obs['agent_pos'].round(1)}, environment_state one-hot "
          f"{int(np.argmax(ms[:64]))}->{int(np.argmax(ms[64:]))} = {decode_move(ms).uci()}, task {obs['task']!r}; "
          f"action space {e.action_space.low.round(0)}..{e.action_space.high.round(0)}; max target step {dmax:.2f} deg "
          f"(limit {STEP_LIMIT_DEG:.2f})")
    for k in ("fen", "move", "task", "phase", "success", "failure", "targets", "piece_err_mm", "pieces", "board", "steps", "sim_time"):
        assert k in info, k
    assert len(info["pieces"]) == 34 and len(info["pieces"]["b_king_1"]) == 7
    e.close()
    RESULTS["spaces"] = dict(img=img.shape, max_step=dmax)


# ---------------------------------------------------------------- (b) reset with fen / move
def test_b_reset_fen_move():
    e = env()
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3"
    obs, info = e.reset(seed=1, options={"fen": fen, "move": "g8f6"})
    assert info["fen"] == fen and info["move"] == "g8f6" and info["task"] == "move the piece from g8 to f6"
    exp = {chess.square_name(sq): pc.symbol() for sq, pc in chess.Board(fen).piece_map().items()}
    assert info["board"] == exp, "physical board does not match the FEN"
    assert "b_knight_2" in info["targets"] or "b_knight_1" in info["targets"]
    name = [n for n in info["targets"] if "knight" in n][0]
    tgt = np.array(info["targets"][name])
    assert np.linalg.norm(tgt[:2] - e.scene.square_xyz("f6")[:2]) < 1e-9, tgt
    assert e.plan is not None and e.plan_error is None
    # SAN input, a capture: the captured pawn's target is the first drop cell
    fen2 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    obs, info = e.reset(seed=2, options={"fen": fen2, "move": "dxe4", "plan": False})
    assert info["move"] == "d5e4" and len(info["targets"]) == 2, info["targets"]
    cap = [n for n in info["targets"] if n.startswith("w_")][0]
    assert np.linalg.norm(np.array(info["targets"][cap])[:2] - e.scene.drop_cell_xyz(0, 0)[:2]) < 1e-9
    # random mid-game positions: Black to move, chosen move legal and plannable, 1..29 plies
    plies = []
    drops = []
    t0 = time.time()
    for seed in range(5):
        obs, info = e.reset(seed=100 + seed)
        b = chess.Board(info["fen"])
        assert b.turn == chess.BLACK and not b.is_game_over()
        assert chess.Move.from_uci(info["move"]) in b.legal_moves
        assert e.plan is not None, e.plan_error
        plies.append(2 * (b.fullmove_number - 1) + 1)
        assert e.observation_space.contains(obs)
        assert decode_move(obs["environment_state"]).uci()[:4] == info["move"][:4]
        # the pieces captured in the random game stand in the drop zone (capture order), not in the parking area
        n_cap = 32 - len(b.piece_map())
        in_drop = sum(1 for n in e.scene.piece_names for cell in e.scene.drop_zone_cells()
                      if np.linalg.norm(e.scene.piece_pos(n)[:2] - e.scene.drop_cell_xyz(*cell)[:2]) < 0.005
                      and e.scene.piece_tilt_deg(n) < 5)
        drops.append((n_cap, in_drop))
        assert in_drop == min(n_cap, 16), (seed, n_cap, in_drop)
    print(f"    fen/move reset OK (targets {list(info['targets'])}); 5 random resets: plies {plies}, (captures, in drop zone) "
          f"{drops}, {(time.time() - t0) / 5:.1f} s per reset incl. planning")
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = random_position(rng)
        assert b.turn == chess.BLACK and not b.is_game_over()
    RESULTS["reset"] = dict(plies=plies)


# ---------------------------------------------------------------- (c) expert rollout
def test_c_expert_rollout():
    e = env()
    n_ok = 0
    rows = []
    for i, seed in enumerate((11, 12, 13, 14, 15)):
        bar(2 + i, f"test_env: expert rollout {i + 1}/5")
        steps = []

        def on_step(obs, a, r, info, _steps=steps):
            _steps.append(np.asarray(a, float))
        t0 = time.time()
        res = run_expert_episode(e, seed=seed, on_step=on_step)
        acts = np.array(steps)
        dmax = float(np.max(np.abs(np.diff(acts[:, :5], axis=0)))) if len(acts) > 1 else 0.0
        n_ok += res["success"]
        rows.append((res["move"], res["success"], res["failure"], res["steps"], max(res["piece_err_mm"].values()), dmax, res["phase"]))
        print(f"    seed {seed}: {res['move']:6s} {'OK  ' if res['success'] else 'FAIL'} {res['steps']:4d} steps, err "
              f"{max(res['piece_err_mm'].values()):4.1f} mm, max action step {dmax:.2f} deg, phase {res['phase']}, "
              f"reward {res['reward']:.1f}, failure {res['failure']}, abort {res['expert_abort']} ({time.time() - t0:.0f}s)")
        assert dmax <= STEP_LIMIT_DEG + 1e-6, dmax
        if res["success"]:
            assert res["reward"] == 1.0 and res["phase"] == "done" and res["at_watch"]
    print(f"    {n_ok}/5 expert rollouts succeed through env.step")
    assert n_ok >= 4, rows
    RESULTS["rollout"] = (n_ok, 5)


# ---------------------------------------------------------------- (d) record_demos
def test_d_record_demos():
    from chess_sim.record_demos import record, verify
    out = os.path.join(TMP, "demos2")
    bar(7, "test_env: record_demos 2 episodes")
    t0 = time.time()
    summ = record(2, out, "test/so101_chess", seed=5, env=env(), verbose=True, progress_path=os.path.join(TMP, "progress.txt"))
    assert summ["saved"] == 2, summ
    info = verify(out, "test/so101_chess")
    assert info["num_episodes"] == 2 and info["fps"] == C.CONTROL_HZ, info
    assert info["shapes"]["observation.images.wrist"] == (3, C.DEFAULT_CONFIG.render_h, C.DEFAULT_CONFIG.render_w), info["shapes"]
    assert info["shapes"]["observation.state"] == (6,) and info["shapes"]["action"] == (6,)
    assert info["shapes"]["observation.environment_state"] == (MOVE_ENCODING_DIM,), info["shapes"]
    assert info["task"].startswith("move the piece from ")
    assert info["task"].endswith(f"from {info['move'][:2]} to {info['move'][2:4]}"), (info["task"], info["move"])
    with open(os.path.join(out, "episodes.json")) as f:
        side = json.load(f)
    saved = [r for r in side["episodes"] if r["episode_index"] is not None]
    assert len(saved) == 2 and info["num_frames"] == sum(r["steps"] for r in saved), (info["num_frames"], [r["steps"] for r in saved])
    assert all(r["success"] for r in saved)
    print(f"    dataset {out}: {info['num_episodes']} episodes, {info['num_frames']} frames, fps {info['fps']}, "
          f"features {info['features'][:4]}, move decoded from observation.environment_state {info['move']} "
          f"({time.time() - t0:.0f}s)")
    RESULTS["dataset"] = info
    RESULTS["dataset_dir"] = out


# ---------------------------------------------------------------- (e) play_game headless
def test_e_play_game_headless():
    bar(8, "test_env: play_game --headless --plies 6")
    prog = os.path.join(TMP, "play_game_progress.txt")
    cmd = [PY, os.path.join(ROOT, "chess_sim", "play_game.py"), "--headless", "--plies", "6", "--seed", "0", "--time", "0.05",
           "--progress-path", prog]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    out = "\n".join(l for l in p.stdout.splitlines() if "objc[" not in l)
    print("    " + out.replace("\n", "\n    "))
    if p.returncode != 0:
        print("    stderr:", p.stderr[-2000:])
    summ = [l for l in out.splitlines() if l.startswith("SUMMARY")]
    assert summ, "no SUMMARY line"
    import re
    m = re.search(r"arm moves (\d+)/(\d+) successful", summ[0])
    ok, n = int(m.group(1)), int(m.group(2))
    failed = [l.strip() for l in out.splitlines() if l.strip().startswith("failed:")]
    print(f"    {ok}/{n} arm moves successful in {time.time() - t0:.0f}s; failed: {failed}")
    assert n == 3 and ok == n and p.returncode == 0, (ok, n, failed, p.returncode)
    # progress bar written by the game in the spec's one-line format, finished with DONE
    import re as _re
    assert os.path.exists(prog), "play_game wrote no progress bar"
    line = open(prog).read().strip()
    print(f"    progress bar: {line}")
    m = _re.match(r"^\[#{10}\] 100%  6/6  elapsed \d+m\d\ds  ETA 0m00s \| play_game: 6 plies, arm 3/3 OK, result .* DONE$", line)
    assert m, line
    RESULTS["play_game"] = (ok, n)


# ---------------------------------------------------------------- (f) termination / truncation
def test_f_termination():
    e = env()
    obs, info = e.reset(seed=3, options={"fen": FEN_AFTER_E4, "move": "e7e5", "plan": False})
    # topple a bystander by hand -> terminated with failure 'toppled:<name>'
    victim = e.scene.board_state()["d7"]
    p = e.scene.piece_pos(victim)
    # lay it on its side (90 deg about x, base centre one base radius above the surface): it cannot right itself
    e.scene.set_piece_pose(victim, [p[0], p[1], e.scene.board_top_z + 0.009], quat=[np.cos(np.radians(45)), np.sin(np.radians(45)), 0, 0])
    e.scene.settle(0.3)
    assert e.scene.piece_tilt_deg(victim) > 60, e.scene.piece_tilt_deg(victim)
    obs, r, term, trunc, info = e.step(obs["agent_pos"])
    assert term and info["failure"] == f"toppled:{victim}" and info["phase"] == "failed" and r == 0.0, info["failure"]
    # truncation at max_steps while holding still
    e2 = SO101ChessMoveEnv(scene=e.scene, max_steps=20, obs_pixels=False, dense_reward=True)
    obs, info = e2.reset(seed=4, options={"fen": FEN_AFTER_E4, "move": "e7e5", "plan": False})
    assert "pixels" not in obs and "pixels" not in e2.observation_space.spaces
    for k in range(20):
        obs, r, term, trunc, info = e2.step(obs["agent_pos"])
        assert not term, info["failure"]
        assert r < 0.0, r                       # dense shaping is negative until the piece is placed
    assert trunc and info["steps"] == 20 and info["phase"] == "reach"
    print(f"    toppled bystander -> terminated ({info and victim}); 20-step hold -> truncated, dense reward {r:.3f}")
    RESULTS["termination"] = True


# ---------------------------------------------------------------- (g) guards + rank-dependent retreat height
def test_g_guards_and_retreat():
    e = SO101ChessMoveEnv(scene=env().scene, obs_pixels=False)
    try:
        e.step(np.zeros(6))
        raise AssertionError("step() before reset() accepted")
    except RuntimeError as ex:
        assert "reset" in str(ex), ex
    obs, info = e.reset(seed=0, options={"fen": FEN_AFTER_E4, "move": "e7e5", "plan": False})
    assert abs(info["retreat_height"] - RETREAT_HEIGHT) < 1e-9 and RETREAT_HEIGHT == C.LIFT_CLEARANCE, info["retreat_height"]
    for bad in (np.array([np.nan] * 6), np.array([0, 0, 0, np.inf, 0, 0], float)):
        try:
            e.step(bad)
            raise AssertionError("non-finite action accepted")
        except ValueError:
            pass
    assert np.all(np.isfinite(e.scene.get_ctrl())), e.scene.get_ctrl()
    obs, r, term, trunc, info = e.step(obs["agent_pos"])          # the env still works after the rejected actions
    assert not term and np.all(np.isfinite(info["pieces"]["b_king_1"]))
    try:
        e.reset(seed=1, options={"move": "e7e5"})
        raise AssertionError("'move' without 'fen' accepted")
    except ValueError:
        pass
    # rank-dependent retreat height: 90 mm where the lift height is reachable, lower on ranks 7-8 (reach ceiling)
    fen2 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    heights = {}
    for fen, mv in ((FEN_AFTER_E4, "e7e5"), (FEN_AFTER_E4, "g8f6"), (fen2, "d8d6"), (fen2, "d8d7"),
                    ("r3kbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2", "e8c8")):
        obs, info = e.reset(seed=0, options={"fen": fen, "move": mv, "plan": False})
        heights[mv] = info["retreat_height"]
        xy = np.array(list(info["targets"].values())[-1])[:2]
        ceiling = e.expert.lift_z_at(xy) - e.scene.board_top_z
        assert info["retreat_height"] <= C.LIFT_CLEARANCE + 1e-9 and info["retreat_height"] <= ceiling + 1e-9, (mv, info["retreat_height"], ceiling)
    assert abs(heights["e7e5"] - 0.090) < 1e-9 and abs(heights["g8f6"] - 0.090) < 1e-9 and abs(heights["d8d6"] - 0.090) < 1e-9
    assert 0.02 <= heights["d8d7"] < 0.060 and 0.02 <= heights["e8c8"] < heights["d8d7"], heights
    e_fixed = SO101ChessMoveEnv(scene=env().scene, obs_pixels=False, retreat_height=0.060)
    obs, info = e_fixed.reset(seed=0, options={"fen": FEN_AFTER_E4, "move": "e7e5", "plan": False})
    assert abs(info["retreat_height"] - 0.060) < 1e-9
    # a rank-7 move (d8d7, lowered retreat) and a rank-5 move (90 mm) succeed through env.step with the latched retreat
    t0 = time.time()
    res7 = run_expert_episode(e, seed=0, options={"fen": fen2, "move": "d8d7"})
    res5 = run_expert_episode(e, seed=0, options={"fen": FEN_AFTER_E4, "move": "e7e5"})
    print(f"    NaN/inf actions and 'move' without 'fen' rejected; retreat heights (mm): "
          f"{ {k: round(v * 1000, 1) for k, v in heights.items()} }; rollouts d8d7 {res7['success']} ({res7['steps']} steps), "
          f"e7e5 {res5['success']} ({res5['steps']} steps) in {time.time() - t0:.0f}s")
    assert res7["success"] and res7["phase"] == "done", res7
    assert res5["success"] and res5["phase"] == "done", res5
    RESULTS["guards"] = heights


# ---------------------------------------------------------------- (h) play_game API: double promotion fallback
def test_h_double_promotion_fallback():
    from chess_sim.play_game import Game, expected_symbols, start_progress, finish_progress
    bar(9, "test_env: Game API double promotion")
    scene = env().scene
    g = Game(scene, time_limit=0.05, verbose=False)
    try:
        g.board = chess.Board("4k3/8/8/8/8/8/p1p5/4K2R b - - 0 1")      # Black's queen was captured: two promotions
        scene.set_position(g.board, settle_steps=50)
        prog = os.path.join(TMP, "api_progress.txt")
        start_progress(g, 3, prog, "api")
        t0 = time.time()
        r1 = g.arm_move([chess.Move.from_uci("a2a1q")]); g.records.append(r1)
        assert r1["executor"] == "arm" and r1["success"], r1
        assert scene.board_state_symbols() == expected_symbols(g.board)
        g.play_ply(chess.Move.from_uci("h1h2"))
        r2 = g.arm_move([chess.Move.from_uci("c2c1q")]); g.records.append(r2)
        assert r2["executor"] == "teleport" and not r2["success"] and r2["plan_errors"], r2
        # the fallback used the regular queen body: physical == logical, both queens on the board
        assert scene.board_state_symbols() == expected_symbols(g.board), (scene.board_state_symbols(), expected_symbols(g.board))
        assert scene.board_state_symbols()["c1"] == "q" and scene.board_state_symbols()["a1"] == "q"
        assert g.physical_matches()
        finish_progress(g, g.summary())
        line = open(prog).read().strip()
        assert line.startswith("[##########] 100%  3/3") and line.endswith("DONE"), line
        print(f"    a2a1q by the arm ({r1['steps']} steps, {r1['piece_err_mm']}); c2c1q teleported with the regular queen body "
              f"({r2['plan_errors'][0][:60]}...); physical == logical; bar: {line[-60:]} ({time.time() - t0:.0f}s)")
    finally:
        g.close()
    RESULTS["double_promotion"] = True


# ---------------------------------------------------------------- (i) play_game interactive mode (stdin)
def test_i_play_game_interactive():
    """The default (terminal-input) game loop, driven over a pipe with --no-viewer: the typed white move is
    teleported, the arm answers, 'hint' / 'fen' / an illegal move / 'quit' are handled, the bar ends at N/N."""
    import re
    bar(10, "test_env: play_game --no-viewer (interactive over stdin)")
    prog = os.path.join(TMP, "interactive_progress.txt")
    cmd = [PY, os.path.join(ROOT, "chess_sim", "play_game.py"), "--no-viewer", "--seed", "0", "--time", "0.05",
           "--progress-path", prog]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, input="e2e4\nhint\nfen\nxx9\nquit\n", capture_output=True, text=True, timeout=600)
    out = "\n".join(l for l in p.stdout.splitlines() if "objc[" not in l)
    if p.returncode != 0:
        print("    stdout:", out[-3000:], "\n    stderr:", p.stderr[-2000:])
    assert "Exception in thread" not in p.stderr and "Traceback" not in p.stderr, p.stderr[-2000:]
    # the 'White to move> ' prompt has no newline (the typed Enter supplies it in a terminal), so with piped
    # stdin the next output line is glued to it: strip that prefix before matching
    lines = [re.sub(r"^(White to move> )+", "", l) for l in out.splitlines()]
    white = [l for l in lines if re.match(r"\s*1\. white\s+e4\s+\(e2e4\)\s+teleport\s+OK", l)]
    black = [l for l in lines if re.match(r"\s*2\. black\s+\S+\s+\(\w{4}\)\s+arm\s+OK", l)]
    hint = [l for l in lines if l.startswith("hint: ")]
    fen = [l for l in lines if _is_fen(l.strip())]
    bad = [l for l in lines if "illegal or unparsable move 'xx9'" in l]
    summ = [l for l in lines if l.startswith("SUMMARY")]
    assert white and black and hint and fen and bad and summ, (white, black, hint, fen, bad, summ)
    m = re.search(r"SUMMARY plies (\d+)\s+arm moves (\d+)/(\d+) successful", summ[0])
    plies, ok, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert plies == 2 and n == 1 and ok == 1 and p.returncode == 0, (plies, ok, n, p.returncode)
    assert chess.Board(fen[0].strip()).turn == chess.WHITE and chess.Board(fen[0].strip()).fullmove_number == 2, fen[0]
    line = open(prog).read().strip()
    assert re.match(r"^\[#{10}\] 100%  2/2  elapsed \d+m\d\ds  ETA 0m00s \| play_game: 2 plies, arm 1/1 OK, result \* DONE$", line), line
    print(f"    {white[0].strip()} | {black[0].strip()[:60]} | {hint[0]} | fen {fen[0].strip()[:40]}... | {bad[0][:45]}... | "
          f"{summ[0][:60]} | bar: {line[:30]}...{line[-12:]} ({time.time() - t0:.0f}s)")
    # a bad explicit --stockfish-path is an error, not a silent fallback to the default binary
    p2 = subprocess.run([PY, os.path.join(ROOT, "chess_sim", "play_game.py"), "--headless", "--plies", "1",
                         "--stockfish-path", "/nonexistent/stockfish"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p2.returncode != 0 and "FileNotFoundError" in p2.stderr and "/nonexistent/stockfish" in p2.stderr, (p2.returncode, p2.stderr[-500:])
    assert "SUMMARY" not in p2.stdout
    # --plies 0: no ply played, the bar still finishes with DONE (ProgressBar needs total >= 1 -> 1/1)
    prog0 = os.path.join(TMP, "plies0_progress.txt")
    p3 = subprocess.run([PY, os.path.join(ROOT, "chess_sim", "play_game.py"), "--headless", "--plies", "0", "--progress-path", prog0],
                        cwd=ROOT, capture_output=True, text=True, timeout=300)
    line0 = open(prog0).read().strip()
    assert p3.returncode == 0 and "SUMMARY plies 0" in p3.stdout and line0.endswith("play_game: 0 plies, arm 0/0 OK, result * DONE"), (p3.stdout[-300:], line0)
    print(f"    bad --stockfish-path -> exit {p2.returncode} ({p2.stderr.strip().splitlines()[-1][:80]}); --plies 0 bar: {line0[:30]}...")
    RESULTS["interactive"] = (plies, ok, n)


# ---------------------------------------------------------------- (j) board reader, keep_scene, --fen validation
def test_j_board_reader_and_loop():
    from chess_sim.board_reader import PrivilegedBoardReader, VisionBoardReader, infer_move, expected_symbols, make_reader
    bar(11, "test_env: board reader / keep_scene / --fen")
    e = env()
    s = e.scene
    b = chess.Board()
    s.reset(seed=0, randomise=False, fen=b.fen())
    r = PrivilegedBoardReader()
    seen = r.read(s)
    assert seen == expected_symbols(b) and r.last_image.shape == (C.DEFAULT_CONFIG.render_h, C.DEFAULT_CONFIG.render_w, 3)
    assert infer_move(b, seen) is None                                      # nothing moved: no legal move explains it
    st = s.board_state()
    s.set_piece_pose(st["g1"], s.square_xyz("f3", 0.0005))                # the human plays Nf3
    s.settle(0.1)
    mv = infer_move(b, r.read(s))
    assert mv == chess.Move.from_uci("g1f3"), mv
    b.push(mv)
    # a capture and a promotion are told apart by the symbols
    b2 = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    s.set_position(b2)
    st = s.board_state()
    s.set_piece_pose(st["a7"], s.cfg.parking_xy(30) + (0.0005,))
    s.set_piece_pose("w_queen_spare_1", s.square_xyz("a8", 0.0005))
    s.settle(0.1)
    assert infer_move(b2, r.read(s)) == chess.Move.from_uci("a7a8q")
    try:
        VisionBoardReader().read(s)
        raise AssertionError("vision reader did not raise")
    except NotImplementedError:
        pass
    assert isinstance(make_reader("privileged"), PrivilegedBoardReader)
    # keep_scene: the episode runs on the scene as it is (a jittered piece keeps its place); a mismatch is rejected
    s.reset(seed=0, randomise=False, fen=b.fen())
    st = s.board_state()
    p = s.square_xyz("e7", 0.0005) + np.array([0.003, -0.002, 0.0])
    s.set_piece_pose(st["e7"], p)
    s.settle(0.1)
    obs, info = e.reset(options={"fen": b.fen(), "move": "e7e5", "plan": False, "keep_scene": True})
    assert info["move"] == "e7e5" and np.linalg.norm(s.piece_pos(st["e7"])[:2] - p[:2]) < 0.001, s.piece_pos(st["e7"])
    for bad in ({"fen": chess.STARTING_FEN, "move": "e2e4", "plan": False, "keep_scene": True}, {"keep_scene": True}):
        try:
            e.reset(options=bad)
            raise AssertionError(f"keep_scene accepted {bad}")
        except ValueError:
            pass
    # play_game --fen validation: a malformed FEN exits 1 immediately (it used to hang with the engine open)
    t0 = time.time()
    p1 = subprocess.run([PY, os.path.join(ROOT, "chess_sim", "play_game.py"), "--headless", "--plies", "1", "--fen", "not a fen"],
                        cwd=ROOT, capture_output=True, text=True, timeout=60)
    dt = time.time() - t0
    assert p1.returncode == 1 and "not a valid FEN" in p1.stdout and dt < 30, (p1.returncode, p1.stdout[-200:], dt)
    p2 = subprocess.run([PY, os.path.join(ROOT, "chess_sim", "play_game.py"), "--headless", "--plies", "1",
                         "--fen", "qqq1k3/8/8/8/8/8/8/4K3 w - - 0 1"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert p2.returncode == 1 and "more pieces" in p2.stdout, p2.stdout[-200:]
    print(f"    privileged reader: start position read, Nf3 inferred, a7a8q inferred, vision reader NotImplementedError; "
          f"keep_scene keeps the jittered e7 pawn and rejects a mismatch; bad --fen exits 1 in {dt:.1f}s")
    RESULTS["reader"] = True


# ---------------------------------------------------------------- (k) policy executor (untrained ACT checkpoint)
def test_k_policy_executor():
    from chess_sim.run_policy import PolicyExecutor
    from chess_sim.play_game import Game, run_auto
    bar(11, "test_env: policy executor (untrained ACT)")
    ds_dir = RESULTS.get("dataset_dir")
    if ds_dir is None or not os.path.isdir(ds_dir):           # (d) did not run: record one episode now
        from chess_sim.record_demos import record
        ds_dir = os.path.join(TMP, "demos1")
        record(1, ds_dir, "test/so101_chess", seed=5, env=env(), verbose=False, progress_path=os.path.join(TMP, "progress.txt"))
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
    t0 = time.time()
    meta = LeRobotDatasetMetadata("test/so101_chess", root=ds_dir)
    cfg = make_policy_config("act", device="cpu", push_to_hub=False, chunk_size=20, n_action_steps=20, dim_model=64,
                             n_encoder_layers=1, n_decoder_layers=1, n_heads=2, dim_feedforward=128)
    pol = make_policy(cfg, ds_meta=meta)
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=meta.stats)
    ck = os.path.join(TMP, "ckpt", "pretrained_model")
    os.makedirs(ck, exist_ok=True)
    pol.save_pretrained(ck)
    pre.save_pretrained(ck)
    post.save_pretrained(ck)
    ex = PolicyExecutor(ck, "cpu")
    assert ex.cfg.type == "act" and "observation.environment_state" in ex.input_keys and "observation.images.wrist" in ex.input_keys, ex.input_keys
    e = env()
    obs, info = e.reset(seed=0, options={"fen": FEN_AFTER_E4, "move": "e7e5", "plan": False})
    a = ex.act(obs, info["task"])
    assert a.shape == (6,) and a.dtype == np.float32 and np.all(np.isfinite(a)), a
    res = ex.run_episode(e, seed=0, options={"fen": FEN_AFTER_E4, "move": "e7e5"}, max_steps=12)
    assert res["steps"] == 12 and res["move"] == "e7e5" and isinstance(res["success"], bool), res
    # inside the game loop: the policy executes Black's ply on the live scene (keep_scene), the reader sees White's
    g = Game(e.scene, time_limit=0.05, verbose=False, policy=ex, policy_max_steps=15)
    try:
        summ = run_auto(g, 2, os.path.join(TMP, "policy_game_progress.txt"))
    finally:
        g.close()
    assert summ["plies"] == 2 and summ["arm_moves"] == 1 and summ["executor"] == "policy", summ
    assert summ["reads"] == 1 and summ["reads_ok"] == 1 and g.records[0]["observed"] == g.records[0]["uci"], g.records[0]
    assert g.records[1]["executor"] == "policy" and g.records[1]["steps"] == 15 and g.physical_matches(), g.records[1]
    print(f"    untrained ACT ({sum(p.numel() for p in pol.parameters()) / 1e6:.1f} M params) loaded from {ck}: inputs {ex.input_keys}; "
          f"12 policy steps in the env ({res['phase']}); play_game: White {g.records[0]['uci']} read as {g.records[0]['observed']}, "
          f"Black {g.records[1]['uci']} executed by the policy for {g.records[1]['steps']} steps "
          f"({'OK' if g.records[1]['success'] else 'FAIL, resynced'}) ({time.time() - t0:.0f}s)")
    RESULTS["policy"] = True


def _is_fen(s: str) -> bool:
    try:
        return len(s.split()) == 6 and chess.Board(s).is_valid()
    except ValueError:
        return False


TESTS = [test_a_spaces, test_b_reset_fen_move, test_c_expert_rollout, test_d_record_demos, test_e_play_game_headless,
         test_f_termination, test_g_guards_and_retreat, test_h_double_promotion_fallback, test_i_play_game_interactive,
         test_j_board_reader_and_loop, test_k_policy_executor]


def main() -> int:
    global PB
    PB = ProgressBar(13)
    PB.update(0, "test_env: starting", force=True)
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    failed = 0
    times = []
    try:
        for i, t in enumerate(TESTS):
            if only and not any(t.__name__.startswith(f"test_{o}_") for o in only):
                continue
            PB.update(min(12, i + 1), f"test_env: {t.__name__}", force=True)
            t0 = time.time()
            try:
                t()
                dt = time.time() - t0
                print(f"PASS {t.__name__} ({dt:.1f}s)")
            except Exception:
                dt = time.time() - t0
                failed += 1
                print(f"FAIL {t.__name__} ({dt:.1f}s)")
                traceback.print_exc()
            times.append((t.__name__, dt))
            sys.stdout.flush()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("timings: " + ", ".join(f"{n} {d:.1f}s" for n, d in times) + f" | total {sum(d for _, d in times):.1f}s")
    n_run = len(times)
    print(f"{n_run - failed}/{n_run} passed")
    PB.done(f"test_env: {n_run - failed}/{n_run} PASS" if not failed else f"test_env: {failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
