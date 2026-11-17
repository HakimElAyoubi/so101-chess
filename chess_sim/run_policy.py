"""Run a trained LeRobot policy checkpoint in SO101ChessMove-v0 (the policy replaces the scripted expert).

    python chess_sim/run_policy.py --policy-path <ckpt>/pretrained_model \\
        [--episodes 10] [--seed 0] [--device cpu|mps] [--fen FEN --move e7e5] [--randomise] [--max-steps 900]
        [--progress-path FILE]

<ckpt>/pretrained_model is what lerobot-train writes (outputs/.../checkpoints/last/pretrained_model: config.json,
model.safetensors, policy_preprocessor.json, policy_postprocessor.json ...).  Any LeRobot policy type works (act,
diffusion, smolvla, pi0 ...): the type is read from config.json.  The observation adapter is LeRobot's own
preprocess_observation: 'pixels'/'wrist' -> observation.images.wrist (CHW float [0,1]), 'agent_pos' ->
observation.state, 'environment_state' (the one-hot move) -> observation.environment_state, plus the task string;
then the checkpoint's preprocessor (normalisation, device) -> policy.select_action -> postprocessor (unnormalise).
Reports per-episode success / failure / steps and the success rate; exit code 0 iff every episode succeeded.
Progress bar: chess_sim/data/progress.txt (--progress-path), viewer chess_sim/progress_view.command.

PolicyExecutor is also what play_game.py --policy-path uses to execute Black's moves inside the game loop.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_sim import config as C
from chess_sim.progress import ProgressBar
from chess_sim.chess_env import SO101ChessMoveEnv


class PolicyExecutor:
    """A LeRobot policy checkpoint + its pre/post processors, driven one env step at a time."""

    def __init__(self, policy_path: str, device: str = "cpu"):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        self.path = str(policy_path)
        self.device = device
        if not os.path.isfile(os.path.join(self.path, "config.json")):
            raise FileNotFoundError(f"{self.path} has no config.json (pass the pretrained_model directory of a checkpoint)")
        cfg = PreTrainedConfig.from_pretrained(self.path)
        cfg.pretrained_path = self.path
        cfg.device = device
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(self.path, config=cfg)
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=self.path, preprocessor_overrides={"device_processor": {"device": device}})
        self.torch = torch
        self.input_keys = list(cfg.input_features.keys())
        self.reset()

    def reset(self) -> None:
        """Call at the start of every episode / move (clears the policy's action queue)."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def batch(self, obs: dict, task: str) -> dict:
        from lerobot.envs.utils import preprocess_observation
        b = preprocess_observation(obs)
        b["task"] = [task]
        return b

    def act(self, obs: dict, task: str) -> np.ndarray:
        """LeRobot action (float32[6]) for one env observation."""
        b = self.preprocessor(self.batch(obs, task))
        with self.torch.no_grad():
            a = self.policy.select_action(b)
        a = self.postprocessor(a)
        return np.asarray(a[0].detach().cpu().numpy(), np.float32).reshape(6)

    def run_episode(self, env: SO101ChessMoveEnv, seed: Optional[int] = None, options: Optional[dict] = None,
                    max_steps: Optional[int] = None, on_step=None) -> dict:
        """reset + policy rollout through env.step; returns a summary dict like chess_env.run_expert_episode."""
        options = dict(options or {})
        options.setdefault("plan", False)
        obs, info = env.reset(seed=seed, options=options)
        self.reset()
        t0 = time.time()
        steps = 0
        total_r = 0.0
        terminated = truncated = False
        limit = max_steps or env.max_steps
        while not (terminated or truncated) and steps < limit:
            a = self.act(obs, info["task"])
            obs, r, terminated, truncated, info = env.step(a)
            total_r += r
            steps += 1
            if on_step is not None:
                on_step(obs, a, r, info)
        return dict(success=bool(info["success"]), failure=info["failure"], steps=steps, reward=total_r, fen=info["fen"],
                    move=info["move"], task=info["task"], phase=info["phase"], piece_err_mm=info["piece_err_mm"],
                    at_watch=info["at_watch"], wall_s=time.time() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", required=True, help="<checkpoint>/pretrained_model directory")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu (default) or mps")
    ap.add_argument("--fen", default=None)
    ap.add_argument("--move", default=None)
    ap.add_argument("--randomise", action="store_true")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--progress-path", default=C.PROGRESS_TXT)
    a = ap.parse_args(argv)
    ex = PolicyExecutor(a.policy_path, a.device)
    print(f"policy {ex.cfg.type} from {a.policy_path} on {a.device}; inputs {ex.input_keys}")
    env = SO101ChessMoveEnv(randomise=a.randomise, max_steps=a.max_steps)
    pb = ProgressBar(a.episodes, path=a.progress_path)
    pb.update(0, "run_policy: starting", force=True)
    rng = np.random.default_rng(a.seed)
    n_ok = 0
    try:
        for k in range(a.episodes):
            options = {}
            if a.fen:
                options["fen"] = a.fen
            if a.move:
                options["move"] = a.move
            seed = int(rng.integers(0, 2 ** 31 - 1))
            pb.update(k, f"run_policy: episode {k + 1}/{a.episodes}", force=True)

            def tick(obs, act, r, info, _k=k):
                pb.update(_k, f"run_policy: episode {_k + 1}/{a.episodes} {info['move']} step {info['steps']} {info['phase']} | {n_ok} OK")
            res = ex.run_episode(env, seed=seed, options=options, on_step=tick)
            n_ok += res["success"]
            print(f"  episode {k + 1}: {res['move']} {'OK' if res['success'] else 'FAIL ' + str(res['failure'] or res['phase'])} "
                  f"{res['steps']} steps, err {({n: round(v, 1) for n, v in res['piece_err_mm'].items()})}, {res['wall_s']:.1f}s")
            sys.stdout.flush()
    except BaseException:
        pb.fail(f"run_policy: aborted FAILED")
        raise
    pb.done(f"run_policy: {n_ok}/{a.episodes} episodes succeeded DONE")
    print(f"SUMMARY {n_ok}/{a.episodes} episodes succeeded")
    return 0 if n_ok == a.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
