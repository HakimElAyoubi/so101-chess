"""Load the fine-tuned SmolVLA (final model or an intermediate checkpoint) on the Mac and run one control tick.

Same path as LeRobot's synchronous rollout: prepare_observation_for_inference -> preprocessor -> select_action ->
postprocessor. The trained preprocessor already renames observation.images.wrist -> camera1."""
from pathlib import Path

import numpy as np
import torch

REPO = "LeHak0/smolvla_chess_v1"
# model variants: which repo, and what extra input the model was trained with
VARIANTS = {"v1": "LeHak0/smolvla_chess_v1",                    # task text only
            "coords": "LeHak0/smolvla_chess_coords",            # + from/to square xy appended to the state
            "marker": "LeHak0/smolvla_chess_marker",            # + target markers drawn into the wrist image
            "skills": "LeHak0/smolvla_chess_skills"}            # grasp / place skills for the staged player (plan step 2-3)
SKILL_ACTION_STEPS = 25          # skills: re-plan every 25 of the 50 actions (closer to the pieces, plan step 2)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CHECKPOINTS = ["final", "007500", "005000", "010000", "015000", "020000"]


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def local_path(checkpoint="final", repo=REPO):
    """Download (once) and return the folder holding config.json / model.safetensors / processors."""
    from huggingface_hub import snapshot_download
    if checkpoint != "final" and not (checkpoint.isdigit() and len(checkpoint) == 6):
        raise ValueError("Checkpoint must be final or a six-digit step.")
    if Path(repo).is_dir():                                  # a local folder laid out like the Hub repo (tests, a run's box)
        return Path(repo) if checkpoint == "final" else Path(repo) / "checkpoints" / checkpoint
    if checkpoint == "final":
        root = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer/*"],
                                 ignore_patterns=["checkpoints/*", "logs/*"])
        return Path(root)
    root = snapshot_download(repo, allow_patterns=[f"checkpoints/{checkpoint}/*"])
    return Path(root) / "checkpoints" / checkpoint


def load_policy(checkpoint="final", variant="v1", repo=None):
    """Policy (SmolVLA / ACT checkpoints) or the RL skill (rl/skill_policy.py) when `repo` is an RL run folder or
    a Hub path LeHak0/so101_rl_skills/<run> (rl_policy.json present)."""
    import sys
    if repo == "scripted:place":
        from scripted_place import ScriptedPlacePolicy
        return ScriptedPlacePolicy(checkpoint,variant,repo=repo)
    if repo and str(repo).startswith("astra:"):
        from astra_skill import AstraSkill
        return AstraSkill(checkpoint, variant, repo=repo)
    if repo and str(repo).startswith("claude:"):                    # Claude as the skill (play/llm_skill.py)
        from llm_skill import ClaudeSkill
        return ClaudeSkill(checkpoint, variant, repo=repo)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))
    try:
        from skill_policy import RLSkillPolicy, is_rl_repo
    except Exception:
        return Policy(checkpoint, variant, repo=repo)
    if repo and is_rl_repo(repo):
        return RLSkillPolicy("best" if checkpoint in ("final", None) else checkpoint, variant, repo=repo)
    return Policy(checkpoint, variant, repo=repo)


class Policy:
    def __init__(self, checkpoint="final", variant="v1", repo=None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class
        self.checkpoint, self.variant, self.dev = checkpoint, variant, device()
        path = local_path(checkpoint, repo or VARIANTS[variant])
        self.proj, self.goal = None, None
        if variant in ("coords", "marker", "skills"):
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "goal"))
            from targets import LATEST, Projector
            self.proj, self.session = Projector(), LATEST   # live board pose = the latest calibrated session
        if not (path / "config.json").exists() and (path / "pretrained_model" / "config.json").exists():
            path = path / "pretrained_model"
        self.path = path
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = self.dev.type                          # trained on cuda
        if variant == "skills":
            cfg.n_action_steps = min(cfg.n_action_steps, SKILL_ACTION_STEPS)
        self.type = cfg.type                                # smolvla | act (step-3 skill models)
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(self.dev).eval()
        ov = {"device_processor": {"device": self.dev.type}}
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=path, preprocessor_overrides=ov,
                                                       postprocessor_overrides=ov)
        self.n_steps = cfg.n_action_steps

    def reset(self):
        self.policy.reset()

    def observation(self, state6, image_hwc_uint8, task):
        """Model input for this variant (adds the goal coordinates / draws the target markers)."""
        state = np.asarray(state6, np.float32)
        img = image_hwc_uint8
        if self.variant == "coords":
            f, t = self.proj.target_xyz(task, self.session)
            state = np.concatenate([state, np.array([f[0], f[1], t[0], t[1]], np.float32)])
        elif self.variant == "marker":
            img = self.proj.draw_markers(img.copy(), state6, task, self.session)
        elif self.variant == "skills" and self.goal is not None:   # the staged player sets goal: move, marker, drop spot
            g = self.goal                                   # (goal None: the image already has its marker, play/val_check.py)
            img = self.proj.draw_markers(img.copy(), state6, g["move"], self.session, which=g["which"], to_xyz=g.get("to_xyz"))
        return {"observation.state": state, "observation.images.wrist": img}

    @torch.inference_mode()
    def act(self, state6, image_hwc_uint8, task):
        """One action per call (the policy's own 50-step queue). state6: [6] LeRobot deg / %, image: HxWx3 RGB."""
        from lerobot.policies.utils import prepare_observation_for_inference
        batch = prepare_observation_for_inference(self.observation(state6, image_hwc_uint8, task), self.dev, task, "so_follower")
        a = self.post(self.policy.select_action(self.pre(batch)))
        return a.squeeze(0).float().cpu().numpy()

    def fixed_noise(self, seed=0):
        """One flow-matching noise sample, reused for every chunk of a trial so consecutive plans agree."""
        if self.type != "smolvla":
            return None                                     # ACT samples no noise at inference
        g = torch.Generator().manual_seed(seed)
        c = self.policy.config
        return torch.randn((1, c.chunk_size, c.max_action_dim), generator=g).to(self.dev)

    @torch.inference_mode()
    def chunk(self, state6, image_hwc_uint8, task, noise=None):
        """A whole action chunk [n, 6] for the current observation (used by the asynchronous executor)."""
        from lerobot.policies.utils import prepare_observation_for_inference
        batch = prepare_observation_for_inference(self.observation(state6, image_hwc_uint8, task), self.dev, task, "so_follower")
        kw = {"noise": noise} if self.type == "smolvla" else {}
        a = self.policy.predict_action_chunk(self.pre(batch), **kw)                 # (1, n, d) normalised
        a = self.post(a)
        return a.squeeze(0).float().cpu().numpy()[:, :6]
