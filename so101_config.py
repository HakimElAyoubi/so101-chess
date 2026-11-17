"""Machine-specific settings for the SO-101 chess project.

Copy config.example.yaml to config.yaml and fill it in; environment variables override the file:

    SO101_FOLLOWER_PORT, SO101_LEADER_PORT, SO101_FOLLOWER_ID, SO101_LEADER_ID, SO101_STOCKFISH, SO101_LIVE_MODEL_REPO

Ports are found with `lerobot-find-port`. A port that is not set is reported by name when the arm is opened, instead of
failing at import, so the simulated runs (--fake) work without any configuration.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load():
    path = ROOT / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except ImportError:                       # PyYAML is optional: the file is flat "key: value" lines
        out = {}
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" in line:
                key, value = line.split(":", 1)
                out[key.strip()] = value.strip().strip("'\"")
        return out


_CONFIG = _load()


def _get(key, env, default=None):
    value = os.environ.get(env) or _CONFIG.get(key)
    return str(value) if value not in (None, "") else default


FOLLOWER_PORT = _get("follower_port", "SO101_FOLLOWER_PORT", "<follower_port not set: see config.example.yaml>")
LEADER_PORT = _get("leader_port", "SO101_LEADER_PORT", "<leader_port not set: see config.example.yaml>")
FOLLOWER_ID = _get("follower_id", "SO101_FOLLOWER_ID", "follower_arm_1")
LEADER_ID = _get("leader_id", "SO101_LEADER_ID", "leader_arm_1")
STOCKFISH = _get("stockfish_path", "SO101_STOCKFISH") or shutil.which("stockfish") or "stockfish"

# Optional default for the live-game model picker; never silently substitutes an older model.
LIVE_MODEL_REPO = _get("live_model_repo", "SO101_LIVE_MODEL_REPO")
