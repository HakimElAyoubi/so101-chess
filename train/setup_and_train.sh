#!/usr/bin/env bash
# First SmolVLA fine-tune on LeHak0/so101_chess_moves. For a rented Linux GPU box (Lambda / Vast.ai, Ubuntu + CUDA).
#
#   1. copy this file to the machine (or paste it), then:   bash setup_and_train.sh
#   2. when asked, log in to Hugging Face (token with WRITE access) and to Weights & Biases
#   3. training runs inside tmux (survives disconnects):    tmux attach -t train
#
# Result: checkpoints in ~/outputs/train/smolvla_chess_v1, final model pushed (private) to LeHak0/smolvla_chess_v1.
set -euo pipefail

LEROBOT_COMMIT=e595b7902714ba51f91e47523f66f89c5181b649    # same LeRobot as on the Mac
DATASET=${DATASET:-LeHak0/so101_chess_moves}       # variants: LeHak0/so101_chess_moves_coords, ..._marker
MODEL_REPO=${MODEL_REPO:-LeHak0/smolvla_chess_v1}
RUN=${RUN:-smolvla_chess_v1}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-64}                  # lower to 32 if the GPU has < 40 GB and you get out-of-memory
POLICY=${POLICY:-smolvla}           # smolvla (fine-tune lerobot/smolvla_base) | act (trained from scratch)
EXTRA=${EXTRA:-}                    # more lerobot-train arguments (held-out episodes, augmentation, unfrozen encoder...)
MIN_EPISODES=${MIN_EPISODES:-200}

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== system packages (ffmpeg for video decoding, tmux, git)"
SUDO=$(command -v sudo || true)      # containers run as root without sudo
if command -v apt-get >/dev/null; then $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq ffmpeg tmux git curl >/dev/null; fi

echo "== Python env (uv venv with Python 3.12)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~
[ -d lerobot ] || git clone -q https://github.com/huggingface/lerobot.git
cd lerobot && git fetch -q && git checkout -q $LEROBOT_COMMIT
[ -d ~/venv ] || uv venv -q --python 3.12 ~/venv
source ~/venv/bin/activate
uv pip install -q -e ".[smolvla,dataset,training]"   # training extra pins wandb
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== logins (skipped if already logged in)"
need=""
hf auth whoami >/dev/null 2>&1 || need="hf auth login"
python -c "import wandb, sys; sys.exit(0 if wandb.Api().api_key else 1)" >/dev/null 2>&1 || need="${need:+$need && }wandb login"
if [ -n "$need" ]; then
  if [ -t 0 ]; then eval "$need"; else echo "NEED_LOGIN: run on this machine:  source ~/venv/bin/activate && $need   then rerun this script"; exit 3; fi
fi

echo "== dataset check"
python - <<PY
from huggingface_hub import HfApi, hf_hub_download
import json
info = json.load(open(hf_hub_download("$DATASET", "meta/info.json", repo_type="dataset")))
print("dataset:", info["total_episodes"], "episodes,", info["total_frames"], "frames")
assert info["total_episodes"] >= $MIN_EPISODES, "the Hub has fewer than $MIN_EPISODES episodes: finish the recorder session on the Mac first"
PY

if [ "$POLICY" = act ]; then
  POLICY_ARGS="--policy.type=act"                     # ACT reads observation.images.wrist directly (no rename)
else
  # INIT: the weights to start from (default the SmolVLA base; a fine-tuned repo continues from its final checkpoint)
  POLICY_ARGS="--policy.path=${INIT:-lerobot/smolvla_base} --rename_map='{\"observation.images.wrist\": \"observation.images.camera1\"}'"
fi
CMD="source ~/venv/bin/activate && cd ~/lerobot && lerobot-train \
  $POLICY_ARGS \
  --dataset.repo_id=$DATASET \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --save_freq=5000 \
  --log_freq=100 \
  --output_dir=$HOME/outputs/train/$RUN \
  --job_name=$RUN \
  --policy.device=cuda \
  --policy.repo_id=$MODEL_REPO \
  --policy.private=true \
  --wandb.enable=true \
  --wandb.project=so101_chess $EXTRA 2>&1 | tee $HOME/train_$RUN.log"
tmux new-session -d -s train "$CMD; echo; echo 'TRAINING FINISHED (press enter)'; read"
echo
echo "== training started in tmux session 'train'"
echo "   live view:      tmux attach -t train     (detach: Ctrl-b then d)"
echo "   curves:         your W&B project 'so101_chess'"
echo "   run 1: 20k SmolVLA steps took 1 h 23 min on an H100; checkpoints every 5000 steps in ~/outputs/train/$RUN"
