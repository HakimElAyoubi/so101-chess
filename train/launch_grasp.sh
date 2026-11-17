#!/usr/bin/env bash
# Grasp-only skill models on LeHak0/so101_chess_grasp_skill (goal/build_grasp_dataset.py: the 196 grasp clips of the
# game demos + the grasp drill), one per rented GPU box. The place skill stays with the step-3 model.
#     bash launch_grasp.sh G1|G2      (setup_and_train.sh, after_training.sh and grasp_val.txt in the same folder)
#   G1: continue skill model B (SmolVLA, whole model) on grasps only, 10k steps     -> LeHak0/smolvla_chess_grasp_b
#   G2: the same recipe from the SmolVLA base, 20k steps                             -> LeHak0/smolvla_chess_grasp
# Both: the held-out clips of grasp_val.txt excluded, image augmentation on, B's learning rate for a whole VLM.
set -euo pipefail
cd "$(dirname "$0")"
VAL=$(tr -d ' \n' < grasp_val.txt)
export DATASET=LeHak0/so101_chess_grasp_skill MIN_EPISODES=${MIN_EPISODES:-400} POLICY=smolvla BATCH=64
EXTRA="'--dataset.exclude_episodes=$VAL' --dataset.image_transforms.enable=true --policy.train_expert_only=false"
EXTRA="$EXTRA --policy.freeze_vision_encoder=false --policy.optimizer_lr=3e-5 --num_workers=32"
export EXTRA
case "${1:-}" in
  G1) export RUN=smolvla_grasp_b MODEL_REPO=LeHak0/smolvla_chess_grasp_b STEPS=10000 INIT=LeHak0/smolvla_chess_skills_vision ;;
  G2) export RUN=smolvla_grasp MODEL_REPO=LeHak0/smolvla_chess_grasp STEPS=20000 ;;
  *) echo "usage: bash launch_grasp.sh G1|G2"; exit 2 ;;
esac
bash setup_and_train.sh
bash after_training.sh
