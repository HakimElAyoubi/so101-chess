#!/usr/bin/env bash
# Plan step 3 (train/PLAN_TO_95.md): the three skill models on LeHak0/so101_chess_skills, one per rented GPU box.
#     bash launch_skills.sh A|B|C      (setup_and_train.sh and after_training.sh in the same folder)
#   A: SmolVLA, as run 1 (VLM frozen, only the action expert trains)       -> LeHak0/smolvla_chess_skills
#   B: SmolVLA with the vision encoder and VLM training too                -> LeHak0/smolvla_chess_skills_vision
#   C: ACT from scratch (ResNet-18 trained end to end; ignores the task text, the dot / ring tell it the stage)
#                                                                          -> LeHak0/act_chess_skills
# All three: the 40 validation clips held out (goal/skill_clips.json "val_episodes"), image augmentation on.
set -euo pipefail
cd "$(dirname "$0")"
VAL='[6,7,14,15,28,29,66,67,97,98,110,111,184,185,192,193,212,213,220,221,228,229,232,233,246,247,281,282,303,304,305,306,346,347,370,371,374,375,388,389]'
export DATASET=LeHak0/so101_chess_skills MIN_EPISODES=395
COMMON="'--dataset.exclude_episodes=$VAL' --dataset.image_transforms.enable=true"
# The augmentation runs on the CPU on 1280x720 frames: with LeRobot's default 4 data workers the GPUs waited 0.3-0.9 s per
# step for data (first launch, 26 Oct: ~1 step/s, ~$24 projected). More workers per box (they have 32-64 cores):
case "${1:-}" in
  A) export POLICY=smolvla RUN=smolvla_skills MODEL_REPO=LeHak0/smolvla_chess_skills STEPS=20000 BATCH=64
     export EXTRA="$COMMON --num_workers=24" ;;
  B) export POLICY=smolvla RUN=smolvla_skills_vision MODEL_REPO=LeHak0/smolvla_chess_skills_vision STEPS=20000 BATCH=64
     # 393M trainable instead of 100M: a lower learning rate for a whole pretrained VLM (Mac smoke test: grad norm 193 at
     # step 1 vs 42 for A at 1e-4)
     export EXTRA="$COMMON --policy.train_expert_only=false --policy.freeze_vision_encoder=false --policy.optimizer_lr=3e-5 --num_workers=32" ;;
  C) export POLICY=act RUN=act_skills MODEL_REPO=LeHak0/act_chess_skills STEPS=40000 BATCH=32
     export EXTRA="$COMMON --num_workers=16" ;;
  *) echo "usage: bash launch_skills.sh A|B|C"; exit 2 ;;
esac
bash setup_and_train.sh
bash after_training.sh
