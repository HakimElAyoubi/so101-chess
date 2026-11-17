#!/bin/bash
# Guided game on the arm (play/guided_game.py): a fixed, pre-written line of moves, human White and robot Black.
# Robot moves replay a demonstration recorded with the leader when one exists for that move. Pass the grasp skill
# with --grasp-repo, e.g. --grasp-repo "astra:gpt-6.1-terra" (needs OPENAI_API_KEY).
source "$(dirname "$0")/_common.sh"
"$PY" -u play/policy_player.py --scripted-game "$@" 2> >(quiet)
