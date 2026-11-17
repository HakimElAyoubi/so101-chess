#!/bin/bash
# The staged player on the arm (web UI at http://localhost:8830). The project's skill model B:
#     scripts/play.sh --sync --model-repo LeHak0/smolvla_chess_skills_vision --checkpoint 020000
source "$(dirname "$0")/_common.sh"
"$PY" -u play/policy_player.py --variant skills "$@" 2> >(quiet)
