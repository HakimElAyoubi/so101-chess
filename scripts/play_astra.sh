#!/bin/bash
# Staged player with a GPT vision model for the grasp and the calibrated scripted motion for the place
# (play/astra_skill.py, play/ASTRA_SKILL.md). Needs OPENAI_API_KEY, or ~/.openai_api_key; ASTRA_MODEL=mock needs none.
source "$(dirname "$0")/_common.sh"
MODEL=${ASTRA_MODEL:-gpt-6.1-terra}
export ASTRA_EFFORT="${ASTRA_EFFORT:-low}" ASTRA_SERVICE_TIER="${ASTRA_SERVICE_TIER:-default}"
if [ -z "$OPENAI_API_KEY" ] && [ -r "$HOME/.openai_api_key" ]; then export OPENAI_API_KEY=$(tr -d '[:space:]' < "$HOME/.openai_api_key"); fi
if [ "$MODEL" != "mock" ] && [ -z "$OPENAI_API_KEY" ]; then echo "No OPENAI_API_KEY and no ~/.openai_api_key."; exit 1; fi
"$PY" -u play/policy_player.py --variant skills --sync --model-repo "scripted:place" --checkpoint final --grasp-repo "astra:$MODEL" --grasp-checkpoint final "$@" 2> >(quiet)
