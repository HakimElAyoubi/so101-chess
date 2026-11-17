#!/bin/bash
# Staged player with Claude as the grasp and place skill (play/llm_skill.py, play/CLAUDE_SKILL.md).
# Needs ANTHROPIC_API_KEY, or the file ~/.anthropic_api_key (one line, chmod 600).
source "$(dirname "$0")/_common.sh"
MODEL=${CLAUDE_MODEL:-claude-opus-5-5}
export CLAUDE_EFFORT="${CLAUDE_EFFORT:-low}"
if [ -z "$ANTHROPIC_API_KEY" ] && [ -r "$HOME/.anthropic_api_key" ]; then export ANTHROPIC_API_KEY=$(tr -d '[:space:]' < "$HOME/.anthropic_api_key"); fi
[ -z "$ANTHROPIC_API_KEY" ] && { echo "No ANTHROPIC_API_KEY and no ~/.anthropic_api_key."; exit 1; }
"$PY" -u play/policy_player.py --variant skills --sync --model-repo "claude:$MODEL" --checkpoint final --grasp-repo "claude:$MODEL" --grasp-checkpoint final "$@" 2> >(quiet)
