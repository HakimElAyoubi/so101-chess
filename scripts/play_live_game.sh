#!/bin/bash
# Human White versus live Stockfish. Pick trained SmolVLA models on the page.
# For a hardware-free UI/control-flow check: scripts/play_live_game.sh --fake --standin scripted
source "$(dirname "$0")/_common.sh"
"$PY" -u play/policy_player.py --live-game "$@" 2> >(quiet)
