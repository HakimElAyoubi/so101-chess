#!/bin/bash
# The MuJoCo twin plays a game against Stockfish (--auto: Stockfish plays both sides; --headless: no window).
# macOS needs mjpython for the viewer window.
source "$(dirname "$0")/_common.sh"
if [[ "$OSTYPE" == darwin* ]] && command -v mjpython >/dev/null && [[ " $* " != *" --headless "* ]]; then PY=mjpython; fi
exec "$PY" -u chess_sim/play_game.py "$@"
