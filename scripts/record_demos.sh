#!/bin/bash
# Demonstrations: the follower is first aligned to the leader, then the web recorder plays a game with Stockfish and
# you demonstrate the black moves with the leader arm.   scripts/record_demos.sh --repo-id <you>/so101_chess_moves
source "$(dirname "$0")/_common.sh"
set -o pipefail
"$PY" -u calibration/align_to_leader.py 2>&1 | grep -v -E "objc\[|Warning|matmul" || exit 1
"$PY" -u recording/web_recorder.py "$@" 2> >(quiet)
