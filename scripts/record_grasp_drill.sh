#!/bin/bash
# Grasp-only drill (recording/GRASP_DRILL.md).   scripts/record_grasp_drill.sh --repo-id <you>/so101_chess_grasp
source "$(dirname "$0")/_common.sh"
set -o pipefail
"$PY" -u calibration/align_to_leader.py 2>&1 | grep -v -E "objc\[|Warning|matmul" || exit 1
"$PY" -u recording/grasp_recorder.py "$@" 2> >(quiet)
