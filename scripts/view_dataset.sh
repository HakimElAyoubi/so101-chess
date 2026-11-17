#!/bin/bash
# Browse one episode of a recorded dataset in Rerun.   scripts/view_dataset.sh <you>/so101_chess_moves <episode>
source "$(dirname "$0")/_common.sh"
[ $# -lt 2 ] && { echo "usage: scripts/view_dataset.sh <repo_id> <episode_index>"; exit 2; }
lerobot-dataset-viz --repo-id "$1" --episode-index "$2" 2>&1 | grep -v -E "objc\[|Warning"
