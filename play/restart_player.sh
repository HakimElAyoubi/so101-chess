#!/bin/bash
# Restart the policy player from a script (the automatic calibration, a crashed page...): the running player is stopped
# (the arm keeps holding its pose) and the launcher is opened THROUGH Terminal.app, which holds the macOS camera
# permission; a player started from any other process cannot open the wrist camera.
#     bash play/restart_player.sh [scripts/play.sh]
cd "$(dirname "$0")/.."
LAUNCHER=${1:-scripts/play.sh}
pkill -f "policy_player.py" && sleep 1.5
open -a Terminal "$PWD/$LAUNCHER"
for i in $(seq 1 150); do
  if curl -s -m 1 http://127.0.0.1:8830/raw 2>/dev/null | grep -q '"jpeg_b64": "/9j'; then echo "player up with the camera ($i s)"; exit 0; fi
  sleep 1
done
echo "the player did not come up with a camera frame within 150 s"; exit 1
