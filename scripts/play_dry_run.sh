#!/bin/bash
# The same player with no hardware: the follower is simulated on the twin and a scripted skill stands in for the model.
source "$(dirname "$0")/_common.sh"
"$PY" -u play/policy_player.py --variant skills --fake --standin scripted --port 8840 "$@" 2> >(quiet)
