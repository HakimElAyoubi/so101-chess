#!/bin/zsh
# Live view of chess_sim/data/progress.txt: refreshes every second, closes 5 s after the job ends.
F="$(dirname "$0")/data/progress.txt"
while true; do
  clear; echo "chess_sim job progress  ($(date +%H:%M:%S))"; echo; cat "$F" 2>/dev/null
  grep -q -E "DONE|FAILED" "$F" 2>/dev/null && { sleep 5; break; }
  sleep 1
done
