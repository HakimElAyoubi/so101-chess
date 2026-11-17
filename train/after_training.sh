#!/usr/bin/env bash
# Run ON THE GPU MACHINE while training is running, then your laptop can go offline:
#     bash after_training.sh
# It starts a second tmux session ("after") that:
#   1. waits for the training session "train" to finish,
#   2. makes sure the model is on the Hub (private LeHak0/smolvla_chess_v1): uploads the final model if the
#      automatic push failed, plus every intermediate checkpoint (5k, 10k, 15k, 20k) and the training log,
#   3. stops the machine so it stops billing (Vast.ai: automatic; Lambda: can't self-terminate, see the log).
# Progress / result: ~/after_training.log   (check later with: tail ~/after_training.log)
set -euo pipefail

cat > ~/after_training_job.sh <<'JOB'
#!/usr/bin/env bash
set -uo pipefail
RUN=${RUN:-smolvla_chess_v1}
REPO=${MODEL_REPO:-LeHak0/smolvla_chess_v1}
OUT=$HOME/outputs/train/$RUN
source ~/venv/bin/activate
log() { echo "$(date '+%F %T') $*" | tee -a ~/after_training.log; }

log "waiting for the training session to finish"
# match the python entry point only (the tmux/bash wrappers also contain "lerobot-train" in their command line)
while pgrep -f "venv/bin/lerobot-train" >/dev/null; do sleep 60; done
log "training process ended"

LAST=$(ls -d $OUT/checkpoints/[0-9]* 2>/dev/null | sort | tail -1)
if [ -z "$LAST" ]; then log "ERROR: no checkpoint found in $OUT; NOT stopping the machine"; exit 1; fi
log "last checkpoint: $LAST"

# make sure the final model is on the Hub (lerobot-train pushes it at the end; redo it if that failed)
if ! python -c "from huggingface_hub import HfApi; import sys; sys.exit(0 if 'model.safetensors' in HfApi().list_repo_files('$REPO') else 1)" 2>/dev/null; then
  log "final model not on the Hub yet: uploading $LAST/pretrained_model"
  hf repo create $REPO --private -y 2>/dev/null || true
  hf upload $REPO $LAST/pretrained_model . --private || { log "ERROR: upload failed; NOT stopping the machine"; exit 1; }
fi
# every checkpoint (to compare 5k / 10k / 15k / 20k on the real arm) + the training log
for c in $OUT/checkpoints/[0-9]*; do
  n=$(basename $c); log "uploading checkpoint $n"
  hf upload $REPO $c/pretrained_model checkpoints/$n --private >/dev/null || log "WARNING: checkpoint $n upload failed"
done
[ -f ~/train_$RUN.log ] && hf upload $REPO ~/train_$RUN.log logs/train_$RUN.log --private >/dev/null
log "everything is on the Hub: https://huggingface.co/$REPO"

# SSH / tmux shells on Vast may not inherit the container's variables: read them from PID 1 if needed
for v in CONTAINER_ID CONTAINER_API_KEY; do
  [ -z "${!v:-}" ] && export $v="$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$v=//p")"
done
if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
  log "Vast.ai: stopping this instance (disk is kept; GPU billing stops)"
  # the uv venv has no pip, and a bare 'pip install vastai' failed silently on one box (26 Oct: the H100 kept billing):
  # install with uv into the venv and call it by its full path, retrying, and say so loudly if it still can't stop
  ~/.local/bin/uv pip install -q --python ~/venv/bin/python vastai >/dev/null 2>&1 || python -m pip install -q vastai >/dev/null 2>&1
  VAST=$( [ -x ~/venv/bin/vastai ] && echo ~/venv/bin/vastai || command -v vastai )
  for i in 1 2 3; do
    [ -n "$VAST" ] && "$VAST" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" 2>&1 | tee -a ~/after_training.log && break
    sleep 30
  done
  log "stop requested (if this machine is still running in a few minutes: ERROR, stop it from the Vast console)"
else
  log "Not a Vast.ai container: this script can't stop the machine itself."
  log "Lambda: terminate the instance in the Lambda Cloud dashboard (shutting it down from inside does NOT stop billing)."
fi
JOB
chmod +x ~/after_training_job.sh
# tell the user now whether the automatic stop will work on this machine
ID=${CONTAINER_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_ID=//p')}
KEY=${CONTAINER_API_KEY:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_API_KEY=//p')}
if [ -n "$ID" ] && [ -n "$KEY" ]; then echo "Vast.ai instance $ID detected: it will stop itself after the upload.";
else echo "WARNING: Vast.ai instance id / API key not found: the machine will NOT stop itself. Stop it in the Vast console after ~4 h."; fi
tmux kill-session -t after 2>/dev/null || true
tmux new-session -d -s after "RUN=${RUN:-smolvla_chess_v1} MODEL_REPO=${MODEL_REPO:-LeHak0/smolvla_chess_v1} bash ~/after_training_job.sh"
echo "after-training watcher started (tmux session 'after'). Log: ~/after_training.log"
echo "You can disconnect your laptop now."
