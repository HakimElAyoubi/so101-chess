#!/bin/zsh
# Bring a freshly rented Vast.ai box up for one grasp run: copy the train files, start the launcher inside tmux
# ("setup" session; it asks for the Hugging Face and W&B logins there), print the command to attach to it.
#     bash train/box_up.sh <ssh host> <ssh port> G1|G2
set -e
HOST=$1; PORT=$2; RUN=$3; KEY=~/.ssh/vast_softgroup
cd "$(dirname "$0")"
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p $PORT root@$HOST"
echo "== waiting for sshd on $HOST:$PORT"
for i in $(seq 1 40); do $SSH -o BatchMode=yes true 2>/dev/null && break; sleep 10; done
$SSH -o BatchMode=yes "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; df -h / | tail -1"
echo "== copying launch_grasp.sh setup_and_train.sh after_training.sh grasp_val.txt"
scp -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -P $PORT launch_grasp.sh setup_and_train.sh after_training.sh grasp_val.txt root@$HOST:~/
$SSH "command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null); tmux has-session -t setup 2>/dev/null && echo 'setup session already running' || tmux new -d -s setup \"bash launch_grasp.sh $RUN 2>&1 | tee ~/launch_$RUN.log\""
echo "== started: $RUN on $HOST:$PORT. Attach for the two logins with:"
echo "ssh -i $KEY -o IdentitiesOnly=yes -p $PORT root@$HOST -t tmux attach -t setup"
