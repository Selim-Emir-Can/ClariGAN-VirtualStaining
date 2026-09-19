#!/bin/bash
# Minimal GPU job queue. Each line of queue.txt is "<run_kfold target> <extra args>" and is
# launched as  GPUS=<g> ./run_kfold.sh <line>  on the first GPU in $GPUSET with no compute
# process owned by $USER. Polls every 60 s. Lines are consumed top-down; done lines move to
# queue_done.txt. Stop with: touch /local/emir/ClariDi/queue_stop
#   GPUSET="1 2 3 4 5 6 7 8 9" nohup ./queue_runner.sh > logs/queue_runner.log 2>&1 &
ROOT=/local/emir/ClariDi
GPUSET=${GPUSET:-"1 2 3 4 5 6 7 8 9"}
cd $ROOT
busy_gpus() {   # indices of GPUs with a compute process owned by us
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | while IFS=, read -r uuid pid; do
    pid=$(echo $pid | tr -d ' '); owner=$(ps -o user= -p $pid 2>/dev/null)
    [ "$owner" = "$USER" ] && nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$uuid" | cut -d, -f1
  done | sort -u
}
while true; do
  [ -f queue_stop ] && { echo "$(date '+%F %T') stop requested"; exit 0; }
  [ -s queue.txt ] || { sleep 60; continue; }
  busy=$(busy_gpus)
  for g in $GPUSET; do
    echo "$busy" | grep -qx "$g" && continue
    line=$(grep -vE '^\s*(#|$)' queue.txt | head -n 1); [ -n "$line" ] || break
    # remove that first non-comment line from the queue
    python3 - "$line" <<'PY'
import sys; L=open("queue.txt").read().split("\n"); t=sys.argv[1]
i=next(k for k,x in enumerate(L) if x.strip()==t.strip()); del L[i]; open("queue.txt","w").write("\n".join(L))
PY
    tag=$(echo "$line" | tr -c 'A-Za-z0-9_' '_' | sed 's/_*$//')
    echo "$(date '+%F %T') GPU $g <- $line" | tee -a queue_done.txt
    GPUS=$g nohup ./run_kfold.sh $line > logs/q_${tag}.log 2>&1 &
    sleep 20   # let it claim the GPU before the next poll
    busy=$(busy_gpus)
  done
  sleep 60
done
