#!/bin/bash
# Shared job queue for the remaining split-v2 runs (replaces run_v2.sh's fixed per-GPU lists, Oct 7 23:5x).
# Each GPU worker pops the next job from v2_queue.txt (flock). A worker given WAIT_PID first waits for that
# job (the fold already running on its GPU). Workers in LIMITED_GPUS start a job only if it can end before
# LIMIT_DEADLINE (user: GPUs 8, 9 until 09:00 Oct 8).
#   setsid nohup ./run_v2_queue.sh > logs_v2_queue.log 2>&1 &
cd /local/emir/ClariDi
S="--split_file $PWD/data/splits/model_design_exp_split_v2.csv --results_root $PWD/results_spatial_v2 --deliverables_root $PWD/deliverables_spatial_v2"
C=configs
declare -A ARGS=(
  [stock]="stock --tag sp2_stock --experiment sp_stock_vqgan"
  [refB]="claridi --tag sp2_refB --experiment sp_refB_unstained --config $C/Template-LBBDM-f16_refB_unstained.yaml"
  [speccond]="claridi --tag sp2_speccond --experiment sp_specimen_cond --config $C/Template-LBBDM-f16_specimen_cond.yaml"
  [refA]="claridi --tag sp2_refA --experiment sp_refA_stained --config $C/Template-LBBDM-f16_refA_stained.yaml"
)
Q=$PWD/v2_queue.txt
[ -s $Q ] || for m in stock refB speccond refA; do for k in 0 1 2 3 4; do echo "$m $k"; done; done > $Q
LIMITED_GPUS=" 8 9 "; LIMIT_DEADLINE="2026-10-08 09:00"; EST=7000

pop() { flock $Q.lock bash -c "head -n1 $Q; sed -i 1d $Q"; }
worker() {   # gpu [wait_pid]
  local g=$1 wp=$2
  if [ -n "$wp" ]; then while kill -0 $wp 2>/dev/null; do sleep 60; done; fi
  while true; do
    if [[ "$LIMITED_GPUS" == *" $g "* ]] && (( $(date +%s) + EST > $(date -d "$LIMIT_DEADLINE" +%s) )); then
      echo "$(date '+%F %T') GPU $g released (next job would end after $LIMIT_DEADLINE)"; break; fi
    job=$(pop); [ -n "$job" ] || break
    read -r m k <<< "$job"
    echo "$(date '+%F %T') GPU $g <- $m fold $k"
    FOLDS="$k" GPUS=$g DEADLINE="2099-01-01 00:00" ./run_until.sh ${ARGS[$m]} $S
  done
  echo "$(date '+%F %T') GPU $g QUEUE_DONE"
}
for spec in ${WORKERS:?"WORKERS=\"1:pid 3:pid ... 8 9\""}; do
  g=${spec%%:*}; wp=""; [[ $spec == *:* ]] && wp=${spec#*:}
  worker $g $wp >> logs_v2_gpu$g.log 2>&1 &
done
wait; echo "$(date '+%F %T') V2_ALL_DONE"
