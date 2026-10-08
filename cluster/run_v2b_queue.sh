#!/bin/bash
# Split v2, remaining methods (user, Oct 8): pixel-space BBDM, cWGAN, trainable encoder, pix2pix; 5 folds each,
# longest first. One worker per GPU in GPU_LIST; a worker starts only once its GPU is free of v2 diffusion jobs
# (v2_queue.txt empty and no ClariDi job on that GPU for two checks 60 s apart). Queue: v2b_queue.txt (flock).
#   GPU_LIST="1 3 4 6 7" setsid nohup ./run_v2b_queue.sh >> logs_v2b_launcher.log 2>&1 &
cd /local/emir/ClariDi
SPLIT=$PWD/data/splits/model_design_exp_split_v2.csv
S="--split_file $SPLIT --results_root $PWD/results_spatial_v2 --deliverables_root $PWD/deliverables_spatial_v2"
G_="--split_file $SPLIT --deliverables_root $PWD/deliverables_spatial_v2"
declare -A ARGS=(
  [pixel]="pixel --tag sp2_pixel --experiment pixel_space $S"
  [encoder]="encoder --tag sp2_encoder --experiment trainable_encoder $S"
  [cwgan]="cwgan --tag sp2_cwgan --experiment cwgan --out_root $PWD/baselines_out_v2/cwgan $G_"
  [pix2pix]="pix2pix --tag sp2_pix2pix --experiment pix2pix --out_root $PWD/baselines_out_v2/pix2pix $G_"
)
Q=$PWD/v2b_queue.txt
[ -s $Q ] || for m in pixel cwgan encoder pix2pix; do for k in 0 1 2 3 4; do echo "$m $k"; done; done > $Q
busy() { [ -s v2_queue.txt ] || pgrep -u emir -f "(--gpu_ids|--gpu) $1( |\$)" >/dev/null; }
pop() { flock $Q.lock bash -c "head -n1 $Q; sed -i 1d $Q"; }
worker() {
  local g=$1
  while busy $g || { sleep 60; busy $g; }; do sleep 60; done
  while true; do
    job=$(pop); [ -n "$job" ] || break
    read -r m k <<< "$job"
    echo "$(date '+%F %T') GPU $g <- $m fold $k"
    n0=$(wc -l < logs_v2b_gpu$g.log)
    FOLDS="$k" GPUS=$g DEADLINE="2099-01-01 00:00" ./run_until.sh ${ARGS[$m]}
    if tail -n +$n0 logs_v2b_gpu$g.log | grep -qE "fold $k exit [1-9]"; then   # failed: stop this GPU, keep the rest queued
      echo "$m $k" >> v2b_failed.txt; echo "$(date '+%F %T') GPU $g FAILED $m fold $k; worker stopped"; break; fi
  done
  echo "$(date '+%F %T') GPU $g QUEUE_DONE"
}
for g in ${GPU_LIST:?}; do worker $g >> logs_v2b_gpu$g.log 2>&1 & done
wait; echo "$(date '+%F %T') V2B_ALL_DONE"
