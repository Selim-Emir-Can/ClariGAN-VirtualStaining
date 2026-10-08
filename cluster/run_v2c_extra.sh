#!/bin/bash
# GPUs 8 and 9 until 10:00 Oct 8 (user): once a GPU leaves the diffusion queue (run_v2_queue.sh, 09:00 limit), take
# jobs from v2b_queue.txt that can finish before LIMIT, longest that fits first (conservative per-run estimates from
# the LOSO campaign). A failed job stops the worker (v2b_failed.txt).
#   setsid nohup ./run_v2c_extra.sh >> logs_v2b_launcher.log 2>&1 &
cd /local/emir/ClariDi
LIMIT="2026-10-08 10:00"; GPU_LIST="8 9"
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
busy() { [ -s v2_queue.txt ] || pgrep -u emir -f "(--gpu_ids|--gpu) $1( |\$)" >/dev/null; }
pop_fit() {   # prints and removes the longest job that ends before LIMIT
  local left=$(( $(date -d "$LIMIT" +%s) - $(date +%s) ))
  flock $Q.lock python3 - "$Q" "$left" <<'PY'
import sys
q, left = sys.argv[1], int(sys.argv[2])
EST = {"pixel": 9.5, "cwgan": 8.0, "encoder": 4.3, "pix2pix": 3.8}     # hours, conservative
lines = [l for l in open(q).read().splitlines() if l.strip()]
fit = [(EST[l.split()[0]], i) for i, l in enumerate(lines) if EST[l.split()[0]] * 3600 <= left]
if fit:
    _, i = max(fit, key=lambda t: (t[0], -t[1]))
    print(lines[i]); del lines[i]
    open(q, "w").write("".join(l + "\n" for l in lines))
PY
}
worker() {
  local g=$1
  while busy $g || { sleep 60; busy $g; }; do sleep 60; done
  while true; do
    job=$(pop_fit); [ -n "$job" ] || { echo "$(date '+%F %T') GPU $g released (nothing fits before $LIMIT)"; break; }
    read -r m k <<< "$job"
    echo "$(date '+%F %T') GPU $g <- $m fold $k"
    n0=$(wc -l < logs_v2b_gpu$g.log)
    FOLDS="$k" GPUS=$g DEADLINE="2099-01-01 00:00" ./run_until.sh ${ARGS[$m]}
    if tail -n +$n0 logs_v2b_gpu$g.log | grep -qE "fold $k exit [1-9]"; then
      echo "$m $k" >> v2b_failed.txt; echo "$(date '+%F %T') GPU $g FAILED $m fold $k; worker stopped"; break; fi
  done
}
for g in $GPU_LIST; do touch logs_v2b_gpu$g.log; worker $g >> logs_v2b_gpu$g.log 2>&1 & done
wait; echo "$(date '+%F %T') V2C_EXTRA_DONE"
