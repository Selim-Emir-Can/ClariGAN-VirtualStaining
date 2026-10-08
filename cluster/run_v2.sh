#!/bin/bash
# Spatial split v2 (data/splits/model_design_exp_split_v2.csv): 5 models x 5 folds = 25 runs (~1.9 h each).
# Jobs go round-robin over the GPUs in priority order (Ours, Vanilla, B, label, A; fold-major within a model);
# each GPU runs its queue one job at a time. Results: results_spatial_v2/, deliverables_spatial_v2/<experiment>/.
# Logs: logs_v2_gpu<id>.log.   GPU_LIST="1 3 4" ./run_v2.sh     (nohup/setsid it; never launch without the user's GPUs)
cd /local/emir/ClariDi
read -ra G <<< "${GPU_LIST:?set GPU_LIST, e.g. GPU_LIST=\"1 3 4\"}"
export DEADLINE=${DEADLINE:-"2099-01-01 00:00"} EST_S=7200
S="--split_file $PWD/data/splits/model_design_exp_split_v2.csv --results_root $PWD/results_spatial_v2 --deliverables_root $PWD/deliverables_spatial_v2"
C=configs
declare -A ARGS=(
  [primary]="claridi --tag sp2_primary --experiment sp_primary"
  [stock]="stock --tag sp2_stock --experiment sp_stock_vqgan"
  [refB]="claridi --tag sp2_refB --experiment sp_refB_unstained --config $C/Template-LBBDM-f16_refB_unstained.yaml"
  [speccond]="claridi --tag sp2_speccond --experiment sp_specimen_cond --config $C/Template-LBBDM-f16_specimen_cond.yaml"
  [refA]="claridi --tag sp2_refA --experiment sp_refA_stained --config $C/Template-LBBDM-f16_refA_stained.yaml"
)
JOBS=(); for m in primary stock refB speccond refA; do for k in 0 1 2 3 4; do JOBS+=("$m $k"); done; done
mkdir -p results_spatial_v2 deliverables_spatial_v2
for i in "${!G[@]}"; do
  (
    for ((j = i; j < ${#JOBS[@]}; j += ${#G[@]})); do
      read -r m k <<< "${JOBS[j]}"
      echo "$(date '+%F %T') GPU ${G[i]} <- $m fold $k"
      FOLDS="$k" GPUS=${G[i]} ./run_until.sh ${ARGS[$m]} $S
    done
    echo "$(date '+%F %T') GPU ${G[i]} QUEUE_DONE"
  ) > logs_v2_gpu${G[i]}.log 2>&1 &
done
wait; echo "$(date '+%F %T') V2_ALL_DONE"
