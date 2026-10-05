#!/bin/bash
# Fold 4 of the spatial split for the 5 model-design experiments, on GPUs 2, 3, 6 (user, Oct 4).
# Each GPU runs its experiments one after another (~2 h each incl. eval). Logs: logs_sp_<name>_f4.log
#   DEADLINE="2026-10-05 12:00" ./run_fold4.sh      (no fold starts unless it can end before DEADLINE)
cd /local/emir/ClariDi
export DEADLINE=${DEADLINE:-"2099-01-01 00:00"} FOLDS="4" EST_S=7200
S="--split_file $PWD/data/splits/model_design_exp_split.csv --results_root $PWD/results_spatial --deliverables_root $PWD/deliverables_spatial"
C=configs
run(){ local gpu=$1 name=$2; shift 2; GPUS=$gpu ./run_until.sh "$@" $S > logs_sp_${name}_f4.log 2>&1; }
( run 2 refA     claridi --tag sp_refA     --experiment sp_refA_stained   --config $C/Template-LBBDM-f16_refA_stained.yaml
  run 2 stock    stock   --tag sp_stock    --experiment sp_stock_vqgan ) &
( run 3 refB     claridi --tag sp_refB     --experiment sp_refB_unstained --config $C/Template-LBBDM-f16_refB_unstained.yaml
  run 3 primary  claridi --tag sp_primary  --experiment sp_primary ) &
( run 6 speccond claridi --tag sp_speccond --experiment sp_specimen_cond  --config $C/Template-LBBDM-f16_specimen_cond.yaml ) &
wait; echo "$(date '+%F %T') FOLD4_ALL_DONE"
