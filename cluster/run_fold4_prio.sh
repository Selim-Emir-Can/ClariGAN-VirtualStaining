#!/bin/bash
# Fold 4 re-prioritised (user, Oct 4 19:20): A, B and ours (primary) first.
# GPU2: refA -> stock (original run_fold4.sh subshell, still running) | GPU3: refB -> specimen_cond | GPU6: primary.
# specimen_cond was aborted 3 min in on GPU6; its partial dir was moved to results_spatial/ABORTED_*.
cd /local/emir/ClariDi
export DEADLINE=${DEADLINE:-"2099-01-01 00:00"} FOLDS="4" EST_S=7200
S="--split_file $PWD/data/splits/model_design_exp_split.csv --results_root $PWD/results_spatial --deliverables_root $PWD/deliverables_spatial"
C=configs
run(){ local gpu=$1 name=$2; shift 2; GPUS=$gpu ./run_until.sh "$@" $S > logs_sp_${name}_f4.log 2>&1; }
REFB_PID=${REFB_PID:-4126294}
( run 6 primary  claridi --tag sp_primary  --experiment sp_primary ) &
( while kill -0 $REFB_PID 2>/dev/null; do sleep 60; done
  run 3 speccond claridi --tag sp_speccond --experiment sp_specimen_cond  --config $C/Template-LBBDM-f16_specimen_cond.yaml ) &
wait; echo "$(date '+%F %T') FOLD4_PRIO_DONE"
