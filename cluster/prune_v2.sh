#!/bin/bash
# Prune split-v2 runs as each one finishes: keep top_model_epoch_*.pth + config.yaml, delete optimizer states and
# the last/latest copies (13 GB -> 2.3 GB). A run is finished only when BOTH hold: its deliverables
# timing_fold_<k>.csv exists, AND no run_kfold.sh for that tag + fold is alive.
# Exits once V2_ALL_DONE is logged and nothing is left to prune. Log: logs_prune_v2.log. Stop: touch prune_stop
ROOT=/local/emir/ClariDi; cd $ROOT
declare -A EXP=([sp2_primary]=sp_primary [sp2_stock]=sp_stock_vqgan [sp2_refB]=sp_refB_unstained
                [sp2_speccond]=sp_specimen_cond [sp2_refA]=sp_refA_stained)
log() { echo "$(date '+%F %T') $*"; }
log "armed"
while true; do
  [ -f prune_stop ] && { log "stop requested"; exit 0; }
  left=0
  for CK in results_spatial_v2/*/*/checkpoint; do
    [ -d "$CK" ] || continue
    ls $CK/last_model.pth $CK/latest_model_*.pth $CK/*optim_sche*.pth >/dev/null 2>&1 || continue
    left=1
    run=$(basename $(dirname $(dirname $CK)))                  # ClariGAN_<...>_fold_<k>_<tag>
    rest=${run#*_fold_}; k=${rest%%_*}; tag=${rest#*_}
    [ -f deliverables_spatial_v2/${EXP[$tag]}/timing_fold_${k}.csv ] || continue
    pgrep -f "run_kfold\.sh .*--tag $tag .*--folds $k\$" >/dev/null && continue
    top=$(ls $CK/top_model_epoch_*.pth 2>/dev/null | tail -n1)
    [ -n "$top" ] || { log "$run: no top_model, skipped"; continue; }
    before=$(du -sm $CK | cut -f1)
    for f in $CK/*.pth; do [ "$f" = "$top" ] || rm -f "$f"; done
    log "$run pruned: ${before} MB -> $(du -sm $CK | cut -f1) MB (kept $(basename $top))"
  done
  if [ $left = 0 ] && grep -q V2_ALL_DONE logs_v2_launcher.log; then log "all v2 runs pruned; exiting"; exit 0; fi
  sleep 120
done
