#!/bin/bash
# Prune every split-v2 run once finished (supersedes prune_v2.sh). A run is finished when its deliverables
# timing_fold_<k>.csv exists AND no run_kfold.sh for that tag + fold is alive.
#  diffusion (results_spatial_v2/*/*/checkpoint): keep top_model_epoch_*.pth + config.yaml
#  GANs (baselines_out_v2/<b>/checkpoints/<name>): keep latest_net_G.pth (the only weights test.py loads) + logs
# Exits after V2_ALL_DONE and V2B_ALL_DONE with nothing left. Log: logs_prune_v2.log. Stop: touch prune_v2b_stop
ROOT=/local/emir/ClariDi; cd $ROOT
declare -A EXP=([sp2_primary]=sp_primary [sp2_stock]=sp_stock_vqgan [sp2_refB]=sp_refB_unstained
                [sp2_speccond]=sp_specimen_cond [sp2_refA]=sp_refA_stained [sp2_pixel]=pixel_space
                [sp2_encoder]=trainable_encoder [sp2_cwgan]=cwgan [sp2_pix2pix]=pix2pix)
log() { echo "$(date '+%F %T') $*"; }
done_run() {  # tag fold
  [ -n "${EXP[$1]}" ] && [ -f deliverables_spatial_v2/${EXP[$1]}/timing_fold_$2.csv ] &&
    ! pgrep -u emir -f "run_kfold\.sh .*--tag $1 .*--folds $2\$" >/dev/null
}
log "armed"
while true; do
  [ -f prune_v2b_stop ] && { log "stop requested"; exit 0; }
  left=0
  for CK in results_spatial_v2/*/*/checkpoint; do
    [ -d "$CK" ] || continue
    ls $CK/last_model.pth $CK/latest_model_*.pth $CK/*optim_sche*.pth >/dev/null 2>&1 || continue
    left=1; run=$(basename $(dirname $(dirname $CK))); rest=${run#*_fold_}; k=${rest%%_*}; tag=${rest#*_}
    done_run $tag $k || continue
    top=$(ls $CK/top_model_epoch_*.pth 2>/dev/null | tail -n1); [ -n "$top" ] || { log "$run: no top_model"; continue; }
    b=$(du -sm $CK | cut -f1); for f in $CK/*.pth; do [ "$f" = "$top" ] || rm -f "$f"; done
    log "$run pruned: $b MB -> $(du -sm $CK | cut -f1) MB (kept $(basename $top))"
  done
  for CK in baselines_out_v2/*/checkpoints/*/; do
    [ -d "$CK" ] || continue
    n=$(ls $CK/*.pth 2>/dev/null | grep -v '/latest_net_G.pth$' | wc -l); [ $n -gt 0 ] || continue
    left=1; run=$(basename $CK); rest=${run#*_fold_}; k=${rest%%_*}; tag=${rest#*_}
    done_run $tag $k || continue
    [ -f $CK/latest_net_G.pth ] || { log "$run: no latest_net_G"; continue; }
    b=$(du -sm $CK | cut -f1); ls $CK/*.pth | grep -v '/latest_net_G.pth$' | xargs rm -f
    log "$run pruned: $b MB -> $(du -sm $CK | cut -f1) MB (kept latest_net_G.pth)"
  done
  if [ $left = 0 ] && grep -q V2_ALL_DONE logs_v2_launcher.log && grep -q V2B_ALL_DONE logs_v2b_launcher.log 2>/dev/null; then
    log "all v2 runs pruned; exiting"; exit 0; fi
  sleep 120
done
