#!/bin/bash
# Prune checkpoint dirs of folds whose evaluation has ALREADY completed (their sample
# dir under k-fold_samples/ exists and holds outputs). Keeps top_model_epoch_*.pth and
# config.yaml, deletes optimizer states and the rotating last/latest copies.
# Safe to run while other folds are still training: only touches finished folds.
#   ./prune_finished_folds.sh          # dry run, lists what it would delete
#   ./prune_finished_folds.sh --yes    # delete
ROOT=/local/emir/ClariDi
total=0
for ck in $ROOT/results/*/*/checkpoint; do
  run=$(basename $(dirname $(dirname $ck)))        # e.g. ClariGAN_stratified_fold_0_specimen_grouped
  save=${run#*_fold_}; save="fold_$save"            # e.g. fold_0_specimen_grouped
  samp=$ROOT/k-fold_samples/$save
  [ -d "$samp" ] && [ -n "$(ls -A $samp 2>/dev/null)" ] || continue
  top=$(ls $ck/top_model_epoch_*.pth 2>/dev/null | tail -n1)
  [ -n "$top" ] || { echo "SKIP $run: no top_model checkpoint"; continue; }
  for f in $ck/*.pth; do
    [ "$f" = "$top" ] && continue
    sz=$(stat -c %s "$f"); total=$((total+sz))
    if [ "$1" = "--yes" ]; then rm -f "$f"; echo "deleted $f"; else echo "would delete $f"; fi
  done
done
echo "$( [ "$1" = "--yes" ] && echo freed || echo would free ) $((total/1000000000)) GB"
