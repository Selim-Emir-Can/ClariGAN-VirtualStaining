#!/bin/bash
# Run folds 0..4 of one experiment on one GPU, starting a fold only if it can finish before DEADLINE.
#   GPUS=1 DEADLINE="2026-09-29 09:00" ./run_until.sh <run_kfold target> <extra args...>
# The per-fold estimate starts at EST_S seconds and is replaced by the measured duration of the last fold.
set -uo pipefail
cd /local/emir/ClariDi
DL=$(date -d "$DEADLINE" +%s); EST=${EST_S:-8000}
for k in ${FOLDS:-0 1 2 3 4}; do
  now=$(date +%s)
  if (( now + EST > DL )); then echo "$(date '+%F %T') STOP before fold $k: est ${EST}s would end after $DEADLINE"; break; fi
  echo "$(date '+%F %T') GPU $GPUS <- fold $k (est ${EST}s)"
  t0=$(date +%s)
  SCHEME=spatial ./run_kfold.sh "$@" --folds $k
  rc=$?; EST=$(( $(date +%s) - t0 ))
  echo "$(date '+%F %T') fold $k exit $rc after ${EST}s"
  (( rc == 0 )) || break
done
echo "$(date '+%F %T') RUN_UNTIL_DONE"
