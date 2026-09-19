#!/bin/bash
# ClariDi specimen-grouped cross-validation.
#   ./run_kfold.sh dry                 -> print fold table + leakage check only
#   GPUS=0,1,2 ./run_kfold.sh claridi  -> primary ClariDi (LBBDM-f16, finetuned VQGAN)
#   GPUS=0,1,2 ./run_kfold.sh pixel    -> pixel-space BBDM ablation
#   GPUS=0,1,2 ./run_kfold.sh encoder  -> trainable VQGAN encoder ablation (fine-tuned init, as in the paper)
#   GPUS=0,1,2 ./run_kfold.sh stock    -> leak-free stock-ImageNet-VQGAN ablation
#   GPUS=4 ./run_kfold.sh pix2pix        -> pix2pix baseline on the same grouped folds (single GPU)
#   GPUS=4 ./run_kfold.sh cwgan          -> cWGAN baseline on the same grouped folds (single GPU)
set -euo pipefail
ROOT=/local/emir/ClariDi
DATA=$ROOT/data/bbdm256          # pre-resized, bit-identical to data/bbdm, 12x faster to load
VQGAN=${VQGAN:-$ROOT/weights/epoch=000022.ckpt}
STOCK_VQGAN=${STOCK_VQGAN:-$ROOT/weights/vqgan_imagenet_f16_16384_stock.ckpt}
GPUS=${GPUS:-0}
export CLARIDI_NUM_WORKERS=${CLARIDI_NUM_WORKERS:-4}   # 256px PNGs decode fast; 8/job oversubscribed the box
export TORCH_HOME=$ROOT/.cache/torch   # keep LPIPS/VGG weights off the full home disk
SCHEME=${SCHEME:-loso}
source /home/emir/miniconda3/etc/profile.d/conda.sh
conda activate chatgarment
cd $ROOT/repo/BBDM

common="--data_root $DATA --scheme $SCHEME --gpu_ids $GPUS \
        --results_root $ROOT/results --samples_root $ROOT/k-fold_samples"

TARGET="${1:-dry}"; shift || true   # remaining args pass through to kfold_grouped.py
case "$TARGET" in
  dry)
    python kfold_grouped.py $common --dry_run "$@" ;;
  claridi)
    python kfold_grouped.py $common --tag specimen_grouped --experiment claridi_primary \
      --config configs/Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml --vqgan_ckpt "$VQGAN" "$@" ;;
  pixel)
    python kfold_grouped.py $common --tag specimen_grouped_pixel --experiment pixel_space \
      --config configs/Template-BBDM_pixel_256_matchUNet.yaml "$@" ;;
  encoder)
    python kfold_grouped.py $common --tag specimen_grouped_trainable_encoder --experiment trainable_encoder \
      --config configs/Template-LBBDM-f16_imagenetVQGAN_finetuned_trainable_encoder.yaml --vqgan_ckpt "$VQGAN" "$@" ;;
  stock)
    [ -f "$STOCK_VQGAN" ] || { echo "missing stock VQGAN: $STOCK_VQGAN"; exit 1; }
    python kfold_grouped.py $common --tag stockVQGAN_leakfree --experiment claridi_stock_vqgan \
      --config configs/Template-LBBDM-f16_stockVQGAN_leakfree.yaml --vqgan_ckpt "$STOCK_VQGAN" "$@" ;;
  reeval)
    # regenerate deliverable samples for an already-trained primary fold (wave-1 folds ran the legacy eval)
    F=$(echo "$@" | grep -oE -- "--folds [0-9]+" | awk '{print $2}')
    CK=$(ls $ROOT/results/ClariGAN_stratified_fold_${F}_specimen_grouped/LBBDM-f16/checkpoint/top_model_epoch_*.pth | tail -n1)
    python eval_fold.py --config configs/Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml --ckpt "$CK" \
      --data_root $DATA --scheme $SCHEME --fold $F --experiment claridi_primary \
      --out_root $ROOT/deliverables --gpu $GPUS --seed 1234 ;;
  pix2pix|cwgan)
    cd $ROOT/repo/baselines
    python kfold_grouped_baselines.py --baseline $TARGET --data_root $DATA --scheme $SCHEME --gpu $GPUS \
      --out_root $ROOT/baselines_out/$TARGET "$@" ;;
  *) echo "unknown target: $TARGET"; exit 1 ;;
esac
