# ClariDi specimen-grouped re-run (Stanford cluster)

Everything lives under `/local/emir/ClariDi`. Nothing is written outside it.

| path | what |
|---|---|
| `repo/` | shallow clone of ClariGAN-VirtualStaining @ main |
|  conda env `chatgarment` (shared, torch 2.1.2+cu121; only additive installs) | conda env, python 3.9 / torch 2.5.1+cu118 |
| `data/bbdm/train/{A,B}` | 753 input / target PNG pairs from HF `SelimEmirCan/claridi` |
| `data/bbdm/manifest.csv` | tile_id, specimen, piece, part, tissue, scale, masked, row, col, filenames |
| `weights/epoch=000022.ckpt` | finetuned VQGAN, md5 verified |
| `results/` | per-fold checkpoints (pruned to top_model after eval) + `fold_assignments.{csv,json,txt}` |
| `baselines_out/` | pix2pix / cWGAN outputs on the same folds |
| `data/bbdm256/` | pre-resized 256px copy, bit-identical, 131 MB |
| `k-fold_samples/` | per-fold test outputs |

## What changed in the code

- `repo/BBDM/specimen_kfold.py` (new) replaces `stratified_kfold_85_5_10`.
  Splits by the manifest `specimen` column, so all patches of a specimen, at both
  crop scales, masked or not, land in exactly one partition. Validation is one whole
  held-out training specimen of the same tissue, never a slice of the test specimen.
  `assert_no_leakage` runs per fold and checks specimen overlap, the specimen origin
  of every listed file, duplicate files across partitions, and total coverage.
  The stale `Z -> H` merge is not carried over; `Z` is part of specimen `D`.
- `repo/BBDM/kfold_grouped.py` (new) replaces `k-fold_validation.py`. Every path is a
  CLI flag: `--data_root --config --vqgan_ckpt --results_root --samples_root --gpu_ids`.
  Real NCCL DDP across the requested GPUs, then single-GPU evaluation.
- `repo/BBDM/configs/*.yaml` for the three models we run now point at cluster paths.

## Effective batch size

Config is batch 8 with `accumulate_grad_batches: 4`, i.e. effective 32. The driver
divides accumulation by the GPU count so the effective batch stays 32: 2 GPUs -> accum 2,
4 GPUs -> accum 1. 3 GPUs does not divide 4, so pass `--accumulate_grad_batches`
explicitly or use 2 or 4 GPUs.

## Commands

```
./run_kfold.sh dry                  # fold table + leakage check, trains nothing
GPUS=0,1 ./run_kfold.sh claridi     # primary
GPUS=0,1 ./run_kfold.sh pixel       # pixel-space diffusion ablation
GPUS=0,1 ./run_kfold.sh encoder     # trainable VQGAN encoder ablation
SCHEME=grouped ./run_kfold.sh dry   # 5-fold fallback instead of leave-one-specimen-out
```

See `RUN_NOTES.md` for the autoencoder caveat and the list of live vs stale configs.
