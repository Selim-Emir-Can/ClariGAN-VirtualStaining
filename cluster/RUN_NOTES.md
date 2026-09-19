# ClariDi specimen-grouped re-run — run notes

## Scope of the held-out split (read before reporting numbers)

The **diffusion stage is specimen-held-out**. The **frozen autoencoder is not.**

The VQGAN (`weights/epoch=000022.ckpt`, md5 `3dcd0c2eba10bbbdbbef3970d1a214a0`) was
fine-tuned once on the full dataset before this work, and the same checkpoint is reused,
frozen, in every fold. Under leave-one-specimen-out the encoder has therefore seen images
of the specimen held out for testing.

This is a deliberate, cost-driven decision: per-fold VQGAN re-fine-tuning is future work.
**Do not describe these results as fully specimen-held-out end to end.** The paper
discloses this explicitly.

Why it is a weaker channel than the patch-level leak it replaces: the VQGAN is trained for
reconstruction only and never sees the uncleared-to-cleared mapping, and its reconstruction
ceiling of 23.28 dB sits well above the full model's 19.10 dB, so it is not what drives the
reported numbers. The patch-level leak, by contrast, put spatially overlapping crops of the
same tissue in train and test, with the target mapping attached.

The pixel-space ablation (`Template-BBDM_pixel_256_matchUNet.yaml`) uses no VQGAN, so that
run is specimen-held-out end to end and is the clean point of comparison.

## Splitter

Leave-one-specimen-out, 11 folds, grouped on the dataset's `specimen` column. Specimen `D`
comprises pieces `D` and `Z`; specimen `H` comprises `H` part 0 and part 1. Verified against
the materialized manifest. The old `stratified_kfold_85_5_10` mapped `Z -> H`, which is
wrong (`Z` is `D_part1`); nothing here reproduces that. Validation is one whole held-out
training specimen of the same tissue, never a slice of the test specimen.

`assert_no_leakage` runs per fold before training and checks: no specimen in two partitions;
every file traced independently back to its expected specimen; no file in two partitions;
total coverage equals the manifest. Verified to fire on both injected-leak negative controls.

## Leak-free ablation (stock ImageNet VQGAN)

A fourth run trains the same L-BBDM in the latent space of the **stock ImageNet f16/16384
VQGAN**, which has never seen tissue. Combined with specimen-grouped folds, no component has
seen the held-out specimen, so that run alone is specimen-held-out end to end. Same
architecture as the primary; only `ckpt_path` differs. One run per fold, no VQGAN training.

Config: `Template-LBBDM-f16_stockVQGAN_leakfree.yaml` (written fresh; the similarly named
`Template-LBBDM-f16_imagenetVQGAN.yaml` is stale and points at a `finetuned-full` checkpoint).

Absolute numbers are expected to be lower, because the VQGAN was fine-tuned precisely
because the stock one reconstructs tissue poorly. This run is a **clean lower bound**, not a
competitor to the headline, and both are reported. The question it answers is whether
ClariDi's ordering against the baselines survives; if it does, the leakage criticism is met.
It does **not** replace the primary result.

Priority: after the primary LOSO run, ahead of the pixel-loss variant.

### Reconstruction ceilings (required for interpretation)

The end-to-end gap between the two runs otherwise conflates removal of encoder leakage with
loss of domain-specific reconstruction quality. `vqgan_reconstruction_ceiling.py` encodes and
decodes the held-out targets through each autoencoder and reports PSNR / SSIM / LPIPS, using
exactly the preprocessing the model sees (Resize to 256, ToTensor). Run it for both
checkpoints; the published fine-tuned figure is 23.28 dB PSNR on the stain domain.

## Augmentation, precisely (for Methods)

The training dataset reports 8x its real length per epoch, but not all eight copies are
augmented. Verified empirically on the loader: copies 1-4 are augmented, copies 0, 6, 7
are returned raw, and copy 5 is raw except its first index. So each real pair is seen
about **four times raw and four times augmented per epoch**, not eight times augmented.
Augmentation is online (fresh random parameters each time), five operations (horizontal
flip, vertical flip, rotation up to 180 degrees, translation up to 5 percent, resized crop
at scale 0.8-1.0), identical parameters applied to input and target, training split only,
and applied to the 256-pixel tensor after the resize inside the loader. The `flip` config
key is dead: the transform draws its own horizontal-flip coin and ignores the probability
passed to it.

## GAN baselines on the same folds

pix2pix and cWGAN must be re-run on the grouped folds too; otherwise the headline table
pits a clean ClariDi against baselines that still enjoy the patch-level leak. Their
original drivers each carried a verbatim copy of the broken splitter (same Z -> H bug).
`baselines/kfold_grouped_baselines.py` replaces both and imports the shared
`BBDM/specimen_kfold.py`, so all models train and test on identical folds (the baseline
fold table is byte-identical to `results/fold_assignments.csv`). Both GANs use the same
BBDM augmentation pipeline through the `bbdm_aligned` dataset wrapper. Generator width
`ngf=144` (~273M parameters) matches BBDM's 259M trainable. 50 epochs (25 + 25 decay).
Outputs go to `baselines_out/<baseline>/`.

## Scope (locked 2026-09-18 20:10)

Six experiments, nothing else: claridi_primary, claridi_stock_vqgan (the one new ablation),
trainable_encoder (fine-tuned init, exactly as in the paper, so the only variable is frozen vs
trainable), pixel_space, pix2pix, cwgan. pixel_loss is DROPPED (a discarded multi-term loss,
not in the manuscript). A stock-initialised trainable-encoder variant was considered and
rejected because it changes two variables at once.

## Disk policy

A fold at steady state holds 13 GB: the model (2.37 GB, includes EMA weights and the frozen
VQGAN) saved three ways (last / latest / top) and the Adam state (2.07 GB) saved three ways.
Only `top_model_epoch_*.pth` is needed to reproduce the evaluation. The driver now prunes
each fold to that file plus `config.yaml` once its evaluation has finished (13 GB -> 2.4 GB;
`--keep_optim` disables this). `prune_finished_folds.sh` does the same for folds trained
before the change; it only touches folds whose sample directory already has outputs.
Pre-resized data (`data/bbdm256`, 131 MB) replaces the 2.5 GB originals for all runs
after wave one and is bit-identical.

## Deliverables (HF repo `SelimEmirCan/claridi-results`, private)

No scoring on the cluster: the manuscript side scores the PNGs. Layout:

    fold_assignments.csv, manifest.csv, run_metadata.json
    data_256/                                pre-resized 256px inputs + targets (bit-identical)
    ceilings/{finetuned,stock}/<tile_id>.png target encoded+decoded through each VQGAN
    <experiment>/config.yaml
    <experiment>/timing_fold_<k>.csv         wall-clock, GPUs, epochs, per-patch inference s
    <experiment>/seeds_fold_<k>.json         gen_idx -> torch seed
    <experiment>/samples/fold_<k>/<tile_id>_gen<j>.png   ALL 5 generations, BARE 256x256

Rules: bare outputs only (no titles / montages / overlays; the legacy
sample_to_eval_combined_with_uncertainty path is NOT used for these files); gen<j> is the
raw generation index j=0..4, gen0 = seed 1234, never reordered or dropped; baselines write
one deterministic PNG per tile as gen0; tile_id = the HF dataset tile_id, 753 tiles each in
exactly one fold; samples are never pruned without reporting the size first.
Checkpoints: keep the exact 2.37 GB top checkpoint per fold, not stripped; only the
primary's 11 go to HF (model release), the rest stay on the cluster.

## Second pass (reviewer CRITICAL 10), queued after the six paper experiments

- **Multi-seed on the primary**: claridi_primary re-trained with training seeds 5678 and
  9012 on the SAME LOSO-11 folds (`claridi_primary_seed5678`, `_seed9012`). Sampling seeds
  unchanged (gen j = 1234 + j) so the comparison isolates training-seed variance.
- **Deterministic U-Net with L1** (`unet_l1`): the pix2pix generator (unet_256, ngf=144,
  ~273M) trained with L1 only, no discriminator / adversarial / perceptual term
  (`baselines/pytorch-CycleGAN-and-pix2pix/models/unet_l1_model.py`). Inference in eval
  mode: dropout off, batch-norm statistics fixed. One output per tile as gen0.
- Deferred, to be stated as future work: conditional diffusion without the Brownian
  bridge; separate-scale (5x5-only / 10x10-only) models.

## GAN inference modes and seeds

- pix2pix: `--eval --dropout_at_inference --inference_seed 1234`: batch-norm statistics
  fixed, dropout layers re-enabled, seeded once before inference (reviewer 10's request).
- unet_l1: `--eval`: dropout off, batch-norm fixed, fully deterministic.
- cWGAN: the fork's default (no eval flag, train-mode BN and dropout), as originally run.
- Both GAN forks now take `--seed` (1234) and set cudnn deterministic=True, benchmark=False,
  matching the BBDM runs (main.set_random_seed) and eval_fold.py. Recorded in run_metadata.

## Reporting conventions

- Single-generation metrics are the headline. Best-of-five appears only as an explicitly
  labelled oracle bound.
- Report per-specimen macro-averages alongside patch-weighted means. Under LOSO the
  per-specimen macro-average is the fold-level average.
- Record fixed seeds (driver default 1234) and save the fold-assignment table with results.

## Live configs

| config | status |
|---|---|
| `Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml` | PRIMARY (ClariDi), uses the VQGAN ckpt |
| `Template-LBBDM-f16_imagenetVQGAN_finetuned_trainable_encoder.yaml` | ablation, uses the ckpt |
| `Template-BBDM_pixel_256_matchUNet.yaml` | pixel-space, no VQGAN |
| `Template-LBBDM-f16_stockVQGAN_leakfree.yaml` | leak-free ablation, stock ImageNet VQGAN |

The other 11 files in `configs/` are stale and point at checkpoints that do not exist. Only
the four above have been repointed at cluster paths. If a config fails to load, check this
list first.

## Pre-flight checks already done (no GPU used)

- VQGAN checkpoint md5 matches; loads and round-trips a 256x256 image on CPU.
- Full ClariDi model builds and returns a finite loss on a real fold-0 training batch on CPU.
- Fold 0 dataset sizes: 4992 training samples (624 patches x 8 augmentations), 50 validation,
  79 test.
- Leakage assertion passes on all 11 folds; fold table saved to `results/fold_assignments.*`.
