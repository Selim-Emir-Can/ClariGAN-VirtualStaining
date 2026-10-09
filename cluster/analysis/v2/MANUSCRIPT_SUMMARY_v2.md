# ClariDi: model-design experiments on spatial split v2 — summary for the manuscript session

Written 2026-10-09 on apollo (/local/emir/ClariDi). Code: GitHub branch `spatial-split`. Results site (unlisted):
https://selim-emir-can.github.io/ClariDi/. Full tables: `metrics_all_draws_v2.md` (next to this file).
Every number below comes from `analysis/metrics_all_draws.py --v2` (scores) or the run logs (timings).

## 1. Task and scope

- UA (uncleared autofluorescence) -> C&SF (cleared & stained fluorescence) virtual staining, 256x256 px tiles.
- 753 tiles from 11 specimens: brain A, D, E, G, H, I (483 tiles); heart B, C, F, J, K (270 tiles).
  Two crop grids per piece: 179 tiles from 5x5 crops and 574 from 10x10 crops.
- **Scope: within-specimen interpolation.** Every specimen contributes training tiles to every fold (different
  regions of the same specimen). These results measure how well a model fills in held-out regions of specimens
  it has seen during training. They do NOT measure generalization to unseen specimens (that is the separate
  leave-one-specimen-out campaign, 11 folds, archived on HF `archive_loso/`).
- **5 folds**, final (an earlier plan of 10/11 spatial folds was dropped). Every tile is tested exactly once.

## 2. Split v2 (`data/splits/model_design_exp_split_v2.csv`, built by `BBDM/spatial_split.py`)

- Units: a 5x5 crop together with the 10x10 crops it overlaps (connected components of cross-scale pixel
  overlap), so overlapping pixels never fall in different partitions.
- Units are ordered along each piece's long axis and cut into 5 bands. Fold k: test = band k,
  val = the neighbouring band (k+1; band 3 for fold 4), train = the other three. Every specimen appears in every
  partition. No pixel is shared across partitions (asserted in code).
- Test tiles per fold: 151 / 148 / 157 / 143 / 154 (sum 753). Train / val / excluded per fold:
  453/146/3, 443/153/9, 449/139/8, 452/152/6, 455/141/3.
- Changes from v1 (the only differences): (a) D part 0 handling, below; (b) the long axis is now measured from the
  tissue's own extent (v1 measured it from the image origin, which picked the short axis for D part 0 and J);
  60 tiles changed band (J 44, D part 0 16).
- **Limitation (state in the paper):** a test band touches a training band along one edge (shared tile border,
  no shared pixels). Test tiles at that edge are spatially adjacent to training tissue, so performance near the
  edge may be optimistic relative to the band interior. Not quantified separately.
- Equal tile counts per band are not equal areas (bands differ up to ~2.2x in area in v1; not re-measured for v2).

## 3. D part 0 handling

- In D part 0 the 5x5 crops lie on a 6x6 grid, so they straddle the 10x10 grid. Grouping overlapping crops merged
  the whole piece into 2 units (19 + 2 tiles); in v1 it therefore had no test tiles in folds 0, 1, 3, no val tiles
  in folds 0, 2, 4, and only 2 training tiles in folds 1 and 2.
- v2: a piece whose largest overlap group holds more than 1/5 of its tiles (only D part 0) is banded tile by
  tile, and each fold is purged: a val tile that overlaps a test tile, and a train tile that overlaps a test or
  val tile, sits out that fold ("excluded"; 3–9 tiles per fold, all from D part 0).
- D part 0 per fold (test / val / train / excluded): 4/2/12/3 · 4/1/7/9 · 5/0/8/8 · 4/2/9/6 · 4/2/12/3.
  Fold 2 has no D part 0 val tiles; specimen D keeps val tiles from D part 1 in every fold.
- Sensitivity: all tables are also computed without D part 0 (732 tiles). Removing it changes headline LPIPS by
  at most 0.001 and PSNR by at most 0.05 dB for any method.

## 4. Methods (9), all trained and tested on the same 5 folds

| label in tables | what it is | outputs per tile |
|---|---|---|
| Vanilla L-BBDM | latent Brownian-bridge diffusion (L-BBDM, f16) with the stock ImageNet VQ-GAN | 5 draws |
| Ours (L-BBDM) | L-BBDM with the VQ-GAN fine-tuned on C&SF (frozen during diffusion training) | 5 draws |
| + specimen label (oracle) | Ours + specimen-identity embedding (12 classes: 11 specimens + null; ADM-style label embedding added to the timestep embedding; 10% class dropout) | 5 draws |
| + A stained refs (extra input) | Ours + channel mean/std of frozen-VQ-GAN latents of same-specimen stained (C&SF) training tiles, never from the tile's own unit (k = 8 random in training, all at eval), via zero-init LayerNorm-MLP added to the timestep embedding; learned null, 10% dropout | 5 draws |
| + B unstained ctx | as A, but from unstained (UA) training tiles of the specimen | 5 draws |
| Pixel-space BBDM | BBDM directly in pixel space (no VQ-GAN); U-Net sized to roughly match Ours' trainable parameter count | 5 draws |
| Ours + trainable encoder | Ours, but the VQ-GAN encoder and quant_conv (initialized from the fine-tuned VQ-GAN) are trained jointly with the U-Net; decoder frozen | 5 draws |
| cWGAN | conditional WGAN-GP baseline (U-Net generator unet_256, ngf 144, layer norm, batch 4, 25+25 epochs) | 1 |
| pix2pix | pix2pix baseline (unet_256, ngf 144, 25+25 epochs); inference `--eval --dropout_at_inference --inference_seed 1234` | 1 |

- **Labels.** "(oracle)": the specimen-label model needs the specimen's identity at test time and that the
  specimen was seen in training; it is an oracle ablation, not a deployable method. "(extra input)": model A uses
  stained tissue of the same specimen at test time, an input plain UA -> C&SF does not have; frame it as a
  few-shot / extra-input setting. B uses only unstained tissue of the same specimen (no stained information),
  but still same-specimen context.
- Diffusion sampling: draw j uses seed 1234 + j (j = 0..4); all draws kept, none selected or reordered.
- GANs give one output per tile from a single seeded inference pass. They are not strictly deterministic
  (pix2pix keeps dropout on at inference; cWGAN runs in train mode, as the fork does by default).
- Cost per fold (wall time, one GPU): L-BBDM variants 2.0 h; trainable encoder 3.5 h; pix2pix 2.6 h;
  cWGAN 5.5 h; pixel-space BBDM 8.7 h.

## 5. Scoring protocol (fixed before looking at the numbers; unchanged from v1)

- Every tile is a held-out prediction. Each of the 5 draws is scored separately against C&SF at 256 px
  (no best-of-n, no example selection).
- Metrics per output: LPIPS (AlexNet), PSNR and SSIM (RGB, data_range 255), and colour error
  |G/(R+G) of output − G/(R+G) of C&SF| ("|Δ green share|", tile means).
- Per draw: mean over the tiles of each specimen, then the unweighted mean over the 11 specimens
  (specimen-macro). This gives 5 numbers per method; tables report mean ± SD across the 5 draws.
  The SD reflects sampling variability only, not training-run variability (one training seed per fold).
- GANs: same per-specimen aggregate on their single output; one value, no SD.
- Secondary: tile-micro mean, brain/heart, per specimen, and without D part 0.
- Metrics are context. The planned primary evidence is blind human grading plus nuclei-level agreement; the user
  distrusts PSNR/SSIM and per-tile LPIPS.

## 6. Results (split v2, all 753 tiles, 11 specimens, specimen-macro, mean ± SD across draws)

| method | LPIPS ↓ | PSNR ↑ | SSIM ↑ | \|Δ green share\| ↓ |
|---|---|---|---|---|
| Vanilla L-BBDM | 0.521 ± 0.004 | 18.45 ± 0.18 | 0.431 ± 0.007 | 0.123 ± 0.013 |
| Ours (L-BBDM) | 0.501 ± 0.002 | 18.29 ± 0.09 | 0.481 ± 0.002 | 0.110 ± 0.006 |
| + specimen label (oracle) | 0.454 ± 0.002 | 19.44 ± 0.11 | 0.509 ± 0.003 | 0.068 ± 0.003 |
| + A stained refs (extra input) | 0.457 ± 0.002 | 19.16 ± 0.10 | 0.500 ± 0.006 | 0.078 ± 0.003 |
| + B unstained ctx | 0.485 ± 0.002 | 18.73 ± 0.10 | 0.490 ± 0.008 | 0.106 ± 0.004 |
| Pixel-space BBDM | 0.594 ± 0.002 | 17.73 ± 0.17 | 0.457 ± 0.004 | 0.173 ± 0.002 |
| Ours + trainable encoder | 0.704 ± 0.005 | 14.84 ± 0.20 | 0.363 ± 0.011 | 0.172 ± 0.003 |
| cWGAN (1 draw; collapsed in 4/5 folds, see §7) | 0.888 | 12.01 | 0.397 | 0.347 |
| pix2pix (1 draw) | 0.582 | 19.35 | 0.509 | 0.134 |

Tile-micro, without-D-part-0, brain, heart and per-specimen tables: `metrics_all_draws_v2.md`.
Tiles where each method has the lowest mean LPIPS (context only): specimen label 218, A 201, B 120, Ours 107,
pix2pix 53, Vanilla 27, pixel 22, trainable encoder 5, cWGAN 0.

### What the numbers support (and what they do not)

- Ours vs Vanilla (stock VQ-GAN): better LPIPS (0.501 vs 0.521), SSIM (0.481 vs 0.431) and colour error
  (0.110 vs 0.123), but **not PSNR** (18.29 vs 18.45; Vanilla's draw SD is 0.18). In v1, Ours also led on PSNR
  (18.63 vs 17.69). Do not claim a PSNR advantage over Vanilla.
- **pix2pix beats Ours on PSNR (19.35 vs 18.29) and SSIM (0.509 vs 0.481)**, but not on LPIPS (0.582 vs 0.501)
  or colour error (0.134 vs 0.110). This is the usual pattern of a regression-like generator producing smoother,
  mean-like outputs; the claim for Ours must rest on perceptual quality (LPIPS, human grading), not PSNR/SSIM.
- Ours beats pixel-space BBDM on every metric (LPIPS 0.501 vs 0.594), and the trainable encoder is far worse
  than the frozen fine-tuned VQ-GAN on every metric (LPIPS 0.704, PSNR 14.84): support for a frozen,
  fine-tuned latent space.
- Specimen label (oracle) and A (extra input) are tied on LPIPS (0.454 vs 0.457); label is ahead on PSNR, SSIM and
  colour. Both use information unavailable to plain UA -> C&SF. B (same-specimen unstained context only) improves
  on Ours on every metric (LPIPS 0.485 vs 0.501), a smaller gain.
- Brain vs heart: A is best on brain LPIPS (0.448 vs label 0.449); label is best on heart (0.460 vs A 0.469).
- **Training-seed variance is not measured.** Vanilla moved by +0.08 SSIM and +0.76 dB PSNR between v1 and v2,
  although only 60 of 753 tiles changed band; this suggests run-to-run variance comparable to some between-method
  gaps. Gaps of a few thousandths in LPIPS between the leading variants should not be claimed as differences.

## 7. cWGAN instability (report it; not retrained, by decision)

- In v2 folds 0, 1, 3 and 4 the cWGAN generator diverged in the first epoch: L1 rose from ~19 to ~34 within ~700
  iterations and never recovered (final L1 32–43 vs 7 in fold 2), and validation L1 stayed identical from epoch 5
  to 50 (fold 1: 0.376744). Outputs are saturated: every pixel is exactly 0 or 255 per channel (e.g. 75% black,
  25% pure red), the same image for every tile. Only fold 2 trained normally.
- Most likely mechanism: the generator's tanh output saturates after an early critic spike, gradients through it
  vanish, and the generator cannot recover. Options were identical across folds (only the data lists and GPU
  differ), so it is optimization instability, not a configuration or data error. Not tested further.
- **The LOSO cWGAN has the same failure in 3 of 11 folds** (folds 2, 8, 9: 100% saturated pixels, checked on
  HF `cwgan/samples.parquet`). Any LOSO cWGAN numbers in the manuscript average in 3 collapsed models and need a
  footnote or correction.
- The cWGAN row above is scored as run (collapsed folds included). Suggested wording: "cWGAN training was unstable:
  the generator collapsed to a saturated, input-independent output in 4 of 5 folds (and in 3 of 11 folds of the
  leave-one-specimen-out evaluation)."
- All other methods were screened the same way (fraction of saturated pixels per fold): none collapsed.

## 8. Changes from v1 to v2 (for the record)

Headline (specimen-macro LPIPS / PSNR / SSIM), v1 -> v2:
Vanilla 0.529/17.69/0.347 -> 0.521/18.45/0.431 · Ours 0.506/18.63/0.492 -> 0.501/18.29/0.481 ·
label 0.460/19.03/0.501 -> 0.454/19.44/0.509 · A 0.457/18.82/0.471 -> 0.457/19.16/0.500 ·
B 0.471/19.28/0.502 -> 0.485/18.73/0.490. v1 had only the five L-BBDM variants. v1 outputs are deleted;
v1 survives as `analysis/metrics_all_draws.md` and the earlier site images.

## 9. Files

- On HF (private `SelimEmirCan/claridi-results`, folder `manuscript_handoff/`): this summary,
  `metrics_all_draws_v2.md` (all tables), `lpips_folds01234_v2.txt` (LPIPS by scale/tissue/specimen),
  `model_design_exp_split_v2.csv` (the split, per-tile role in each fold).
- On apollo: per-tile, per-draw scores `analysis/v2/metrics_all_draws.json`; per-draw LPIPS
  `analysis/v2/lpips_folds_0_1_2_3_4.json`; outputs `deliverables_spatial_v2/<experiment>/samples/fold_k/`;
  consolidated parquet `deliverables_spatial_v2/parquet_merged/<experiment>/` (not uploaded yet); checkpoints
  pruned to the top/latest weights per fold (`results_spatial_v2/`, `baselines_out_v2/`).
