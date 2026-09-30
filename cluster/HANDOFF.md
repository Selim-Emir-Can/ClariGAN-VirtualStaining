# ClariDi — HANDOFF (model-design experiments on the spatial split)
Rewritten 2026-09-29 00:35 PDT. Previous history (the leave-one-specimen-out campaign, Sep 18-28)
is in git (`repo/cluster/HANDOFF.md` before commit on this date) and in the HF archive below.

## 0. STATUS 08:06 Sep 29 — OVERNIGHT RUNS FINISHED, GPUs 1-5 RELEASED
All five experiments completed folds 0-3 (exit 0) and stopped before fold 4 by the 09:00 rule.
Every fold has 5 draws x all test tiles (750/705/850/710 PNGs). No emir process on any GPU.
results_spatial 47 GB (20 top checkpoints + configs), deliverables_spatial 1.5 GB.
Folds 0-3 LPIPS (603 tiles; analysis/lpips_folds0123.txt; CONTEXT ONLY, needs the visual check):
  stock 0.543 | primary 0.509 | label 0.471 | A stained 0.462 | B unstained 0.481
  best-per-tile: A 225, label 156, B 138, primary 72, stock 12
  A best on brain (0.448 vs label 0.471) and 10x10; label best on heart (0.473 vs A 0.487);
  specimen A: primary 0.545 -> A 0.325. Fold 4 not run for any experiment.

### Review pages (02:15 Sep 30), served by http.server 8897 (pid 3808976) over the user's tunnel
- review_spatial/titled.html — grader: per tile, 5 titled plots (Condition|GT|Output, draw 0) one per model,
  blind by default; keys 1 good/2 acceptable/3 bad/4 completely incorrect; export -> sp_grades.json
  {tile: {model: grade}}. Titled plots built by review_spatial/make_titled.py -> review_spatial/titled/ (170 MB).
- review_spatial/index.html — side-by-side comparison + LPIPS summary; claridi_spatial_picker.html = same,
  self-contained (60 MB, download and double-click). sheets/ = static PNG montages.
- Fixed: manifest.csv has CRLF line endings; the pages' CSV parser now splits on \r?\n (GT was 'undefined').

## 1. What ran (started 00:16-00:27 Sep 29, all detached with setsid nohup)
Five experiments, one GPU each, folds 0..4 of the spatial split in order. `run_until.sh` starts a
fold only if its estimated duration (first fold 8000 s, then the last measured fold) ends before
DEADLINE=2026-09-29 09:00, so the GPUs free themselves around 09:00 (a fold may overrun slightly).
| GPU | experiment (deliverables name) | config | log |
|---|---|---|---|
| 1 | sp_stock_vqgan — vanilla L-BBDM, stock ImageNet VQGAN | Template-LBBDM-f16_stockVQGAN_leakfree.yaml | logs_sp_stock.log |
| 2 | sp_primary — our L-BBDM, fine-tuned VQGAN | Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml | logs_sp_primary.log |
| 3 | sp_specimen_cond — + specimen-label embedding | Template-LBBDM-f16_specimen_cond.yaml | logs_sp_speccond.log |
| 4 | sp_refA_stained — + stained-reference conditioning (A) | Template-LBBDM-f16_refA_stained.yaml | logs_sp_refA.log |
| 5 | sp_refB_unstained — + unstained-context conditioning (B) | Template-LBBDM-f16_refB_unstained.yaml | logs_sp_refB.log |
Configs live in repo/BBDM/configs/. Checkpoints: results_spatial/ (auto-pruned to the top
checkpoint + config.yaml after each fold's eval). Generations: deliverables_spatial/<experiment>/
samples/fold_k/<tile_id>_gen{0..4}.png (5 draws, gen j = seed 1234+j), timing/seeds per fold.
Cost: ~95 s/epoch x 50 epochs + ~35 min eval ≈ 2 h/fold -> expect ~4 folds per experiment by 09:00.
Check: `nvidia-smi`; `tr '\r' '\n' < logs_sp_primary.log | grep -a 'Epoch: \[' | tail -1`;
fold events: `grep -a 'GPU [0-9] <-\|exit [0-9]* after\|STOP before\|RUN_UNTIL_DONE' logs_sp_*.log`.
Stop a run cleanly: kill its run_until.sh loop, then the run_kfold.sh wrapper and python driver
FIRST, then workers; afterwards check `ps -u emir -o args= | grep '[e]val_fold'` (a child-first kill
once left an orphan eval).

### Incidents
- 01:38 Sep 29: sp_stock fold 0 trained fine (4933 s, top epoch 48) but its eval died:
  eval_fold.py's config shim lacked the new split_file attribute (AttributeError). Fixed in
  eval_fold.py at 01:39, before any other run reached eval. Recovery (.cache/tmp/stock_resume.sh):
  eval-only of fold 0 (--skip_train), then run_until.sh FOLDS="1 2 3 4" on GPU 1. The fold-0
  training log is kept at .cache/tmp/logs_sp_stock_fold0_train.log.

### Fold-0 first read (02:35 Sep 29; LPIPS, CONTEXT ONLY — needs visual check)
analysis/lpips_compare.py -> analysis/lpips_fold0.txt (+ per-tile json). Mean LPIPS over 150 test
tiles x 5 draws: stock 0.528 | primary 0.512 | specimen label 0.466 | A stained 0.462 |
B unstained 0.468. All three conditioned models ~0.045 better than the primary; best-per-tile
counts: label 52, B 42, A 34, primary 21, stock 1. Specimen A: primary 0.698 vs A 0.345.

### Folds 0+1 (04:16 Sep 29; LPIPS, context only) — analysis/lpips_folds01.txt
291 tiles: stock 0.517 | primary 0.500 | label 0.453 | A 0.457 | B 0.462. Same ordering as fold 0.
Best-per-tile: label 93, B 80, A 68, primary 45, stock 5. A/B beat the label on brain (0.438/0.448 vs
0.448) and lose on heart (0.490/0.486 vs 0.461) and on 5x5 (A 0.467 vs label 0.441). All five runs
started fold 2 at 04:02-04:14 (~1.8 h/fold); fold 3 can still start ~06:00 and end ~07:50.

### Folds 0-2 (06:15 Sep 29; LPIPS, context only) — analysis/lpips_folds012.txt
461 tiles: stock 0.534 | primary 0.507 | label 0.470 | A 0.458 | B 0.479. Fold 2 moved A ahead
(best on 174 tiles vs label 112, B 113, primary 55, stock 7). A leads on brain (0.443 vs label 0.470)
and 10x10; the label still leads on heart (0.469 vs A 0.484). All runs in fold 3 since 05:58-06:11,
ETA ~07:55-08:10; fold 4 will not start (09:00 deadline).

## 2. The split: data/splits/model_design_exp_split.csv (repo/BBDM/spatial_split.py)
- Geometry (verified on pixels, corr ~1.00): each piece image is cut into two non-overlapping
  grids; a 5x5 crop = a 2x2 block of 10x10 crops (specimen D's "5x5" grid is really 6x6). Z is
  D's second half (shares D's part-1 frame); Hpart1 is H's second half. Masked tiles are extra
  cells of the same grids.
- Unit of assignment = connected component of cross-scale pixel overlap (a 5x5 crop + the 10x10
  crops inside it). Units are ordered along each specimen-half's long axis into 5 bands of ~equal
  tile count. Fold k: test = band k, val = neighbour band (k+1, or 3 for k=4), train = the rest.
- Every specimen is in train, val and test of every fold; no pixels are shared across partitions
  (asserted). Test tiles per fold: 150/141/170/142/150; train ~441-462. Each tile is tested once.
- Driver flags: `--scheme spatial --split_file data/splits/model_design_exp_split.csv`.
- User's plan: 5 folds while designing the model; 11 folds for the final paper runs.

## 3. Model variants (branch `spatial-split` on GitHub, last commit 0962f44)
- Specimen label: UNetParams.num_classes=12 (11 specimens A-K + null), ADM label embedding added
  to the timestep embedding; BB.params.class_dropout=0.1 -> null label. Labels parsed from tile
  names in BBDM/specimen_labels.py (Z, Z_part1 -> D; Hpart1 -> H; validated on all 4518 names).
  Not novel on its own (Dhariwal & Nichol 2021; CFG training, Ho & Salimans 2022); needs the
  specimen ID at test time. Kept as the "oracle identity" ablation.
- A / B (reference conditioning, the candidate contribution): per training tile, channel mean+std
  of the frozen-VQGAN latent (512-d); A from STAINED targets, B from UNCLEARED inputs. A sample's
  reference vector = mean over same-specimen training tiles, never from its own crop unit
  (k=8 random in training; all of them at eval). LayerNorm-MLP (zero-init output) -> added to the
  timestep embedding; learned null input with 10% dropout. The bank is rebuilt per fold from that
  fold's training tiles (kfold_grouped._maybe_build_ref_bank, eval_fold.py); it is not saved in
  checkpoints. A = few-shot stained exemplars (works on unseen specimens if a few regions are
  stained); B = needs nothing extra at test time. Clean comparison: label vs A vs B.
- Not implemented yet: classifier-free guidance scale at sampling (only the training dropout).
- Earlier idea list: scale (5x5/10x10) conditioning; texture (LPIPS / frequency) loss.

## 4. Next steps
1. ~09:00: confirm every run_until.sh printed RUN_UNTIL_DONE and no eval_fold is left; list which
   folds finished per experiment (deliverables_spatial/<exp>/samples/fold_k with 5 x test tiles).
2. Visual comparison is the verdict (user: "metrics don't matter, visual quality does"). The old
   review picker (review/index.html, make_titled_plots.py, crosstab_flags.py) was deleted with the
   LOSO outputs; it is in the HF archive (archive_loso/loso_logs_scripts.tar.gz) and can be
   restored and pointed at deliverables_spatial/. LPIPS tracked the user's grades in the LOSO
   review; PSNR did not. Do NOT claim GAN baselines beat diffusion from PSNR (retracted before).
3. Nothing is uploaded to HF or deleted without the user's word.

## 5. Standing rules
- Work only under /local/emir/ClariDi; conda env chatgarment; do not change existing configs
  (new experiments get new config files).
- GPUs: only the ones the user assigns (tonight 1-5, until 09:00). Other users hold 0, 7-9.
- Uploads bulk, few commits (HF limit 128 commits/hour). Nothing deleted without the user's word.
- Git: repo/ on branch spatial-split; mirror this file to repo/cluster/HANDOFF.md and commit after
  each change (-c user.name="Emir Can" -c user.email="emir2903@gmail.com"; `git add -f` for
  cluster/ and configs/). Pushing works: `git -c credential.helper='!gh auth git-credential' push`.
- Be concise with the user.
- pgrep/pkill -f match their own command line: use anchored patterns or `grep '[x]yz'`.

## 6. Where the previous campaign lives (leave-one-specimen-out, 11 folds, done Sep 21)
- HF SelimEmirCan/claridi-results (private): all 9 experiments x 11 folds of generations
  (<exp>/samples.parquet), configs, timing, seeds, ceilings, data_256, fold table;
  visual_review_loso/ (user's grade exports: 5x5 complete, 10x10 incomplete);
  archive_loso/ (logs+scripts+review tooling tarball; training curves tarball).
- HF SelimEmirCan/claridi-checkpoints (public, gated): all 165 checkpoint files.
- GitHub branch specimen-grouped-cv: the LOSO code.
- Key LOSO finding: 55% of graded generations in the worst bucket; failures are per specimen
  (all 5 draws identical grade), not sampling; staining varies too much across the 11 pieces for
  leave-one-specimen-out at n = 1 animal. That motivated this spatial split.
- Local: weights/ (fine-tuned VQGAN epoch=000022.ckpt + stock VQGAN) — keep; not uploaded.
