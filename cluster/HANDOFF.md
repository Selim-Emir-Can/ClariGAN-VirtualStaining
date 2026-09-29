# ClariDi — HANDOFF (model-design experiments on the spatial split)
Rewritten 2026-09-29 00:35 PDT. Previous history (the leave-one-specimen-out campaign, Sep 18-28)
is in git (`repo/cluster/HANDOFF.md` before commit on this date) and in the HF archive below.

## 1. What is running (started 00:16-00:27 Sep 29, all detached with setsid nohup)
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
