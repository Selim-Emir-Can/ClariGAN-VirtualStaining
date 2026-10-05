# ClariDi — HANDOFF (model-design experiments on the spatial split)
Rewritten 2026-10-04 19:10 PDT. Older history is in git (repo/cluster/HANDOFF.md) and the HF archive (§7).

## 0. NEXT SESSION: WHAT TO DO
1. Launch fold 4 (the user assigned GPUs 2, 3, 6 on Oct 4; they were idle at 19:05):
     cd /local/emir/ClariDi && setsid nohup ./run_fold4.sh > logs_fold4_launcher.log 2>&1 < /dev/null &
   GPU2: refA then stock | GPU3: refB then primary | GPU6: specimen_cond. ~2 h per job incl. eval,
   so ~4 h total. If the user gives a hand-back time, pass DEADLINE="YYYY-MM-DD HH:MM" (a job only
   starts if its estimate ends before it). Before launching: `nvidia-smi` — GPUs 2, 3, 6 must be empty;
   never touch other GPUs (other users hold 0, 1, 4, 5, 7, 8, 9).
   Watch: logs_sp_<name>_f4.log for 'exit [0-9]* after|STOP before|RUN_UNTIL_DONE|CalledProcessError|
   CUDA out of memory|AttributeError|FileNotFoundError' (pipe through `tr '\r' '\n'`; ignore the benign
   'OSError: Directory not empty: .../pymp-*' finaliser tracebacks).
2. When fold 4 is done for all five: every tile has been tested once. Recompute
   `python analysis/lpips_compare.py 0 1 2 3 4` and `python analysis/color_stats.py` (edit its fold
   assumption if needed), rebuild titled plots `python review_spatial/make_titled.py`, report.
3. Proposed, NOT started (user said "might be worth trying"): specimen-balanced sampling (new config,
   batches draw specimens evenly). Needs GPUs + the user's word.
4. Offered, NOT started: nuclei-level evaluation (Cellpose/StarDist counts, density, detection F1 vs GT)
   as the objective metric; the user distrusts PSNR/SSIM and per-tile LPIPS (rightly).

## 1. Experiments (spatial split, 5 folds; folds 0-3 DONE Sep 29 for all five, fold 4 pending)
| experiment | what | config (repo/BBDM/configs/) | tag |
|---|---|---|---|
| sp_stock_vqgan | vanilla L-BBDM, stock ImageNet VQGAN | Template-LBBDM-f16_stockVQGAN_leakfree.yaml (run_kfold target `stock`) | sp_stock |
| sp_primary | our L-BBDM, fine-tuned VQGAN | Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml (target `claridi`) | sp_primary |
| sp_specimen_cond | + specimen-label embedding | Template-LBBDM-f16_specimen_cond.yaml | sp_speccond |
| sp_refA_stained | + A: stained-reference conditioning | Template-LBBDM-f16_refA_stained.yaml | sp_refA |
| sp_refB_unstained | + B: unstained-context conditioning | Template-LBBDM-f16_refB_unstained.yaml | sp_refB |
Outputs: deliverables_spatial/<experiment>/samples/fold_k/<tile_id>_gen{0..4}.png (5 draws, gen j =
seed 1234+j). Checkpoints: results_spatial/ (top checkpoint + config per fold, 47 GB for 20 folds).
Launcher: run_until.sh (per GPU, folds in order, deadline-aware) via run_kfold.sh with SCHEME=spatial.
Cost: ~95 s/epoch x 50 epochs + ~35 min eval ≈ 1.9-2 h per fold.

## 2. The split: data/splits/model_design_exp_split.csv (repo/BBDM/spatial_split.py)
- Each piece image is cut into two non-overlapping grids; a 5x5 crop = a 2x2 block of 10x10 crops
  (verified on pixels; specimen D's "5x5" grid is really 6x6). Z = D part 1, Hpart1 = H part 1.
- Unit = connected component of cross-scale pixel overlap (a 5x5 crop + the 10x10 crops inside it).
  Units ordered along each specimen half's long axis into 5 bands. Fold k: test = band k,
  val = neighbour band (k+1, or 3 for k=4), train = rest. Every specimen in every partition; no
  shared pixels across partitions (asserted). Test tiles per fold 150/141/170/142/150.
- Known limitation, to state in the paper: a test band touches a training band on one side (shared
  tile edge, no shared pixels). User accepted this; 5 folds for design, 11 for the final paper.
- Driver flags: `--scheme spatial --split_file data/splits/model_design_exp_split.csv`.

## 3. Model variants (branch `spatial-split` on GitHub)
- Specimen label: UNetParams.num_classes=12 (11 specimens + null), ADM label embedding added to the
  timestep embedding; class_dropout 0.1. Labels parsed from tile names (BBDM/specimen_labels.py).
  Not novel (Dhariwal & Nichol 2021; CFG training, Ho & Salimans 2022); "oracle identity" ablation.
- A / B: channel mean+std of the frozen-VQGAN latent (512-d) of same-specimen TRAINING tiles
  (A: stained targets, B: uncleared inputs), never from the tile's own crop unit; k=8 random in
  training, all at eval; LayerNorm-MLP (zero-init) -> added to the timestep embedding; learned null,
  10% dropout. Bank rebuilt per fold (kfold_grouped._maybe_build_ref_bank, eval_fold.py).
- Reviewer-risk notes agreed with the user: A uses stained tissue of the same specimen at test time,
  so it must be framed as a few-shot / extra-input setting, with a FIXED, location-independent
  reference set (K tiles per specimen, outside a buffer), a K sweep (1, 2, 4, 8), and ideally a
  leave-one-specimen-out-with-references test. "Nearest-tile references" were proposed and RETRACTED
  (reintroduces neighbouring-patch leakage). Pooled statistics carry colour more than texture;
  texture options: Gram-matrix (VGG) loss, or cross-attention to full reference latents.

## 4. Results so far (folds 0-3, 603 test tiles). LPIPS is CONTEXT ONLY.
analysis/lpips_folds0123.txt: mean LPIPS stock 0.543 | primary 0.509 | label 0.471 | A 0.462 |
B 0.481; best-per-tile A 225, label 156, B 138, primary 72, stock 12. A best on brain/10x10;
label best on heart. Specimen A (old worst case): primary 0.545 -> A 0.325.
Colour (analysis/color_stats.txt, green share G/(R+G), all 5 draws): GT 0.39 | stock 0.30 |
primary 0.36 | label 0.33 | A 0.42 | B 0.38. Specimens range 0.08 (I) to 0.61 (A); brain (64% of
tiles) is green-rich, so models regress toward the dataset mean (green-poor I/J/K too green, A too
red). The label corrects colour best; A overshoots green on dim/red specimens (its reference is the
specimen-wide mean, dominated by bright tissue). Visual check of specimen A sheet: A recovers the
green nuclear stain where the primary is dark red.
User's stance: visual quality is the verdict; PSNR/SSIM misleading; LPIPS unreliable per tile.
Planned evaluation for the paper: blind human grading (primary) + nuclei-level agreement (objective)
+ LPIPS/FID as secondary context only.

## 5. Review pages (served by `python -m http.server 8897 --bind 127.0.0.1 --directory /local/emir/ClariDi`,
pid 3808976; restart with setsid nohup if dead). User opens them via VS Code Ports (forward
127.0.0.1:8897; their local port was 8898) or PowerShell `ssh -L 18897:127.0.0.1:8897 emir@apollo`
(local 8897 is blocked on their Windows laptop). Pages:
- review_spatial/alldraws.html — per tile, one row per method: Condition | GT | gen0-4. Grade 1-4 per
  method; localStorage "sp_grades_all"; export -> sp_grades_alldraws.json. Names shown by default
  (blind checkbox shuffles/hides).
- review_spatial/titled.html — per tile, titled plots (Condition|GT|Output, draw 0, from
  review_spatial/make_titled.py -> review_spatial/titled/); localStorage "sp_grades".
- review_spatial/index.html — side-by-side + LPIPS summary; claridi_spatial_picker.html = standalone
  60 MB copy; sheets/ = PNG montages.
- The user graded some tiles but has NOT exported yet: grades live in their browser for the exact
  origin (localhost:<port>); remind them to export and copy the JSON to the project root.
- Gotcha fixed: data/bbdm256/manifest.csv has CRLF line endings (split on \r?\n in JS).

## 6. Standing rules
- Work only under /local/emir/ClariDi; conda env chatgarment; existing configs unchanged (new
  experiments get new config files). Only the GPUs the user assigns.
- Nothing uploaded or deleted without the user's word; HF uploads bulk, few commits.
- Never kill a job children-first (kill run_until loop, run_kfold.sh, python driver, then workers;
  check for orphan eval_fold.py).
- Git: repo/ on branch spatial-split; mirror this file to repo/cluster/HANDOFF.md and commit after each
  change (-c user.name="Emir Can" -c user.email="emir2903@gmail.com"; `git add -f` for cluster/ and
  configs/); push with `git -c credential.helper='!gh auth git-credential' push`.
- Be concise with the user. They are writing a paper; claims must survive reviewers.
- Incidents last run: Z_part1 name parse (fixed), eval_fold config shim lacked split_file (fixed).

## 7. Previous campaign (leave-one-specimen-out, 11 folds, done Sep 21) — archived
HF SelimEmirCan/claridi-results (private): generations, configs, ceilings, data_256,
visual_review_loso/ (user's LOSO grades), archive_loso/ (logs, scripts, training curves).
HF SelimEmirCan/claridi-checkpoints (public, gated): 165 checkpoints. GitHub branch specimen-grouped-cv.
Finding: 55% of graded generations in the worst bucket; failures per specimen, not sampling.
weights/ (fine-tuned + stock VQGAN) stays local, not uploaded.
