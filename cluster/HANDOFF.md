# ClariDi — HANDOFF (model-design experiments on the spatial split)
Rewritten 2026-10-04 19:10 PDT. Older history is in git (repo/cluster/HANDOFF.md) and the HF archive (§7).

## 0. NEXT SESSION: WHAT TO DO
1. DONE Oct 4 23:00: fold 4 for all five (750 outputs each); all 753 tiles tested once. GPUs 2, 3, 6 idle
   (user's assignment; ask before using). Final: analysis/lpips_folds01234.txt (+ lpips_folds_0_1_2_3_4.json),
   analysis/color_stats.txt; titled plots rebuilt (fold 4 incl.); review pages + standalone picker now load
   lpips_folds_0_1_2_3_4.json (fold-4 tiles visible). Launch details: run_fold4.sh / run_fold4_prio.sh, git history.
2. Waiting on the user: picks/grades export; then slide deck for the postdoc (methods in LaTeX + 1-2 grids
   per specimen from the user's picks). Do not start until told.
   Deck (Oct 5, private claude.ai artifact): https://claude.ai/artifact/R6DQoPLKF7CRBvhfD2pQ4i — 8 slides,
   training diagram + variant differences only, styled after manuscript Fig. 7 (Times, blue frozen VQ-GAN, green
   L-BBDM, navy dashed). No results/picks yet. Equations rendered with matplotlib mathtext (cm) as PNG assets.
   Manuscript issue flagged to the user: Fig. 2 panel letters (Brain: G-K, Heart: A-F) and its caption (brain A-F,
   heart G-K) both disagree with Fig. 5 and the manifest (brain A,D,E,G,H,I; heart B,C,F,J,K,L).
   Website (Oct 5): LIVE, unlisted at https://selim-emir-can.github.io/ClariDi/ (public repo selim-emir-can.github.io,
   commit 41a31f9; user accepted the exposure; not linked from homepage/sitemap; noindex). Pages: index, tiles,
   wholesample, splits, slides. Source + build scripts: website_ClariDi/ (build_site.py, build_slides.py); repo clone:
   website_clone/. To update: rebuild, rsync -a --delete --exclude 'build_*.py' --exclude 'tools/' website_ClariDi/
   website_clone/ClariDi/, commit, push (TMPDIR on /local). Email draft for Michael given to the user.
   Root fs / (incl /tmp) is 100% full (other users): keep temp files on /local; Bash output capture breaks when /tmp is full.
   Oct 7, Michael's review (needs: fold wording, interpolation framing, D part 0, all-draw aggregate, oracle labels).
   DONE: site commit 02b4154 (scope = within-specimen interpolation; D part 0 note; '(oracle)' / '(extra input)' labels;
   all-draw tables from analysis/metrics_all_draws.py -> metrics_all_draws.md/json: per-draw score, specimen-macro,
   mean +- SD over 5 draws, + without D_p0); proposed split v2 at splits_v2.html (build_split_v2.py).
   Split v2 = data/splits/model_design_exp_split_v2.csv (repo 98d9584): D_p0 (non-nested 6x6 vs 10x10 grids) banded
   per tile with per-fold purge (role_f0..4 incl. 'excluded'; 3-9 tiles/fold, all D_p0); long axis now from tissue
   extent (v1 bug flipped D_p0 and J; 60 tiles change band). v1 CSV/results untouched; v1 folds reproduce.
   Oct 7 23:55: switched to shared queue run_v2_queue.sh (v2_queue.txt, flock): GPUs 1 3 4 6 7 + 8 9; 8/9 start no job
   that would end after 09:00 Oct 8 (user). Original: run_v2.sh, logs_v2_gpu<id>.log, ~10 h; order Ours, Vanilla, B,
   label, A (one fold per GPU each). Spec-leak check fixed (excluded tiles) in 04d79d0. 5 folds is final (user).
   Oct 8 00:46 (user): DELETED v1 results_spatial/ (71 GB ckpts) + deliverables_spatial/ (v1 generations); v1 survives only
   as site JPEGs + analysis/*.json metrics. prune_v2.sh (logs_prune_v2.log) keeps top_model + config per v2 run once its
   timing_fold_k.csv exists and its run_kfold is gone. For v2 analysis/site: point metrics_all_draws.py (D=, split, lpips json)
   and build_site.py (GEN, SPLIT, LPIPS) at deliverables_spatial_v2 / split v2; LPIPS via lpips_compare.py needs the same.
   Oct 8 ~01:00: queued remaining methods on split v2 (user): run_v2b_queue.sh (v2b_queue.txt; pixel, cwgan, encoder,
   pix2pix x 5 folds, longest first) on GPUs 1 3 4 6 7, each starting once its GPU leaves the diffusion queue; a failed
   job stops its GPU's worker (v2b_failed.txt). GAN driver now has --scheme spatial --split_file. prune_v2b.sh replaces
   prune_v2.sh (also GANs: keeps latest_net_G.pth). VQ-GAN weights uploaded to HF claridi-results/weights/.
   run_v2c_extra.sh: GPUs 8/9 (user: until 10:00 Oct 8) take v2b jobs that fit before 10:00 (longest that fits).
   New-machine setup prompt: cluster/NEW_MACHINE_PROMPT.md.
   Was: user checks v2; then retrain 5 models x 5 folds on v2 (sp2_* tags, ~50 GPU-h; needs GPUs: Oct 7 all 10 busy,
   austinchi 0-6, brianchc 7-9); rerun metrics; rebuild site; write the summary for the manuscript session (other machine).
   User: '5 folds' remark = only 5 of a planned 10 folds were run; may switch to 5 folds as final.
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
- Split viewer (Oct 4): review_spatial/split_viz.html (make_split_viz.py -> split_viz/, 5.4 MB). Findings: geometry
  verified (5x5 vs underlying 10x10 r>=0.9); D_p0 is one unit (19/21 tiles, all band 2 -> band 2 has 170 tiles);
  equal tile count != equal area (bands differ up to 2.2x in area); bands thin (11 are one 5x5 crop wide);
  test area within one 10x10 width of train: folds 0,4 ~20% vs folds 1-3 57-67% (val buffers end bands);
  2D blocks barely help. Options for the user (not done): report near/far-from-train test tiles separately;
  area-balanced bands; handle D_p0 6x6 crops.
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

## 4. Results, folds 0-4 (all 753 tiles). LPIPS is CONTEXT ONLY.
analysis/lpips_folds01234.txt: mean LPIPS stock 0.532 | primary 0.508 | label 0.464 | A 0.457 | B 0.473;
best-per-tile A 252, label 212, B 173, primary 97, stock 19. Brain: A best (0.443); heart: label best (0.466
vs A 0.482). A best on A, B, D, E, G, H; label best on I, J, K (green-poor) and F. Specimen A: primary 0.542 -> A 0.340.
Colour (analysis/color_stats.txt, green share G/(R+G), all 5 draws): GT 0.38 | stock 0.30 | primary 0.35 |
label 0.33 | A 0.41 | B 0.38. Same pattern as folds 0-3: regression toward the dataset mean; label tracks
per-specimen colour best on green-poor I/J/K; A overshoots green on dim/red specimens (H 0.37 vs 0.31,
K 0.29 vs 0.21) and recovers green on A (0.58 vs GT 0.61; primary 0.46).
Folds 0-3 numbers (previous): analysis/lpips_folds0123.txt.
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
- Oct 4: alldraws.html + titled.html got 'skip to next specimen' + n / Shift+N (sets the specimen filter
  to the next letter, lands on its first ungraded tile; stat shows 'specimen X: a/b graded').
- review_spatial/titled.html — per tile, titled plots (Condition|GT|Output, draw 0, from
  review_spatial/make_titled.py -> review_spatial/titled/); localStorage "sp_grades".
- review_spatial/index.html ("the picker") — side-by-side + LPIPS summary; picks in localStorage "sp_picks",
  export -> sp_picks.json. Oct 4: added 'next specimen' button + n / Shift+N (sets the specimen filter to the
  next/previous letter; stat shows picks for that letter). User is picking a few tiles per specimen for a
  postdoc slide deck (methods in LaTeX equations + 1-2 grids per specimen from their picks). DO NOT start the
  slides until the user says so; claridi_spatial_picker.html = standalone
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
