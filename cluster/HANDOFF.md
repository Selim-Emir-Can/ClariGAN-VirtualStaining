# ClariDi re-run — handoff for the next Claude session (written 2026-09-18 20:50)

Read this first, then `RUN_NOTES.md` (protocol, scope, caveats) and `README_cluster.md`.
Constraints: work only under `/local/emir/ClariDi`; GPUs 1-9 are ours (GPU 0 belongs to
another user); env is conda `chatgarment` (python 3.10, torch 2.1.2+cu121); never build a
new env; minimise disk; never touch files the user did not ask about.

## What is running (unattended, survives session close: all nohup)
- `queue_runner.sh` (pgrep -f queue_runner.sh) feeds `queue.txt` top-down to any GPU in
  1-9 with no compute process owned by emir; launches `GPUS=<g> ./run_kfold.sh <line>` with
  log `logs/q_<line sanitised>.log`; consumed lines go to `queue_done.txt`. Stop with
  `touch queue_stop`; restart with `rm queue_stop; GPUSET="1 2 3 4 5 6 7 8 9" nohup ./queue_runner.sh > logs/queue_runner.log 2>&1 &`.
- Directly launched (not via queue): primary folds 0-5 on GPUs 4-9 (`logs/claridi_fold{0..5}.log`,
  started 19:27, OLD driver code: legacy eval into `k-fold_samples/`, no auto-prune) and
  stock folds 0-1 on GPUs 1-2 (`logs/stock_fold{0,1}.log`). Stock fold 2 came from the queue.
- Queue order: claridi 6-10 -> pixel probe -> reeval 0-5 -> stock 3-10 -> pix2pix 0-10 ->
  cwgan 0-10 -> encoder 0-10 -> [pixel_space, to insert] -> claridi seed5678 x11 ->
  claridi seed9012 x11 -> unet_l1 x11.

## Pending actions for the next session (in order)
1. ~21:35-22:00: primary folds 0-5 finish training, run the legacy eval (~25 min), then
   exit. Then run `./prune_finished_folds.sh` (dry) and `./prune_finished_folds.sh --yes`
   to drop their optimizer/last/latest checkpoints (13 GB -> 2.4 GB each). It only touches
   folds whose `k-fold_samples/<save_name>` has outputs.
2. Pixel-space probe (queue line `pixel --folds 0 --max_epoch 1 ...`, log `logs/q_pixel*.log`):
   read `training time:` (1 epoch) and the eval `s/generation`. Project per fold =
   50 x epoch + (tiles x 5 x s/gen), x11 for LOSO. RULE (agreed with the manuscript
   session): LOSO-11 if projected total <= 150 GPU-h, else grouped-5. Then insert lines
   ABOVE the "second pass" marker in `queue.txt`:
     LOSO:      `pixel --folds k`  for k in 0..10
     grouped-5: `pixel --folds k --scheme grouped --n_folds 5 --results_root /local/emir/ClariDi/results_pixel_grouped5`
                for k in 0..4 (separate results_root so the LOSO fold table in results/ is
                NOT overwritten; afterwards copy results_pixel_grouped5/fold_assignments.csv
                into deliverables/pixel_space/fold_assignments_grouped5.csv).
   Report the measured number to the user either way. Delete `probes/` when done with it.
3. `reeval --folds k` (k=0..5) regenerates the wave-1 samples into
   `deliverables/claridi_primary/` with explicit seeds. After all six succeed, the legacy
   `k-fold_samples/fold_{0..5}_specimen_grouped/` dirs are redundant: report their size to
   the user before deleting (rule: never prune samples without telling them the size).
4. Failures: look for Traceback / CalledProcessError in `logs/q_*.log`. Relaunch by
   re-adding the exact line to the top of `queue.txt` (fix the cause first). Known
   non-bugs: GAN probes failed at test.py only because save_epoch_freq=5 (real 50-epoch
   runs save at epoch 50); pynvml/pkg_resources warnings are noise.
5. When everything is done: `python collect_deliverables.py --check` (integrity: 753 tiles,
   each in exactly one fold, 5 gens, sizes). Upload (`--upload`) ONLY when the user asks;
   the token may be read-only.
6. Ceiling PNGs (`deliverables/ceilings/{finetuned,stock}`, CPU jobs, logs
   `logs/ceiling_pngs_*.log`) should finish on their own (753 each). Check counts.

## Costs (measured; per fold incl. eval) and ETA
primary 2.5 h | stock 2.5 h | trainable_encoder 4.8 h | pix2pix 4.0 h | cwgan 4.2 h |
unet_l1 ~3 h est | pixel_space: from probe. 285 GPU-h excl. pixel-space -> ~Sep 20 morning
on 9 GPUs; +14 h grouped-5 / +30 h LOSO for pixel-space.

## Gotchas learned tonight
- `pkill -f` / `pgrep -f` match their own command line: never use a pattern that appears
  in the command you are running (killed my own shell once; a waiter looped forever).
- `.gitignore` is `*` with allowlists: `git add -f cluster` for the cluster/ scripts.
- The git identity used: `-c user.name="Emir Can" -c user.email="emir2903@gmail.com"`;
  push with `git -c credential.helper='!gh auth git-credential' push origin specimen-grouped-cv`.
- Home disk is 100% full: keep every cache in the project (`TORCH_HOME=.cache/torch`,
  pip `--no-cache-dir`, HF `cache_dir=` then delete).
- eval runs as a subprocess after training, so edits to `eval_fold.py` apply to folds
  already training; edits to `kfold_grouped.py` only apply to jobs launched afterwards.
- Monitors/watchers from the previous session are GONE; re-arm your own (poll
  `logs/q_*.log`, `logs/queue_runner.log`).

## The other Claude session (manuscript side)
Reviews via the GitHub branch `specimen-grouped-cv`; scores the PNGs itself (we do NOT
score); wants bare 256x256 PNGs named `<tile_id>_gen<j>.png`, gen0 = seed 1234, timing.csv,
seeds json, run_metadata.json. Big artifacts for it go to HF (user's account), not pastes.
Agreed and closed: final 9 experiments, pix2pix inference mode, cuDNN recording,
pixel-space rule. It expects in the morning: the probe number, failures, deliverables state.
