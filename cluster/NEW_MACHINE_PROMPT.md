You are setting up the ClariDi virtual-staining experiments on a new machine. I'm writing a paper; every claim must survive reviewers. Be concise with me. Do NOT launch training until I've told you which GPUs to use. Don't delete or upload anything without my say-so.

## Sources
- Code: GitHub `Selim-Emir-Can/ClariGAN-VirtualStaining`, branch `spatial-split` (use `gh` for auth). BBDM code is in `BBDM/`, GAN baselines in `baselines/`, and cluster-side scripts, split CSVs, analysis and site build scripts are in `cluster/`.
- READ FIRST: `cluster/HANDOFF.md` (current state, decisions, standing rules), then `cluster/RUN_NOTES.md` and `cluster/README_cluster.md`.
- Data: private HF dataset `SelimEmirCan/claridi-results`, folder `data_256/` (753 input/target pairs at 256 px + manifest), plus `manifest.csv` at the repo root. The native-resolution originals are in the private HF dataset `SelimEmirCan/claridi` (`cluster/materialize_dataset.py`). They're only needed to regenerate split CSVs.
- Weights (frozen VQ-GANs):
  - Fine-tuned VQ-GAN `epoch=000022.ckpt`, md5 3dcd0c2eba10bbbdbbef3970d1a214a0: private HF dataset `SelimEmirCan/claridi-results`, `weights/epoch=000022.ckpt`.
  - Stock ImageNet VQ-GAN `vqgan_imagenet_f16_16384_stock.ckpt` + `.yaml`, md5 229b53ca2f1e5878d593b9021a5442c9: same HF dataset, `weights/`.
- Old leave-one-specimen-out (LOSO) campaign, archived only: HF `SelimEmirCan/claridi-results` and `SelimEmirCan/claridi-checkpoints`. Not used any more.

## Setup
1. Choose a root with ≥150 GB free on a non-system disk; ≥300 GB if checkpoints are not pruned. Every path in the scripts is hardcoded to `/local/emir/ClariDi`: either create that root, or replace it everywhere in `cluster/*.sh`, `cluster/website/*.py` and `cluster/analysis/*.py`. Keep TMPDIR and TORCH_HOME on that disk.
2. Create the conda env: Python 3.10, torch 2.1.2+cu121, torchvision 0.16.2, pytorch-lightning 1.9.3, omegaconf 2.3.1, lpips 0.1.4, scikit-image 0.25.2, dominate, huggingface_hub. Use `cluster/requirements_chatgarment_freeze.txt` as the full reference. The scripts activate an env named `chatgarment`, so name it that or edit `run_kfold.sh`.
3. Layout under the root:
   - `repo/` (the clone)
   - `data/bbdm256/{train/A,train/B,manifest.csv}`
   - `data/splits/model_design_exp_split_v2.csv` (copy from `cluster/splits/`)
   - `weights/`
   - Copy `cluster/run_kfold.sh`, `run_until.sh`, `run_v2_queue.sh`, `run_v2b_queue.sh` and `prune_v2b.sh` to the root.
   - The configs in `repo/BBDM/configs/` reference cluster paths: check and fix them.
4. Verify:
   - `SCHEME=spatial ./run_kfold.sh dry --split_file <root>/data/splits/model_design_exp_split_v2.csv --results_root <scratch>` must print "leakage assertion PASSED for all 5 folds". The train/val/test/excluded counts per fold should be 453/146/151/3, 443/153/148/9, 449/139/157/8, 452/152/143/6, 455/141/154/3.
   - Run the GAN driver dry run too: `cd repo/baselines; python kfold_grouped_baselines.py --baseline pix2pix --data_root <root>/data/bbdm256 --scheme spatial --split_file <v2 csv> --out_root <scratch> --dry_run`.
   - Before the full launch, do a 1-epoch smoke run of one diffusion target and one GAN on one GPU.

## The experiment (decided)
- Split v2: spatially blocked 5-fold. Each piece is cut into 5 bands along its long axis. In fold k, band k is test and the neighbouring band is val. D part 0 is banded per tile, with tiles that overlap test/val pixels excluded from that fold. Every tile is tested exactly once. 5 folds is final (not 10). This tests within-specimen interpolation, not generalization to new specimens.
- 45 runs, 5 folds each, single GPU per run, results in `results_spatial_v2/`, outputs in `deliverables_spatial_v2/<experiment>/samples/fold_k/<tile>_gen{0..4}.png` (GANs: gen0 only):
  - Diffusion (`run_v2_queue.sh` jobs): Ours `sp_primary`, Vanilla `sp_stock_vqgan`, + B `sp_refB_unstained`, + specimen label (oracle) `sp_specimen_cond`, + A stained refs (extra input) `sp_refA_stained`. About 2 h per run.
  - Then (`run_v2b_queue.sh`): pixel-space BBDM `pixel_space` (~7–9 h), cWGAN `cwgan` (~6–8 h), trainable encoder `trainable_encoder` (~3–4 h), pix2pix `pix2pix` (~3 h). Longest first.
  - Check `cluster/HANDOFF.md` for which of these already finished on the old machine. If their outputs were uploaded, don't rerun them.
- Pruning (`prune_v2b.sh`): after each run's outputs exist, keep only `top_model_epoch_*.pth` + `config.yaml` (diffusion) or `latest_net_G.pth` (GANs).
- Results protocol (fixed in advance, `cluster/analysis/metrics_all_draws.py`; repoint it to `deliverables_spatial_v2` and the v2 split):
  - Score every one of the 5 draws separately against C&SF at 256 px: LPIPS, PSNR, SSIM, |Δ green share|. No best-of-n, no hand-picked examples.
  - Average per specimen, then the unweighted mean over the 11 specimens. Report mean ± SD across the 5 draws.
  - Secondary breakdowns: tile-micro, brain/heart, per specimen.
  - The metrics are context; blind visual grading is the main evidence.
- Website (unlisted): https://selim-emir-can.github.io/ClariDi/. Source is in the repo `selim-emir-can.github.io/ClariDi/`; build scripts are in `cluster/website/`. Rebuild with v2 outputs (`build_site.py`: set GEN, SPLIT, LPIPS to v2), then commit and push.
- At the end: write a summary for the manuscript session on another machine. Cover the methods, the split and its known limitation (a test band touches a training band along one edge), the D part 0 handling, the protocol and the tables, and label the specimen-label and stained-reference variants as oracle / extra-input ablations.
