"""Upload per-fold checkpoints to SelimEmirCan/claridi-checkpoints (public, gated=manual).

  python upload_checkpoints.py --folds 0-5 --dry     # list what would go, with sizes
  python upload_checkpoints.py --folds 0-5           # upload, skipping files already present
  python upload_checkpoints.py --folds 0-5 --verify  # compare remote size against local, no upload
  python upload_checkpoints.py --v2 --folds 0-4 [--dry|--verify]   # split v2 -> spatial_v2/<experiment>/fold_<k>/

Remote layout:  <experiment>/fold_<k>/<filename>
Diffusion experiments send the selected checkpoint (top_model_epoch_*.pth) + config.yaml.
GAN baselines send latest_net_G.pth — the only weights test.py loads.
Nothing is deleted here; deletion is a separate, explicit step after --verify passes.
"""
import argparse, os, sys, glob, hashlib

ROOT = "/local/emir/ClariDi"
REPO = "SelimEmirCan/claridi-checkpoints"

# experiment -> (results dir template, kind)
BBDM = {
    "claridi_primary":     "results/ClariGAN_stratified_fold_{k}_specimen_grouped/LBBDM-f16/checkpoint",
    "claridi_stock_vqgan": "results/ClariGAN_stockVQGAN_fold_{k}_stockVQGAN_leakfree/LBBDM-f16/checkpoint",
    "trainable_encoder":   "results/ClariGAN_stratified_fold_{k}_specimen_grouped_trainable_encoder/LBBDM-f16/checkpoint",
    "pixel_space":         "results/ClariGAN_stratified_pixel_BBDM_matchUNet_fold_{k}_specimen_grouped_pixel/*/checkpoint",
    "claridi_primary_seed5678": "results/ClariGAN_stratified_fold_{k}_specimen_grouped_seed5678/LBBDM-f16/checkpoint",
    "claridi_primary_seed9012": "results/ClariGAN_stratified_fold_{k}_specimen_grouped_seed9012/LBBDM-f16/checkpoint",
}
GAN = ["pix2pix", "cwgan", "unet_l1"]
# split v2 (--v2): remote spatial_v2/<experiment>/fold_<k>/..., experiment names as in claridi-results/spatial_v2/
BBDM_V2 = {
    "sp_stock_vqgan":    "results_spatial_v2/ClariGAN_stockVQGAN_fold_{k}_sp2_stock/LBBDM-f16/checkpoint",
    "sp_primary":        "results_spatial_v2/ClariGAN_stratified_fold_{k}_sp2_primary/LBBDM-f16/checkpoint",
    "sp_specimen_cond":  "results_spatial_v2/ClariGAN_stratified_fold_{k}_sp2_speccond/LBBDM-f16/checkpoint",
    "sp_refA_stained":   "results_spatial_v2/ClariGAN_stratified_fold_{k}_sp2_refA/LBBDM-f16/checkpoint",
    "sp_refB_unstained": "results_spatial_v2/ClariGAN_stratified_fold_{k}_sp2_refB/LBBDM-f16/checkpoint",
    "trainable_encoder": "results_spatial_v2/ClariGAN_stratified_fold_{k}_sp2_encoder/LBBDM-f16/checkpoint",
    "pixel_space":       "results_spatial_v2/ClariGAN_stratified_pixel_BBDM_matchUNet_fold_{k}_sp2_pixel/*/checkpoint",
}
GAN_V2 = ["pix2pix", "cwgan"]


def plan_v2(folds):
    items = []
    for exp, tmpl in BBDM_V2.items():
        for k in folds:
            ds = glob.glob(os.path.join(ROOT, tmpl.format(k=k)))
            assert len(ds) == 1, f"{exp} fold {k}: {ds}"
            tops = sorted(glob.glob(os.path.join(ds[0], "top_model_epoch_*.pth")))
            assert len(tops) == 1 and os.path.exists(os.path.join(ds[0], "config.yaml")), f"{exp} fold {k}: {tops}"
            for f in tops + [os.path.join(ds[0], "config.yaml")]:
                items.append((f, f"spatial_v2/{exp}/fold_{k}/{os.path.basename(f)}", os.path.getsize(f)))
    for b in GAN_V2:
        for k in folds:
            f = os.path.join(ROOT, "baselines_out_v2", b, "checkpoints", f"clarigan_{b}_fold_{k}_sp2_{b}", "latest_net_G.pth")
            assert os.path.exists(f), f
            items.append((f, f"spatial_v2/{b}/fold_{k}/latest_net_G.pth", os.path.getsize(f)))
    return items


def plan(folds):
    """Return [(local_path, remote_path, size)] for the requested folds."""
    items = []
    for exp, tmpl in BBDM.items():
        for k in folds:
            for d in glob.glob(os.path.join(ROOT, tmpl.format(k=k))):
                for pat in ("top_model_epoch_*.pth", "config.yaml"):
                    for f in sorted(glob.glob(os.path.join(d, pat))):
                        items.append((f, f"{exp}/fold_{k}/{os.path.basename(f)}", os.path.getsize(f)))
    for b in GAN:
        for k in folds:
            f = os.path.join(ROOT, "baselines_out", b, "checkpoints",
                             f"clarigan_{b}_fold_{k}_specimen_grouped", "latest_net_G.pth")
            if os.path.exists(f):
                items.append((f, f"{b}/fold_{k}/latest_net_G.pth", os.path.getsize(f)))
    return items


def remote_sizes(api):
    from huggingface_hub import HfApi
    try:
        sib = api.repo_info(REPO, repo_type="model", files_metadata=True).siblings
        return {s.rfilename: (s.size or 0) for s in sib}
    except Exception as e:
        print("could not list remote:", e); return {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", required=True, help="e.g. 0-5 or 0,1,2")
    p.add_argument("--dry", action="store_true"); p.add_argument("--verify", action="store_true")
    p.add_argument("--v2", action="store_true", help="split-v2 checkpoints -> spatial_v2/<experiment>/fold_<k>/")
    a = p.parse_args()
    if "-" in a.folds:
        lo, hi = a.folds.split("-"); folds = list(range(int(lo), int(hi) + 1))
    else:
        folds = [int(x) for x in a.folds.split(",")]

    items = plan_v2(folds) if a.v2 else plan(folds)
    total = sum(s for _, _, s in items)
    print(f"{len(items)} files, {total/1e9:.1f} GB, folds {folds}", flush=True)

    from huggingface_hub import HfApi
    api = HfApi()
    have = remote_sizes(api)

    if a.dry:
        for l, r, s in items:
            mark = "present" if have.get(r) == s else ("SIZE MISMATCH" if r in have else "to upload")
            print(f"  {s/1e6:8.1f} MB  {r:60s} {mark}")
        return
    if a.verify:
        ok = bad = miss = 0
        for l, r, s in items:
            if r not in have: print(f"  MISSING  {r}"); miss += 1
            elif have[r] != s: print(f"  MISMATCH {r}: local {s} remote {have[r]}"); bad += 1
            else: ok += 1
        print(f"\nverify: {ok} match, {bad} size mismatch, {miss} missing (of {len(items)})")
        sys.exit(0 if (bad == 0 and miss == 0) else 1)

    # HF rate-limits repository commits (128/hour). Checkpoints are large, so commit in small
    # batches: few enough commits to stay under the limit, small enough to retry cheaply.
    from huggingface_hub import CommitOperationAdd
    todo = [(l, r, s) for l, r, s in items if have.get(r) != s]
    skipped = len(items) - len(todo)
    if skipped: print(f"{skipped} already present, skipping", flush=True)
    if not todo:
        print("ALL UPLOADS FINISHED (nothing to do)", flush=True); return
    BATCH = 12                     # ~28 GB per commit: 66 files -> 6 commits, cheap to retry
    done_b = 0
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for l, r, _ in chunk]
        gb = sum(s for _, _, s in chunk) / 1e9
        print(f"commit {b//BATCH + 1}: {len(ops)} files, {gb:.1f} GB", flush=True)
        api.create_commit(repo_id=REPO, repo_type="model", operations=ops,
                          commit_message=f"add {len(ops)} checkpoint files")
        done_b += sum(s for _, _, s in chunk)
        print(f"  committed, {done_b/1e9:.1f}/{sum(s for _,_,s in todo)/1e9:.1f} GB", flush=True)
    print("ALL UPLOADS FINISHED", flush=True)


if __name__ == "__main__":
    main()
