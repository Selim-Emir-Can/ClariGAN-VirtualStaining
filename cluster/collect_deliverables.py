"""Assemble the claridi-results deliverable tree and (optionally) upload it to HF.

  python collect_deliverables.py --check                 # validate what is on disk, print sizes
  python collect_deliverables.py --upload                # upload to SelimEmirCan/claridi-results (private)
  python collect_deliverables.py --upload --experiments claridi_primary   # subset
"""
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time

ROOT = "/local/emir/ClariDi"
DELIV = os.path.join(ROOT, "deliverables")
REPO_ID = "SelimEmirCan/claridi-results"
PY = "/home/emir/miniconda3/envs/chatgarment/bin/python"

EXPERIMENTS = {   # experiment name -> (protocol, run_kfold target, notes)
    "claridi_primary":            ("LOSO-11", "claridi",       "L-BBDM f16, fine-tuned VQGAN (frozen; saw all specimens)"),
    "claridi_stock_vqgan":        ("LOSO-11", "stock",         "L-BBDM f16, stock ImageNet VQGAN (frozen; leak-free end to end)"),
    "trainable_encoder":          ("LOSO-11", "encoder",       "VQGAN encoder trainable, fine-tuned init (same as primary); isolates frozen vs trainable"),
    "pixel_space":                ("TBD",     "pixel",         "pure pixel-space BBDM, no VQGAN"),
    "pix2pix":                    ("LOSO-11", "pix2pix",       "GAN baseline, ngf=144, one deterministic output per tile"),
    "cwgan":                      ("LOSO-11", "cwgan",         "GAN baseline, ngf=144, one output per tile"),
    "claridi_primary_seed5678":   ("LOSO-11", "claridi",       "primary re-trained with training seed 5678; sampling seeds unchanged"),
    "claridi_primary_seed9012":   ("LOSO-11", "claridi",       "primary re-trained with training seed 9012; sampling seeds unchanged"),
    "unet_l1":                    ("LOSO-11", "unet_l1",       "deterministic U-Net (pix2pix generator, unet_256 ngf=144) trained with L1 only; eval mode at inference"),
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def du(path):
    tot = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            tot += os.path.getsize(os.path.join(dp, fn))
    return tot


def versions():
    code = ("import torch,torchvision,numpy,PIL,skimage,lpips,yaml;import json;"
            "print(json.dumps({'python':__import__('sys').version.split()[0],'torch':torch.__version__,"
            "'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'cudnn':str(torch.backends.cudnn.version()),"
            "'numpy':numpy.__version__,'pillow':PIL.__version__,'scikit-image':skimage.__version__}))")
    return json.loads(subprocess.check_output([PY, "-c", code], env={**os.environ, "PYTHONWARNINGS": "ignore"}).decode().strip().splitlines()[-1])


def build_metadata(experiments):
    manifest = list(csv.DictReader(open(os.path.join(ROOT, "data/bbdm256/manifest.csv"))))
    fa = list(csv.DictReader(open(os.path.join(ROOT, "results/fold_assignments.csv"))))
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "repo": "https://github.com/Selim-Emir-Can/ClariGAN-VirtualStaining",
        "git_commit": subprocess.check_output(["git", "-C", os.path.join(ROOT, "repo"), "rev-parse", "HEAD"]).decode().strip(),
        "git_dirty_note": "cluster changes on top of that commit: specimen_kfold.py, kfold_grouped.py, eval_fold.py, "
                          "vqgan_ceiling_pngs.py, baselines/kfold_grouped_baselines.py, config path edits, BaseRunner num_workers",
        "dataset": {"hf": "SelimEmirCan/claridi", "n_tiles": len(manifest), "n_specimens": len({r["specimen"] for r in manifest}),
                    "grouping": "specimen column (Z is part of D; Hpart1 is part of H)", "resolution": "256x256, Resize then ToTensor, bit-identical to on-the-fly"},
        "split": {"protocol": "leave-one-specimen-out, 11 folds; val = one whole training specimen of the same tissue",
                  "fold_assignments": "fold_assignments.csv", "leakage_assertion": "assert_no_leakage passed for all folds"},
        "seeds": {"training_seed": "1234 (claridi_primary_seed5678 / _seed9012 use 5678 / 9012); GAN baselines 1234", "generation_rule": "torch.manual_seed(1234 + gen_idx) immediately before each generation",
                  "gen_seed": {j: 1234 + j for j in range(5)},
                  "cudnn": {"bbdm_training": {"deterministic": True, "benchmark": False, "source": "main.set_random_seed"},
                            "bbdm_sampling": {"deterministic": True, "benchmark": False, "source": "eval_fold.py"},
                            "gan_training_and_inference": {"deterministic": True, "benchmark": False, "source": "train.py/test.py patch"}},
                  "gan_inference": {"pix2pix": "eval mode with dropout re-enabled (batch-norm fixed), inference seed 1234",
                                    "cwgan": "fork default (no eval flag): train-mode BN and dropout, as originally run",
                                    "unet_l1": "eval mode: dropout off, batch-norm fixed"}},
        "sampling": {"sample_num": 5, "sample_steps": 200, "sample_type": "linear", "eta": 1.0},
        "training": {"batch_size": 8, "accumulate_grad_batches": 4, "effective_batch": 32, "n_gpus_per_fold": 1,
                     "n_epochs": 50, "checkpoint_selection": "lowest validation loss (top_model_epoch_*.pth)",
                     "augmentation": "online, train split only, on the 256px tensor: hflip, vflip, rotation<=180deg, "
                                     "translation<=5%, resized crop scale 0.8-1.0, same params for input+target; per epoch each "
                                     "pair is seen ~4x raw + ~4x augmented (the 8x oversampling is not 8x augmented)"},
        "hardware": {"gpu": "NVIDIA RTX A6000 49GB", "driver": "560.28.03", "host": platform.node()},
        "software": versions(),
        "checkpoints": {"vqgan_finetuned": {"file": "epoch=000022.ckpt", "md5": md5(os.path.join(ROOT, "weights/epoch=000022.ckpt")),
                                            "tissue_exposed": "yes, all 11 specimens"},
                        "vqgan_stock": {"file": "vqgan_imagenet_f16_16384_stock.ckpt",
                                        "md5": md5(os.path.join(ROOT, "weights/vqgan_imagenet_f16_16384_stock.ckpt")),
                                        "tissue_exposed": "no, ImageNet only"}},
        "experiments": {},
    }
    for e in experiments:
        d = os.path.join(DELIV, e)
        proto, target, note = EXPERIMENTS[e]
        folds = sorted(int(x.split("_")[-1]) for x in os.listdir(os.path.join(d, "samples"))) if os.path.isdir(os.path.join(d, "samples")) else []
        meta["experiments"][e] = {"protocol": proto, "run_target": target, "note": note, "folds_present": folds,
                                  "n_sample_pngs": sum(len(os.listdir(os.path.join(d, "samples", f"fold_{k}"))) for k in folds)}
    return meta


def check(experiments):
    manifest = {r["tile_id"] for r in csv.DictReader(open(os.path.join(ROOT, "data/bbdm256/manifest.csv")))}
    ok = True
    for e in experiments:
        d = os.path.join(DELIV, e, "samples")
        if not os.path.isdir(d):
            print(f"{e}: no samples yet"); continue
        seen = {}
        for fk in sorted(os.listdir(d)):
            k = int(fk.split("_")[-1])
            for fn in os.listdir(os.path.join(d, fk)):
                tid, gen = fn[:-4].rsplit("_gen", 1)
                seen.setdefault(tid, set()).add((k, int(gen)))
        multi = {t: s for t, s in seen.items() if len({k for k, _ in s}) > 1}
        gens = {len({g for _, g in s}) for s in seen.values()}
        print(f"{e}: folds={sorted({k for s in seen.values() for k,_ in s})} tiles={len(seen)}/753 "
              f"gens_per_tile={sorted(gens)} unknown_tiles={len(set(seen)-manifest)} tiles_in_2_folds={len(multi)} "
              f"size={du(os.path.join(DELIV,e))/1e9:.2f} GB")
        ok &= not multi and not (set(seen) - manifest)
    for lab in ("finetuned", "stock"):
        p = os.path.join(DELIV, "ceilings", lab)
        n = len(os.listdir(p)) if os.path.isdir(p) else 0
        print(f"ceilings/{lab}: {n}/753 PNGs")
    print(f"total deliverables size: {du(DELIV)/1e9:.2f} GB")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true"); ap.add_argument("--upload", action="store_true")
    ap.add_argument("--experiments", default=None, help="comma-separated subset")
    a = ap.parse_args()
    exps = a.experiments.split(",") if a.experiments else [e for e in EXPERIMENTS if os.path.isdir(os.path.join(DELIV, e))]
    os.makedirs(DELIV, exist_ok=True)
    for src, dst in (("results/fold_assignments.csv", "fold_assignments.csv"), ("data/bbdm256/manifest.csv", "manifest.csv")):
        s, t = os.path.join(ROOT, src), os.path.join(DELIV, dst)
        if os.path.exists(s) and (not os.path.exists(t) or open(s).read() != open(t).read()):
            open(t, "w").write(open(s).read())
    if not os.path.islink(os.path.join(DELIV, "data_256")) and not os.path.isdir(os.path.join(DELIV, "data_256")):
        os.symlink(os.path.join(ROOT, "data/bbdm256"), os.path.join(DELIV, "data_256"))
    json.dump(build_metadata(exps), open(os.path.join(DELIV, "run_metadata.json"), "w"), indent=1)
    ok = check(exps)
    if a.upload:
        if not ok:
            sys.exit("integrity check failed; not uploading")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
        for item in ["fold_assignments.csv", "manifest.csv", "run_metadata.json"]:
            api.upload_file(path_or_fileobj=os.path.join(DELIV, item), path_in_repo=item, repo_id=REPO_ID, repo_type="dataset")
        for folder in ["data_256", "ceilings"] + exps:
            p = os.path.realpath(os.path.join(DELIV, folder))
            if os.path.isdir(p):
                print("uploading", folder, flush=True)
                api.upload_folder(folder_path=p, path_in_repo=folder, repo_id=REPO_ID, repo_type="dataset",
                                  commit_message=f"add {folder}")
        print("uploaded to", REPO_ID)


if __name__ == "__main__":
    main()
