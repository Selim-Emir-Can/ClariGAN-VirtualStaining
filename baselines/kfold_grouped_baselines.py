"""Specimen-grouped k-fold driver for the GAN baselines (pix2pix, cWGAN).

Replaces k-fold_validation_pix2pix.py / k-fold_validation_cwgan.py, which each carried a
verbatim copy of the leaky patch-level splitter (including the wrong Z -> H merge).
This driver imports the SAME splitter the BBDM experiments use (BBDM/specimen_kfold.py),
so every model in the comparison table is trained and tested on identical folds.

It writes per-fold list files of (input_path, target_path) pairs and drives the upstream
train.py / test.py via subprocess, exactly as the old drivers did.

  python kfold_grouped_baselines.py --baseline pix2pix --data_root <dir> --dry_run
  python kfold_grouped_baselines.py --baseline pix2pix --data_root <dir> --gpu 4 --folds 0
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BBDM_DIR = os.path.abspath(os.path.join(HERE, "..", "BBDM"))
sys.path.insert(0, BBDM_DIR)
from specimen_kfold import (load_manifest, make_folds, assert_no_leakage,  # noqa: E402
                            fold_table, format_fold_table, save_fold_assignments)

BASELINES = {
    "pix2pix": dict(dir=os.path.join(HERE, "pytorch-CycleGAN-and-pix2pix"),
                    epoch_flags=("--n_epochs", "--n_epochs_decay"), gpu_flag=None,
                    batch_size=1),
    "cwgan":   dict(dir=os.path.join(HERE, "cwgan"),
                    epoch_flags=("--niter", "--niter_decay"), gpu_flag="--gpu_ids",
                    batch_size=4),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", choices=list(BASELINES), required=True)
    p.add_argument("--data_root", required=True, help="dir with train/A, train/B, manifest.csv")
    p.add_argument("--manifest", default=None)
    p.add_argument("--scheme", choices=["loso", "grouped"], default="loso")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--folds", default=None, help="comma-separated fold indices (default all)")
    p.add_argument("--gpu", default="0", help="single GPU index")
    p.add_argument("--out_root", default=None,
                   help="checkpoints/results/lists go under here; default <project>/baselines_out/<baseline>")
    p.add_argument("--n_epochs", type=int, default=25, help="initial-lr epochs")
    p.add_argument("--n_epochs_decay", type=int, default=25, help="linear-decay epochs (total 50, matches BBDM)")
    p.add_argument("--ngf", type=int, default=144, help="generator width; 144 ~ 273M params, near BBDM's 259M trainable")
    p.add_argument("--netG", default="unet_256")
    p.add_argument("--batch_size", type=int, default=None, help="default: 1 for pix2pix, 4 for cwgan")
    p.add_argument("--tag", default="specimen_grouped")
    p.add_argument("--deliverables_root", default="/local/emir/ClariDi/deliverables")
    p.add_argument("--experiment", default=None, help="deliverable dir name; default = --baseline")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_train", action="store_true", help="only run test.py on existing checkpoints")
    return p.parse_args()


def write_list(pairs, path):
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(f"{a},{b}\n")
    return path


def export_bare_outputs(a, records, fold, images_dir, t_train, t_test, bs):
    """Copy the generator's bare output for every test tile into the deliverable layout:
    <deliverables>/<experiment>/samples/fold_<k>/<tile_id>_gen0.png  (one deterministic
    output per tile, so it is always gen0). test.py names files <input_stem>_fake_B.png."""
    import time, shutil, csv
    from PIL import Image
    exp = a.experiment or a.baseline
    k = fold["fold"]
    out = os.path.join(a.deliverables_root, exp, "samples", f"fold_{k}")
    os.makedirs(out, exist_ok=True)
    by_stem = {os.path.splitext(r["input_filename"])[0]: r for r in records}
    n = 0
    for fn in os.listdir(images_dir):
        if not fn.endswith("_fake_B.png"):
            continue
        r = by_stem[fn[:-len("_fake_B.png")]]
        assert r["specimen"] in fold["test_specimens"], f"{fn}: not a test-specimen tile"
        dst = os.path.join(out, f"{r['tile_id']}_gen0.png")
        im = Image.open(os.path.join(images_dir, fn)).convert("RGB")
        assert im.size == (256, 256), f"{fn}: unexpected size {im.size}"
        im.save(dst)
        n += 1
    assert n == len(fold["test"]), f"fold {k}: exported {n} outputs, expected {len(fold['test'])}"
    t_test = time.time() - t_test
    with open(os.path.join(a.deliverables_root, exp, f"timing_fold_{k}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "fold", "n_test_tiles", "train_wall_s", "test_wall_s", "per_patch_infer_s_approx",
                    "epochs", "batch_size", "gpu", "n_gpus_train", "outputs_per_tile"])
        w.writerow([exp, k, n, round(t_train, 1), round(t_test, 1), round(t_test / max(n, 1), 4),
                    a.n_epochs + a.n_epochs_decay, bs, a.gpu, 1, 1])
    with open(os.path.join(a.deliverables_root, exp, f"seeds_fold_{k}.json"), "w") as f:
        f.write('{"note": "GAN baseline: one deterministic output per tile (gen0); no sampling seed"}\n')
    print(f"fold {k}: exported {n} bare outputs -> {out}", flush=True)


def main():
    a = parse_args()
    B = BASELINES[a.baseline]
    out_root = a.out_root or os.path.join(os.path.dirname(os.path.dirname(a.data_root.rstrip("/"))), "baselines_out", a.baseline)
    lists_dir = os.path.join(out_root, "fold_lists")
    ckpt_dir = os.path.join(out_root, "checkpoints")
    results_dir = os.path.join(out_root, "results")
    for d in (lists_dir, ckpt_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    records = load_manifest(a.manifest or os.path.join(a.data_root, "manifest.csv"), a.data_root)
    folds = make_folds(records, a.scheme, a.n_folds)
    print(f"{a.baseline}: {len(records)} patches, {len(folds)} folds ({a.scheme})")
    print(format_fold_table(fold_table(folds, records)))
    for f in folds:
        assert_no_leakage(f, records)
    print(f"\nleakage assertion PASSED for all {len(folds)} folds")
    save_fold_assignments(folds, records, out_root)
    print("fold assignment table saved to", os.path.join(out_root, "fold_assignments.csv"))
    if a.dry_run:
        return

    env = dict(os.environ)
    if B["gpu_flag"] is None:            # pix2pix picks cuda:0; select the card via visibility
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    gpu_args = [] if B["gpu_flag"] is None else [B["gpu_flag"], str(a.gpu)]
    bs = a.batch_size if a.batch_size is not None else B["batch_size"]
    wanted = None if a.folds is None else {int(x) for x in a.folds.split(",")}

    for f in folds:
        if wanted is not None and f["fold"] not in wanted:
            continue
        k = f["fold"]
        name = f"clarigan_{a.baseline}_fold_{k}_{a.tag}"
        tr = write_list(f["train"], os.path.join(lists_dir, f"fold_{k}_train.txt"))
        va = write_list(f["val"], os.path.join(lists_dir, f"fold_{k}_val.txt"))
        te = write_list(f["test"], os.path.join(lists_dir, f"fold_{k}_test.txt"))
        print(f"\n=== {a.baseline} fold {k}: test={f['test_specimens']} val={f['val_specimens']} "
              f"train={len(f['train'])} pairs ===", flush=True)

        common = ["--name", name, "--checkpoints_dir", ckpt_dir, "--model", "pix2pix",
                  "--dataset_mode", "bbdm_aligned", "--direction", "AtoB",
                  "--netG", a.netG, "--ngf", str(a.ngf), "--load_size", "256", "--crop_size", "256",
                  "--input_nc", "3", "--output_nc", "3", "--display_id", "0"] + gpu_args   # no visdom/wandb UI
        import time
        t_train = time.time()
        if not a.skip_train:
            cmd = [sys.executable, "train.py", "--dataroot", tr, "--val_dataroot", va,
                   "--batch_size", str(bs),
                   B["epoch_flags"][0], str(a.n_epochs), B["epoch_flags"][1], str(a.n_epochs_decay)] + common
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=B["dir"], env=env)
        t_train = time.time() - t_train
        t_test = time.time()
        cmd = [sys.executable, "test.py", "--dataroot", te, "--results_dir", results_dir,
               "--phase", "test", "--num_test", "100000"] + common
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=B["dir"], env=env)
        print(f"fold {k} done -> {os.path.join(results_dir, name, 'test_latest')}", flush=True)
        export_bare_outputs(a, records, f, os.path.join(results_dir, name, "test_latest", "images"),
                            t_train, t_test, bs)


if __name__ == "__main__":
    main()
