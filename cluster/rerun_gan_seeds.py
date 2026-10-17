"""Five inference draws for the GAN baselines on split v2 (Michael, Oct 2026: 'rerun pix2pix and cWGAN inference with
five seeds'). Inference only, on the existing per-fold weights (latest_net_G.pth); no retraining.

Draw j uses inference seed 1234 + j (j = 0..4), the same convention as the diffusion models.
  pix2pix: --eval --dropout_at_inference --inference_seed s  (batch-norm fixed, dropout on; as originally run)
  cWGAN:   train mode as originally run (dropout + per-image batch-norm statistics), now seeded: --inference_seed s
Writes deliverables_spatial_v2_gan5/<exp>/samples/fold_k/<tile_id>_gen{j}.png (+ seeds/timing per fold);
the original single-draw outputs in deliverables_spatial_v2/ are left untouched.

  python rerun_gan_seeds.py --gpu 0 [--methods pix2pix cwgan] [--folds 0 1 2 3 4]
"""
import argparse, csv, json, os, shutil, subprocess, sys, time
from PIL import Image

R = "/local/emir/ClariDi"
BASE = f"{R}/repo/baselines"
SPLIT = f"{R}/data/splits/model_design_exp_split_v2.csv"
OUT = f"{R}/deliverables_spatial_v2_gan5"
M = {
    "pix2pix": dict(dir=f"{BASE}/pytorch-CycleGAN-and-pix2pix", tag="sp2_pix2pix", gpu_flag=False,
                    extra=["--eval", "--dropout_at_inference"]),
    "cwgan":   dict(dir=f"{BASE}/cwgan", tag="sp2_cwgan", gpu_flag=True, extra=[]),
}
SEEDS = [1234 + j for j in range(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--methods", nargs="+", default=["pix2pix", "cwgan"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    man = {}
    with open(f"{R}/data/bbdm256/manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): v.strip() for k, v in r.items()}; man[os.path.splitext(r["input_filename"])[0]] = r
    split = {r["tile_id"]: r for r in csv.DictReader(open(SPLIT))}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    for m in a.methods:
        c = M[m]; bo = f"{R}/baselines_out_v2/{m}"
        for k in a.folds:
            name = f"clarigan_{m}_fold_{k}_{c['tag']}"
            assert os.path.exists(f"{bo}/checkpoints/{name}/latest_net_G.pth"), name
            test_list = f"{bo}/fold_lists/fold_{k}_test.txt"
            want = {t for t, r in split.items() if r[f"role_f{k}"] == "test"}
            out = f"{OUT}/{m}/samples/fold_{k}"; os.makedirs(out, exist_ok=True)
            t0 = time.time()
            for j, s in enumerate(SEEDS):
                res = f"{bo}/results_seeds/seed_{s}"
                cmd = [sys.executable, "test.py", "--dataroot", test_list, "--results_dir", res, "--phase", "test",
                       "--num_test", "100000", "--name", name, "--checkpoints_dir", f"{bo}/checkpoints",
                       "--model", "pix2pix", "--dataset_mode", "bbdm_aligned", "--direction", "AtoB",
                       "--netG", "unet_256", "--ngf", "144", "--load_size", "256", "--crop_size", "256",
                       "--input_nc", "3", "--output_nc", "3", "--inference_seed", str(s)] + c["extra"]
                if c["gpu_flag"]: cmd += ["--gpu_ids", "0"]       # card chosen via CUDA_VISIBLE_DEVICES
                subprocess.run(cmd, check=True, cwd=c["dir"], env=env, stdout=subprocess.DEVNULL)
                imgs = f"{res}/{name}/test_latest/images"; got = set()
                for fn in os.listdir(imgs):
                    if not fn.endswith("_fake_B.png"): continue
                    t = man[fn[:-len("_fake_B.png")]]["tile_id"]
                    assert t in want, f"{m} fold {k}: {t} is not a test tile of this fold"
                    im = Image.open(f"{imgs}/{fn}").convert("RGB"); assert im.size == (256, 256)
                    im.save(f"{out}/{t}_gen{j}.png"); got.add(t)
                assert got == want, f"{m} fold {k} seed {s}: {len(got)} outputs, expected {len(want)}"
                print(f"{time.strftime('%F %T')} {m} fold {k} gen{j} (seed {s}): {len(got)} tiles", flush=True)
            dt = time.time() - t0
            json.dump({"gen_seed": {str(j): s for j, s in enumerate(SEEDS)}, "inference_flags": c["extra"],
                       "weights": f"{name}/latest_net_G.pth", "note": "inference-only redraws on the trained weights"},
                      open(f"{OUT}/{m}/seeds_fold_{k}.json", "w"), indent=1)
            with open(f"{OUT}/{m}/timing_fold_{k}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["experiment", "fold", "n_test_tiles", "test_wall_s_5draws", "gpu", "outputs_per_tile"])
                w.writerow([m, k, len(want), round(dt, 1), a.gpu, 5])
            print(f"{time.strftime('%F %T')} {m} fold {k} DONE ({dt:.0f} s)", flush=True)
    print(f"{time.strftime('%F %T')} GAN5_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
