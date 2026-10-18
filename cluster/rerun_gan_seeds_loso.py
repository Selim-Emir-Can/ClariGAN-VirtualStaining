"""Five inference draws for the LOSO GAN baselines (11 folds), on the trained weights from HF
SelimEmirCan/claridi-checkpoints (<method>/fold_<k>/latest_net_G.pth). Inference only; no retraining.

Same conventions as rerun_gan_seeds.py (split v2): draw j = inference seed 1234 + j.
  pix2pix: --eval --dropout_at_inference --inference_seed s ; cWGAN: train mode as originally run, now seeded.
Folds are rebuilt with specimen_kfold.make_folds(scheme="loso") and checked against the fold table on HF.
Weights are streamed one fold at a time into /local and deleted after that fold's 5 draws (disk is near full).
Writes deliverables_loso_gan5/<method>/samples/fold_k/<tile_id>_gen{j}.png (+ seeds/timing per fold).

  HF_HOME=/local/emir/tmp/hf_home python rerun_gan_seeds_loso.py --gpu 3 [--methods pix2pix cwgan] [--folds 0 ... 10]
"""
import argparse, csv, json, os, shutil, subprocess, sys, time
from PIL import Image

R = "/local/emir/ClariDi"
BASE = f"{R}/repo/baselines"
sys.path.insert(0, f"{R}/repo/BBDM")
from specimen_kfold import load_manifest, make_folds  # noqa: E402

OUT = f"{R}/deliverables_loso_gan5"
WORK = "/local/emir/tmp/loso_gan5"
CKPT_REPO = "SelimEmirCan/claridi-checkpoints"
M = {"pix2pix": dict(dir=f"{BASE}/pytorch-CycleGAN-and-pix2pix", gpu_flag=False, extra=["--eval", "--dropout_at_inference"]),
     "cwgan":   dict(dir=f"{BASE}/cwgan", gpu_flag=True, extra=[])}
SEEDS = [1234 + j for j in range(5)]


def hf_fold_table():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("SelimEmirCan/claridi-results", "fold_assignments.csv", repo_type="dataset", local_dir=f"{WORK}/meta")
    return {int(r["fold"]): r["specimens"].split() for r in csv.DictReader(open(p)) if r["partition"] == "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--methods", nargs="+", default=["pix2pix", "cwgan"])
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(11)))
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    data_root = f"{R}/data/bbdm256"
    records = load_manifest(f"{data_root}/manifest.csv", data_root)
    folds = {f["fold"]: f for f in make_folds(records, "loso")}
    hf = hf_fold_table()
    for k in a.folds:
        assert folds[k]["test_specimens"] == hf[k], f"fold {k}: rebuilt {folds[k]['test_specimens']} vs HF {hf[k]}"
    by_stem = {os.path.splitext(os.path.basename(r["input_path"]))[0]: r for r in records}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    os.makedirs(f"{WORK}/lists", exist_ok=True)
    for m in a.methods:
        c = M[m]
        for k in a.folds:
            f = folds[k]; spec = f["test_specimens"]
            done = f"{OUT}/{m}/timing_fold_{k}.csv"
            if os.path.exists(done): print(f"{m} fold {k}: already done, skipped", flush=True); continue
            test_list = f"{WORK}/lists/fold_{k}_test.txt"
            with open(test_list, "w") as fh:
                for x, y in f["test"]: fh.write(f"{x},{y}\n")
            want = {by_stem[os.path.splitext(os.path.basename(x))[0]]["tile_id"] for x, _ in f["test"]}
            t0 = time.time()
            w = hf_hub_download(CKPT_REPO, f"{m}/fold_{k}/latest_net_G.pth", local_dir=f"{WORK}/ckpt")
            ckdir, name = f"{WORK}/ckpt/{m}", f"fold_{k}"          # test.py loads <ckdir>/<name>/latest_net_G.pth
            out = f"{OUT}/{m}/samples/fold_{k}"; os.makedirs(out, exist_ok=True)
            for j, s in enumerate(SEEDS):
                res = f"{WORK}/results/{m}/seed_{s}"
                cmd = [sys.executable, "test.py", "--dataroot", test_list, "--results_dir", res, "--phase", "test",
                       "--num_test", "100000", "--name", name, "--checkpoints_dir", ckdir,
                       "--model", "pix2pix", "--dataset_mode", "bbdm_aligned", "--direction", "AtoB",
                       "--netG", "unet_256", "--ngf", "144", "--load_size", "256", "--crop_size", "256",
                       "--input_nc", "3", "--output_nc", "3", "--inference_seed", str(s)] + c["extra"]
                if c["gpu_flag"]: cmd += ["--gpu_ids", "0"]
                subprocess.run(cmd, check=True, cwd=c["dir"], env=env, stdout=subprocess.DEVNULL)
                imgs = f"{res}/{name}/test_latest/images"; got = set()
                for fn in os.listdir(imgs):
                    if not fn.endswith("_fake_B.png"): continue
                    r = by_stem[fn[:-len("_fake_B.png")]]
                    assert r["specimen"] in spec, f"{m} fold {k}: {r['tile_id']} not a test-specimen tile"
                    im = Image.open(f"{imgs}/{fn}").convert("RGB"); assert im.size == (256, 256)
                    im.save(f"{out}/{r['tile_id']}_gen{j}.png"); got.add(r["tile_id"])
                assert got == want, f"{m} fold {k} seed {s}: {len(got)} outputs, expected {len(want)}"
                shutil.rmtree(f"{res}/{name}")
            os.remove(w)                                            # weights stay on HF; free /local
            dt = time.time() - t0
            json.dump({"gen_seed": {str(j): s for j, s in enumerate(SEEDS)}, "inference_flags": c["extra"],
                       "weights": f"{CKPT_REPO}:{m}/fold_{k}/latest_net_G.pth", "test_specimens": spec,
                       "note": "inference-only redraws on the trained LOSO weights"},
                      open(f"{OUT}/{m}/seeds_fold_{k}.json", "w"), indent=1)
            with open(done, "w", newline="") as fh:
                wr = csv.writer(fh); wr.writerow(["experiment", "fold", "test_specimens", "n_test_tiles", "wall_s_incl_download", "gpu", "outputs_per_tile"])
                wr.writerow([m, k, " ".join(spec), len(want), round(dt, 1), a.gpu, 5])
            print(f"{time.strftime('%F %T')} {m} fold {k} ({' '.join(spec)}, {len(want)} tiles) DONE ({dt:.0f} s)", flush=True)
    print(f"{time.strftime('%F %T')} LOSO_GAN5_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
