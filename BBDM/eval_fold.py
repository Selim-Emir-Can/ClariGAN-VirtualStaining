"""Per-fold sample generation for the claridi-results deliverable. NO scoring here:
the manuscript side scores the PNGs with its own script.

For one fold: load the fold's top checkpoint, run the model on every held-out test tile,
draw `sample_num` generations per tile with an EXPLICIT per-generation seed
(gen j uses torch seed = --seed + j, so gen 0 is seed 1234 by default and is the
prespecified single generation; generations are never reordered or dropped), and save
every generation as a BARE 256x256 PNG named by the HF tile_id. No titles, no montage,
no overlays.

Outputs, under <out_root>/<experiment>/:
  samples/fold_<k>/<tile_id>_gen<j>.png
  timing_fold_<k>.csv
  seeds_fold_<k>.json       gen_idx -> torch seed
  config.yaml (copied from the checkpoint dir)

  python eval_fold.py --config <yaml> --ckpt <top_model.pth> --data_root <bbdm256> \
      --fold 0 --experiment claridi_primary --out_root <deliverables> --gpu 4
"""
import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import kfold_grouped                      # registers all runners, gives build_config
from utils import get_runner
from runners.utils import get_dataset
from specimen_kfold import load_manifest, make_folds, assert_no_leakage


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="top_model_epoch_*.pth used for reporting")
    p.add_argument("--vqgan_ckpt", default=None)
    p.add_argument("--data_root", required=True, help="pre-resized 256px dataset (bbdm256)")
    p.add_argument("--manifest", default=None)
    p.add_argument("--scheme", choices=["loso", "grouped"], default="loso")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--experiment", required=True, help="e.g. claridi_primary, claridi_stock_vqgan")
    p.add_argument("--out_root", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--sample_num", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tag", default="eval")
    p.add_argument("--n_gpus_train", type=int, default=1, help="recorded in timing.csv only")
    p.add_argument("--max_tiles", type=int, default=None, help="debug: stop after this many tiles")
    p.add_argument("--clip_denoised", action="store_true",
                   help="default False for EVERY experiment: the original k-fold evaluation "
                        "(sample_to_eval_combined_with_uncertainty) hardcoded clip_denoised=False "
                        "and ignored testing.clip_denoised, including for the pixel-space model")
    return p.parse_args()


def to_uint8(t):
    """Identical conversion to runners.utils.save_single_image (to_normal=True)."""
    t = t.detach().clone().mul_(0.5).add_(0.5).clamp_(0, 1.)
    return t.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def main():
    a = parse_args()
    device = torch.device("cpu") if a.gpu == "-1" else torch.device(f"cuda:{a.gpu}")
    records = load_manifest(a.manifest or os.path.join(a.data_root, "manifest.csv"), a.data_root)
    folds = make_folds(records, a.scheme, a.n_folds)
    fold = folds[a.fold]
    assert_no_leakage(fold, records)
    by_input_stem = {os.path.splitext(r["input_filename"])[0]: r for r in records}
    by_target_stem = {os.path.splitext(r["target_filename"])[0]: r for r in records}

    # identical config construction to the training driver
    class A: pass
    da = A(); da.config = a.config; da.vqgan_ckpt = a.vqgan_ckpt; da.results_root = os.path.join(a.out_root, "_runner_scratch")
    da.seed = a.seed; da.skip_train = True; da.gpu_ids = a.gpu; da.port = "0"; da.max_epoch = None; da.max_steps = None
    da.accumulate_grad_batches = None; da.data_root = a.data_root
    cfg = kfold_grouped.build_config(da, f"fold_{a.fold}_{a.tag}")
    cfg.args.train = False
    cfg.training.use_DDP = False
    cfg.training.device = [device]
    cfg.data.test.batch_size = 1
    cfg.data.dataset_type = "custom_aligned"
    clip = bool(a.clip_denoised)          # False unless explicitly requested (matches the original protocol)

    runner = get_runner(cfg.runner, cfg)
    net = runner.initialize_model(cfg)
    state = torch.load(a.ckpt, map_location=device, weights_only=True)
    net.load_state_dict(state["model"])
    net.eval()

    _, _, test_ds = get_dataset(cfg.data, fold["train"], fold["val"], fold["test"])
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    exp_dir = os.path.join(a.out_root, a.experiment)
    samp_dir = os.path.join(exp_dir, "samples", f"fold_{a.fold}")
    os.makedirs(samp_dir, exist_ok=True)
    cfg_src = os.path.join(os.path.dirname(a.ckpt), "config.yaml")
    if os.path.exists(cfg_src) and not os.path.exists(os.path.join(exp_dir, "config.yaml")):
        shutil.copyfile(cfg_src, os.path.join(exp_dir, "config.yaml"))

    rows, times = [], []
    t_fold = time.time()
    with torch.no_grad():
        for (x, x_name), (x_cond, x_cond_name) in loader:
            r = by_input_stem.get(x_cond_name[0]) or by_target_stem.get(x_name[0])
            assert r is not None, f"no manifest row for {x_cond_name[0]}"
            assert r["specimen"] in fold["test_specimens"], "test tile from a non-test specimen"
            gt_path = r["target_path"]
            x_cond = x_cond.to(device)
            for j in range(a.sample_num):
                seed_j = a.seed + j
                torch.manual_seed(seed_j); torch.cuda.manual_seed_all(seed_j)
                if device.type == "cuda": torch.cuda.synchronize(device)
                t0 = time.time()
                out = net.sample(x_cond, clip_denoised=clip)
                if device.type == "cuda": torch.cuda.synchronize(device)
                dt = time.time() - t0
                png = os.path.join(samp_dir, f"{r['tile_id']}_gen{j}.png")
                Image.fromarray(to_uint8(out[0])).save(png)          # bare 256x256 output
                rows.append({"tile_id": r["tile_id"], "gen_idx": j, "seed": seed_j, "infer_s": round(dt, 4)})
                times.append(dt)
            print(f"fold {a.fold} {r['tile_id']} done", flush=True)
            if a.max_tiles and len(rows) >= a.max_tiles * a.sample_num:
                break

    with open(os.path.join(exp_dir, f"seeds_fold_{a.fold}.json"), "w") as f:
        json.dump({"base_seed": a.seed, "gen_seed": {j: a.seed + j for j in range(a.sample_num)},
                   "rule": "torch.manual_seed(base_seed + gen_idx) immediately before each generation",
                   "sample_steps": int(cfg.model.BB.params.sample_step), "clip_denoised": clip}, f, indent=1)
    with open(os.path.join(exp_dir, f"timing_fold_{a.fold}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "fold", "n_test_tiles", "sample_num", "sample_steps",
                                          "eval_wall_s", "per_generation_infer_s_mean", "per_patch_infer_s_mean",
                                          "gpu", "n_gpus_train", "ckpt"])
        w.writeheader()
        w.writerow({"experiment": a.experiment, "fold": a.fold, "n_test_tiles": len(rows) // a.sample_num,
                    "sample_num": a.sample_num, "sample_steps": cfg.model.BB.params.sample_step,
                    "eval_wall_s": round(time.time() - t_fold, 1),
                    "per_generation_infer_s_mean": round(float(np.mean(times)), 4),
                    "per_patch_infer_s_mean": round(float(np.mean(times)) * a.sample_num, 4),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                    "n_gpus_train": a.n_gpus_train, "ckpt": os.path.basename(a.ckpt)})
    n_tiles = len(rows) // a.sample_num
    print(f"fold {a.fold}: {n_tiles} tiles x {a.sample_num} gens saved to {samp_dir}; "
          f"{np.mean(times):.2f} s/generation", flush=True)
    shutil.rmtree(da.results_root, ignore_errors=True)


if __name__ == "__main__":
    main()
