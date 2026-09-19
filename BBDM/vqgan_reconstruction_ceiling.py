"""Reconstruction ceiling of a frozen VQGAN on the ClariDi stain domain.

Encodes and decodes every held-out target (the cleared+stained image) through the
autoencoder and reports PSNR / SSIM / LPIPS.  This is the upper bound on end-to-end
fidelity for any latent diffusion model trained in that autoencoder's latent space.

Run it for BOTH checkpoints.  Comparing the two end-to-end runs alone conflates the
removal of encoder leakage with the loss of domain-specific reconstruction quality;
the two ceilings separate those effects.

Under leave-one-specimen-out the VQGAN is frozen and fold-independent, so every
specimen is some fold's test set.  Metrics are therefore computed per specimen and
aggregated two ways: a per-specimen macro-average (each specimen counts once, which is
the fold-level average) and a patch-weighted mean over all tiles.

  python vqgan_reconstruction_ceiling.py --data_root <dir> \
      --vqgan_ckpt <ckpt> --config configs/<cfg>.yaml --label finetuned --device cuda:0
"""
import argparse
import csv
import json
import os
import statistics

import torch
import yaml
from PIL import Image

from utils import dict2namespace
from specimen_kfold import load_manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--config", default="configs/Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml")
    p.add_argument("--vqgan_ckpt", default=None, help="overrides the config's ckpt_path")
    p.add_argument("--label", required=True, help="name for this autoencoder, e.g. finetuned / stock")
    p.add_argument("--device", default="cpu", help="cpu or cuda:N")
    p.add_argument("--domain", choices=["target", "input"], default="target",
                   help="target = cleared+stained (the stain domain, headline); input = uncleared")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def load_image(path, size):
    """Exactly the preprocessing ImagePathDataset applies during training and testing:
    Resize((size, size)) then ToTensor, i.e. float in [0, 1]. Tiles are natively larger
    and variable-sized, so the resize is what the model actually sees and the ceiling
    must be measured on the same thing."""
    import torchvision.transforms as transforms
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return tf(Image.open(path).convert("RGB"))


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    from model.VQGAN.vqgan import VQModel
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    cfg = dict2namespace(yaml.load(open(a.config), Loader=yaml.FullLoader))
    if a.vqgan_ckpt:
        cfg.model.VQGAN.params.ckpt_path = a.vqgan_ckpt
    ckpt = cfg.model.VQGAN.params.ckpt_path
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")

    vq = VQModel(**vars(cfg.model.VQGAN.params)).eval().to(device)
    for q in vq.parameters():
        q.requires_grad = False

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)

    recs = load_manifest(a.manifest or os.path.join(a.data_root, "manifest.csv"), a.data_root)
    key = "target_path" if a.domain == "target" else "input_path"

    per_tile = []
    with torch.no_grad():
        for i, r in enumerate(recs):
            x = load_image(r[key], a.image_size).unsqueeze(0).to(device)
            xn = x * 2.0 - 1.0                      # VQGAN operates in [-1, 1]
            quant, _, _ = vq.encode(xn)
            rec = vq.decode(quant).clamp(-1, 1)
            rec01 = (rec + 1.0) / 2.0
            per_tile.append({
                "tile_id": r["tile_id"], "specimen": r["specimen"], "tissue": r["tissue"],
                "scale": r["scale"],
                "psnr": float(psnr(rec01, x)),
                "ssim": float(ssim(rec01, x)),
                "lpips": float(lpips(rec01, x)),
            })
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(recs)}", flush=True)

    specimens = sorted({t["specimen"] for t in per_tile})
    per_spec = []
    for s in specimens:
        ts = [t for t in per_tile if t["specimen"] == s]
        per_spec.append({
            "specimen": s, "tissue": ts[0]["tissue"], "n_patches": len(ts),
            **{m: statistics.fmean(t[m] for t in ts) for m in ("psnr", "ssim", "lpips")},
        })

    summary = {
        "label": a.label, "domain": a.domain, "checkpoint": ckpt, "seed": a.seed,
        "n_patches": len(per_tile), "n_specimens": len(specimens),
        "macro_by_specimen": {m: statistics.fmean(s[m] for s in per_spec)
                              for m in ("psnr", "ssim", "lpips")},
        "patch_weighted": {m: statistics.fmean(t[m] for t in per_tile)
                           for m in ("psnr", "ssim", "lpips")},
        "macro_sd_across_specimens": {m: (statistics.stdev([s[m] for s in per_spec])
                                          if len(per_spec) > 1 else 0.0)
                                      for m in ("psnr", "ssim", "lpips")},
    }

    print(f"\n=== {a.label} VQGAN, {a.domain} domain, {len(per_tile)} tiles / {len(specimens)} specimens ===")
    print(f"{'specimen':9} {'tissue':6} {'n':>4}  {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
    for s in per_spec:
        print(f"{s['specimen']:9} {s['tissue']:6} {s['n_patches']:>4}  "
              f"{s['psnr']:7.2f} {s['ssim']:7.4f} {s['lpips']:7.4f}")
    m, w = summary["macro_by_specimen"], summary["patch_weighted"]
    print(f"{'macro':9} {'':6} {'':>4}  {m['psnr']:7.2f} {m['ssim']:7.4f} {m['lpips']:7.4f}   "
          f"(per-specimen macro-average = fold-level average)")
    print(f"{'weighted':9} {'':6} {len(per_tile):>4}  {w['psnr']:7.2f} {w['ssim']:7.4f} {w['lpips']:7.4f}   "
          f"(patch-weighted mean)")

    out_dir = a.out_dir or os.path.join(os.path.dirname(a.data_root.rstrip("/")), "results",
                                        "vqgan_ceiling")
    os.makedirs(out_dir, exist_ok=True)
    base = f"ceiling_{a.label}_{a.domain}"
    with open(os.path.join(out_dir, base + "_per_specimen.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(per_spec[0].keys())); w_.writeheader(); w_.writerows(per_spec)
    with open(os.path.join(out_dir, base + "_per_tile.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(per_tile[0].keys())); w_.writeheader(); w_.writerows(per_tile)
    with open(os.path.join(out_dir, base + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
