"""Reconstruction-ceiling PNGs: every target tile encoded and decoded through a frozen
VQGAN, saved as a bare 256x256 PNG named by tile_id. No scoring here.
Uses exactly the loader preprocessing (Resize 256, ToTensor, [-1,1]) and the same
tensor->uint8 conversion as the model outputs.

  python vqgan_ceiling_pngs.py --data_root <bbdm256> --config <yaml> --vqgan_ckpt <ckpt> \
      --label finetuned --out_dir <deliverables>/ceilings --device cpu
"""
import argparse, os, csv, time
import torch, yaml
from PIL import Image
import torchvision.transforms as T
from utils import dict2namespace
from specimen_kfold import load_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True); ap.add_argument("--manifest", default=None)
ap.add_argument("--config", default="configs/Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml")
ap.add_argument("--vqgan_ckpt", default=None); ap.add_argument("--label", required=True)
ap.add_argument("--out_dir", required=True); ap.add_argument("--device", default="cpu")
ap.add_argument("--domain", choices=["target", "input"], default="target")
a = ap.parse_args()

from model.VQGAN.vqgan import VQModel
cfg = dict2namespace(yaml.load(open(a.config), Loader=yaml.FullLoader))
if a.vqgan_ckpt: cfg.model.VQGAN.params.ckpt_path = a.vqgan_ckpt
dev = torch.device(a.device)
vq = VQModel(**vars(cfg.model.VQGAN.params)).eval().to(dev)
tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
out = os.path.join(a.out_dir, a.label); os.makedirs(out, exist_ok=True)
recs = load_manifest(a.manifest or os.path.join(a.data_root, "manifest.csv"), a.data_root)
key = "target_path" if a.domain == "target" else "input_path"
t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(recs):
        x = tf(Image.open(r[key]).convert("RGB")).unsqueeze(0).to(dev) * 2 - 1
        q, _, _ = vq.encode(x); rec = vq.decode(q)
        u8 = rec[0].mul(0.5).add(0.5).clamp(0, 1).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        Image.fromarray(u8).save(os.path.join(out, f"{r['tile_id']}.png"))
        if (i + 1) % 100 == 0: print(f"{a.label}: {i+1}/{len(recs)}", flush=True)
with open(os.path.join(a.out_dir, f"ceiling_{a.label}_meta.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["label", "domain", "checkpoint", "n_tiles", "wall_s"])
    w.writerow([a.label, a.domain, cfg.model.VQGAN.params.ckpt_path, len(recs), round(time.time() - t0, 1)])
print(f"{a.label}: wrote {len(recs)} PNGs to {out}"); print("CEILING_PNGS_DONE")
