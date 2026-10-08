"""Per-tile LPIPS (alex) of every generation vs ground truth, for the spatial-split runs.
CONTEXT ONLY: visual review is the verdict. CPU. Usage: python analysis/lpips_compare.py [--v2] [fold ...]
--v2: split v2 (deliverables_spatial_v2, all methods, writes analysis/v2/); the default reproduces split v1.
Draws scored: every existing gen{0..4} (the GANs write gen0 only)."""
import csv, os, sys, json, collections, torch, lpips, numpy as np
from PIL import Image
R = "/local/emir/ClariDi"; V2 = "--v2" in sys.argv
D = f"{R}/deliverables_spatial_v2" if V2 else f"{R}/deliverables_spatial"; A = f"{R}/analysis/v2" if V2 else f"{R}/analysis"
EXPS = ["sp_stock_vqgan", "sp_primary", "sp_specimen_cond", "sp_refA_stained", "sp_refB_unstained"]
if V2: EXPS += ["pixel_space", "trainable_encoder", "cwgan", "pix2pix"]
folds = [int(x) for x in sys.argv[1:] if x != "--v2"] or [0]
man = {r["tile_id"]: r for r in csv.DictReader(open(f"{R}/data/bbdm256/manifest.csv"))}
torch.set_num_threads(16); fn = lpips.LPIPS(net="alex", verbose=False)
T = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
out = {}
for k in folds:
    tiles = sorted({f.rsplit("_gen", 1)[0] for f in os.listdir(f"{D}/{EXPS[1]}/samples/fold_{k}")})
    for tid in tiles:
        gt = T(f"{R}/data/bbdm256/train/B/{man[tid]['target_filename']}")
        rec = {"fold": k, "specimen": man[tid]["specimen"], "tissue": man[tid]["tissue"], "scale": man[tid]["scale"]}
        for e in EXPS:
            ps = [p for p in (f"{D}/{e}/samples/fold_{k}/{tid}_gen{j}.png" for j in range(5)) if os.path.exists(p)]
            if len(ps) == (1 if e in ("cwgan", "pix2pix") else 5):   # complete draw sets only
                with torch.no_grad(): rec[e] = [float(fn(T(p), gt)) for p in ps]
        out[tid] = rec
os.makedirs(A, exist_ok=True)
json.dump(out, open(f"{A}/lpips_folds_{'_'.join(map(str, folds))}.json", "w"))
def table(key):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in out.values():
        for e in EXPS:
            if e in r: g[r[key] if key else "all"][e].append(np.mean(r[e]))
    print(f"\nmean LPIPS (lower = closer), by {key or 'all'}; n = tiles")
    print(f"{'group':>8} {'n':>4} " + " ".join(f"{e.replace('sp_',''):>15}" for e in EXPS))
    for gk in sorted(g):
        print(f"{gk:>8} {len(g[gk][EXPS[1]]):>4} " + " ".join(f"{np.mean(g[gk][e]):>15.3f}" if g[gk][e] else f"{'-':>15}" for e in EXPS))
table(None); table("scale"); table("tissue"); table("specimen")
w = collections.Counter(min((e for e in EXPS if e in r), key=lambda e: np.mean(r[e])) for r in out.values())
print("\ntiles where each model has the lowest mean LPIPS:", dict(w))
