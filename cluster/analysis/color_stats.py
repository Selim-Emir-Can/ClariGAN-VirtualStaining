"""Per-channel colour statistics of generations vs ground truth (all 5 draws), spatial-split folds 0-4 (all available).
Reports mean R and G intensity (0-255) and the green share G/(R+G), per model, overall / by tissue / by specimen,
plus the same for the training targets and for the VQGAN reconstruction ceiling is not available here."""
import csv, json, os, collections, numpy as np
from PIL import Image
R="/local/emir/ClariDi"; D=f"{R}/deliverables_spatial"
EXPS=["sp_stock_vqgan","sp_primary","sp_specimen_cond","sp_refA_stained","sp_refB_unstained"]
man={r["tile_id"]:r for r in csv.DictReader(open(f"{R}/data/bbdm256/manifest.csv"))}
band={r["tile_id"]:int(r["band"]) for r in csv.DictReader(open(f"{R}/data/splits/model_design_exp_split.csv"))}
tiles=[t for t in man if os.path.exists(f"{D}/sp_primary/samples/fold_{band[t]}/{t}_gen0.png")]
def rg(p):
    a=np.asarray(Image.open(p).convert("RGB").resize((256,256)),dtype=np.float32); return a[...,0].mean(),a[...,1].mean()
rows=[]
for t in tiles:
    m=man[t]; k=band[t]; r={"tile":t,"spec":m["specimen"],"tissue":m["tissue"],"scale":m["scale"]}
    r["gt"]=rg(f"{R}/data/bbdm256/train/B/{m['target_filename']}")
    for e in EXPS:
        v=[rg(f"{D}/{e}/samples/fold_{k}/{t}_gen{j}.png") for j in range(5)]; r[e]=(np.mean([x[0] for x in v]),np.mean([x[1] for x in v]))
    rows.append(r)
json.dump(rows,open(f"{R}/analysis/color_stats.json","w"),default=float)
def show(key):
    g=collections.defaultdict(list)
    for r in rows: g[r[key] if key else "all"].append(r)
    cols=["gt"]+EXPS
    print(f"\n{'G/(R+G) green share | mean R | mean G':<40} by {key or 'all'}")
    print(f"{'group':>7} {'n':>4} "+" ".join(f"{c.replace('sp_',''):>18}" for c in cols))
    for k in sorted(g):
        rs=g[k]; out=[]
        for c in cols:
            Rm=np.mean([r[c][0] for r in rs]); Gm=np.mean([r[c][1] for r in rs]); out.append(f"{Gm/(Rm+Gm):.2f} {Rm:5.1f} {Gm:5.1f}")
        print(f"{k:>7} {len(rs):>4} "+" ".join(f"{o:>18}" for o in out))
show(None); show("tissue"); show("spec")
# does the output shift toward the specimen's TRAINING-mean colour? correlation of (output - gt) green share with (train mean - gt)
