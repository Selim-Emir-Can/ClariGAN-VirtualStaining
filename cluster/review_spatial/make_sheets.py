"""Static comparison sheets (PNG) for viewing in VS Code: rows = tiles; columns = condition | GT | 5 models (gen0).
Per specimen: 6 random test tiles (seeded). Plus sheets of the 10 largest gains / losses of A over our L-BBDM (LPIPS)."""
import csv, json, random, os
from PIL import Image, ImageDraw, ImageFont
R="/local/emir/ClariDi"; OUT=f"{R}/review_spatial/sheets"; os.makedirs(OUT, exist_ok=True)
EXPS=[("sp_stock_vqgan","vanilla L-BBDM (stock VQGAN)"),("sp_primary","our L-BBDM"),("sp_specimen_cond","+ specimen label"),
      ("sp_refA_stained","+ A stained refs"),("sp_refB_unstained","+ B unstained ctx")]
S=220; PAD=6; LW=190; HDR=34
man={r["tile_id"]:r for r in csv.DictReader(open(f"{R}/data/bbdm256/manifest.csv"))}
band={r["tile_id"]:int(r["band"]) for r in csv.DictReader(open(f"{R}/data/splits/model_design_exp_split.csv"))}
lp=json.load(open(f"{R}/analysis/lpips_folds_0_1_2_3_4.json"))
mean=lambda a:sum(a)/len(a)
try: F=ImageFont.truetype("DejaVuSans.ttf",15); FS=ImageFont.truetype("DejaVuSans.ttf",13)
except Exception: F=FS=ImageFont.load_default()
def sheet(tids,title,fn):
    cols=["condition (uncleared)","ground truth"]+[n for _,n in EXPS]
    W=LW+len(cols)*(S+PAD); H=HDR+24+len(tids)*(S+22+PAD)
    im=Image.new("RGB",(W,H),(17,17,17)); d=ImageDraw.Draw(im)
    d.text((8,8),title,fill=(230,230,230),font=F)
    for i,c in enumerate(cols): d.text((LW+i*(S+PAD),HDR),c,fill=(170,170,170),font=FS)
    for r,t in enumerate(tids):
        m=man[t]; k=band[t]; y=HDR+24+r*(S+22+PAD); L={e:mean(lp[t][e]) for e,_ in EXPS}; best=min(L,key=L.get)
        d.text((8,y+4),f"{t}\nfold {k} · {m['tissue']} · {m['scale']}",fill=(200,200,200),font=FS)
        paths=[f"{R}/data/bbdm256/train/A/{m['input_filename']}",f"{R}/data/bbdm256/train/B/{m['target_filename']}"]+\
              [f"{R}/deliverables_spatial/{e}/samples/fold_{k}/{t}_gen0.png" for e,_ in EXPS]
        for i,p in enumerate(paths):
            x=LW+i*(S+PAD); im.paste(Image.open(p).convert("RGB").resize((S,S)),(x,y))
            if i>=2:
                e=EXPS[i-2][0]; d.text((x,y+S+3),f"LPIPS {L[e]:.3f}",fill=(80,220,80) if e==best else (150,150,150),font=FS)
                if e==best: d.rectangle([x-2,y-2,x+S+1,y+S+1],outline=(80,220,80),width=2)
    im.save(f"{OUT}/{fn}",optimize=True); return fn
random.seed(0); made=[]
for sp in sorted({man[t]["specimen"] for t in lp}):
    ts=sorted(t for t in lp if man[t]["specimen"]==sp); ts=sorted(random.sample(ts,min(6,len(ts))),key=lambda t:(band[t],t))
    made.append(sheet(ts,f"Specimen {sp}: 6 random test tiles (gen 0). Green = lowest LPIPS (context only).",f"specimen_{sp}.png"))
gain=sorted(lp,key=lambda t:mean(lp[t]['sp_primary'])-mean(lp[t]['sp_refA_stained']),reverse=True)
made.append(sheet(gain[:10],"10 tiles where A (stained refs) improves most over our L-BBDM (LPIPS)","A_largest_gains.png"))
made.append(sheet(gain[::-1][:10],"10 tiles where A (stained refs) is worst relative to our L-BBDM (LPIPS)","A_largest_losses.png"))
print("\n".join(made))
