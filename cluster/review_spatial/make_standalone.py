"""Self-contained comparison picker: one HTML file with all images embedded (JPEG, base64), so it opens
by double-click with no server or tunnel. Built from review_spatial/index.html's logic.
Usage: python make_standalone.py [--size 224] [--max_tiles N] [--out file.html]"""
import argparse, base64, csv, io, json, os
from PIL import Image
ap=argparse.ArgumentParser(); ap.add_argument("--size",type=int,default=224); ap.add_argument("--q",type=int,default=82)
ap.add_argument("--max_tiles",type=int,default=0); ap.add_argument("--out",default="/local/emir/ClariDi/review_spatial/claridi_spatial_picker.html")
a=ap.parse_args()
R="/local/emir/ClariDi"
EXPS=["sp_stock_vqgan","sp_primary","sp_specimen_cond","sp_refA_stained","sp_refB_unstained"]
man={r["tile_id"]:r for r in csv.DictReader(open(f"{R}/data/bbdm256/manifest.csv"))}
band={r["tile_id"]:int(r["band"]) for r in csv.DictReader(open(f"{R}/data/splits/model_design_exp_split.csv"))}
lp=json.load(open(f"{R}/analysis/lpips_folds_0_1_2_3_4.json"))
tids=sorted(lp,key=lambda t:(band[t],t))[:a.max_tiles or None]
def enc(p):
    b=io.BytesIO(); Image.open(p).convert("RGB").resize((a.size,a.size)).save(b,"JPEG",quality=a.q); return base64.b64encode(b.getvalue()).decode()
data=[]
for t in tids:
    m=man[t]; k=band[t]
    data.append({"tile_id":t,"fold":k,"specimen":m["specimen"],"tissue":m["tissue"],"scale":m["scale"],"masked":m["masked"],
      "lp":{e:sum(lp[t][e])/5 for e in EXPS},
      "img":[enc(f"{R}/data/bbdm256/train/A/{m['input_filename']}"),enc(f"{R}/data/bbdm256/train/B/{m['target_filename']}")]+
            [enc(f"{R}/deliverables_spatial/{e}/samples/fold_{k}/{t}_gen0.png") for e in EXPS]})
html=open(f"{R}/review_spatial/index.html").read()
js=html.split("<script>")[1].split("</script>")[0]
# data comes embedded instead of fetched; images are data URIs; draw selector limited to gen 0
js=js.replace('''async function load(){''','''const EMBED=__DATA__;
async function load(){
  tiles=EMBED.map(t=>({...t,input_filename:"",target_filename:""}));
  const fillE=(id,vals)=>vals.forEach(v=>{const o=document.createElement("option");o.value=o.textContent=v;$(id).appendChild(o)});
  fillE("#fFold",[...new Set(tiles.map(t=>t.fold))].sort());fillE("#fSpec",[...new Set(tiles.map(t=>t.specimen))].sort());
  summary();render();return;''')
js=js.replace('src="../data/bbdm256/train/A/${t.input_filename}"','src="data:image/jpeg;base64,${t.img[0]}"')
js=js.replace('src="../data/bbdm256/train/B/${t.target_filename}"','src="data:image/jpeg;base64,${t.img[1]}"')
js=js.replace('src="../deliverables_spatial/${e}/samples/fold_${t.fold}/${t.tile_id}_gen${g}.png"','src="data:image/jpeg;base64,${t.img[2+EXPS.findIndex(x=>x[0]==e)]}"')
js=js.replace("__DATA__",json.dumps(data,separators=(",",":")))
html=html.split("<script>")[0]+"<script>"+js+"</script>"+html.split("</script>")[1]
html=html.replace('<label>draw <select id="fGen"><option value="0">gen 0</option><option value="1">gen 1</option><option value="2">gen 2</option><option value="3">gen 3</option><option value="4">gen 4</option></select></label>',
                  '<label>draw <select id="fGen"><option value="0">gen 0</option></select></label>')
html=html.replace("<title>ClariDi spatial comparison</title>","<title>ClariDi spatial picker</title>")
open(a.out,"w").write(html)
print(f"{len(data)} tiles -> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")
