"""Build the data + image assets for the static ClariDi viewers (website ClariDi/ folder).

  conda activate chatgarment
  python /local/emir/ClariDi/website_ClariDi/build_site.py [--force] [--workers 16]

Writes, next to this script:
  data/claridi_data.js     window.CLARIDI = {...}  (tile metadata, frames, folds, LPIPS context)
  img/ua/<tile>.jpg        UA input (uncleared autofluorescence)
  img/csf/<tile>.jpg       C&SF reference (cleared & stained)
  img/vs/<model>/<tile>_<k>.jpg   VS output, draw k (seed 1234+k)
  img/thumb/<frame>_<scale>.jpg   small stitched C&SF overview for the frame chooser
The HTML/CSS/JS pages are static files maintained by hand in this folder.
Read-only with respect to everything outside website_ClariDi/.
"""
import argparse, csv, json, os, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

R = "/local/emir/ClariDi"
OUT = os.path.dirname(os.path.abspath(__file__))
SPLIT = f"{R}/data/splits/model_design_exp_split_v2.csv"   # v2: band == test fold
MANIFEST = f"{R}/data/bbdm256/manifest.csv"
IMG = f"{R}/data/bbdm256/train"
GEN = f"{R}/deliverables_spatial_v2"
LPIPS = f"{R}/analysis/v2/lpips_folds_0_1_2_3_4.json"
SIZE, Q, Q_VS = 256, 85, 80     # references at q85; VS outputs at q80 to fit all 25 draws in the size budget
THUMB_LONG = 360
SPLIT_LONG = 720

# slug, experiment dir, display name, short name, draws shipped
MODELS = [
    ("stock", "sp_stock_vqgan",    "Vanilla L-BBDM (stock VQGAN)", "Vanilla",          [0, 1, 2, 3, 4]),
    ("ours",  "sp_primary",        "Ours (L-BBDM)",                "Ours",             [0, 1, 2, 3, 4]),
    ("spec",  "sp_specimen_cond",  "Ours + specimen label (oracle)", "+ specimen label (oracle)", [0, 1, 2, 3, 4]),
    ("refA",  "sp_refA_stained",   "Ours + A stained refs (extra input)", "+ A stained refs (extra input)", [0, 1, 2, 3, 4]),
    ("refB",  "sp_refB_unstained", "Ours + B unstained ctx",       "+ B unstained ctx", [0, 1, 2, 3, 4]),
    ("pixel", "pixel_space",       "Pixel-space BBDM",             "Pixel-space BBDM",  [0, 1, 2, 3, 4]),
    ("enc",   "trainable_encoder", "Ours + trainable encoder",     "+ trainable encoder", [0, 1, 2, 3, 4]),
    ("cwgan", "cwgan",             "cWGAN",                        "cWGAN",             [0]),   # one output per tile
    ("p2p",   "pix2pix",           "pix2pix",                      "pix2pix",           [0]),
]


def load_tiles():
    man = {}
    with open(MANIFEST, newline="") as f:          # CRLF-safe
        for r in csv.DictReader(f):
            r = {k.strip(): v.strip() for k, v in r.items()}
            man[r["tile_id"]] = r
    tiles = []
    with open(SPLIT, newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): v.strip() for k, v in r.items()}
            m = man[r["tile_id"]]
            tiles.append(dict(id=r["tile_id"], spec=r["specimen"], tis=r["tissue"], sc=r["scale"],
                              m=int(r["masked"] == "True"), fr=r["frame"], fold=int(r["band"]),
                              x=int(r["x"]), y=int(r["y"]), w=int(r["w"]), h=int(r["h"]),
                              row=int(m["row"]), col=int(m["col"]), u=r["unit"].split("_")[-1],
                              _A=f"{IMG}/A/{m['input_filename']}", _B=f"{IMG}/B/{m['target_filename']}"))
    return tiles


def conv(job):
    src, dst, force, q = job
    if not force and os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return 0
    im = Image.open(src).convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
    tmp = dst + ".tmp"
    im.save(tmp, "JPEG", quality=q, optimize=True, progressive=True)
    os.replace(tmp, dst)
    return 1


def thumb(job):
    frame, sc, ts, dst, long_side, mod = job
    x0 = min(t["x"] for t in ts); y0 = min(t["y"] for t in ts)
    x1 = max(t["x"] + t["w"] for t in ts); y1 = max(t["y"] + t["h"] for t in ts)
    s = long_side / max(x1 - x0, y1 - y0)
    can = Image.new("RGB", (round((x1 - x0) * s), round((y1 - y0) * s)), (40, 40, 40))
    for t in ts:
        px0, py0 = round((t["x"] - x0) * s), round((t["y"] - y0) * s)
        px1, py1 = round((t["x"] + t["w"] - x0) * s), round((t["y"] + t["h"] - y0) * s)
        can.paste(Image.open(t[mod]).convert("RGB").resize((max(1, px1 - px0), max(1, py1 - py0)), Image.LANCZOS), (px0, py0))
    can.save(dst, "JPEG", quality=82, optimize=True)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    tiles = load_tiles()
    lp = json.load(open(LPIPS))
    for d in ["data", "img/ua", "img/csf", "img/thumb"] + [f"img/vs/{m[0]}" for m in MODELS]:
        os.makedirs(f"{OUT}/{d}", exist_ok=True)

    jobs = []
    for t in tiles:
        jobs.append((t["_A"], f"{OUT}/img/ua/{t['id']}.jpg", a.force, Q))
        jobs.append((t["_B"], f"{OUT}/img/csf/{t['id']}.jpg", a.force, Q))
        for slug, exp, _, _, draws in MODELS:
            for k in draws:
                src = f"{GEN}/{exp}/samples/fold_{t['fold']}/{t['id']}_gen{k}.png"
                if not os.path.exists(src):
                    sys.exit(f"missing {src}")
                jobs.append((src, f"{OUT}/img/vs/{slug}/{t['id']}_{k}.jpg", a.force, Q_VS))

    groups = defaultdict(list)
    for t in tiles:
        groups[(t["fr"], t["sc"])].append(t)
    tjobs = [(f, sc, ts, f"{OUT}/img/thumb/{f}_{sc}.jpg", THUMB_LONG, "_B") for (f, sc), ts in sorted(groups.items())]
    # split page: stitched overviews of both grids, C&SF and UA
    os.makedirs(f"{OUT}/img/split", exist_ok=True)
    tjobs += [(f, sc, ts, f"{OUT}/img/split/{f}_{sc}_{name}.jpg", SPLIT_LONG, mod)
              for (f, sc), ts in sorted(groups.items()) for name, mod in (("csf", "_B"), ("ua", "_A"))]

    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(conv, jobs, chunksize=64))
        list(ex.map(thumb, tjobs))
    print(f"{len(jobs)} tile images ({n} written), {len(tjobs)} thumbnails")

    # ------------------------------------------------------------------ metadata
    frames = []
    for fr in sorted({t["fr"] for t in tiles}):
        ft = [t for t in tiles if t["fr"] == fr]
        x0 = min(t["x"] for t in ft); y0 = min(t["y"] for t in ft)
        x1 = max(t["x"] + t["w"] for t in ft); y1 = max(t["y"] + t["h"] for t in ft)
        names = sorted({t["id"].split("_")[0] for t in ft})
        part = fr.split("_p")[-1]
        label = f"{ft[0]['spec']}" + (f" part {part}" if any(f2 != fr and f2.split('_')[0] == fr.split('_')[0]
                                                          for f2 in {t['fr'] for t in tiles}) else "")
        frames.append(dict(id=fr, spec=ft[0]["spec"], tis=ft[0]["tis"], label=label, names=names,
                           x0=x0, y0=y0, W=x1 - x0, H=y1 - y0,
                           n={sc: sum(t["sc"] == sc for t in ft) for sc in ("10x10", "5x5")}))
    out_tiles = []
    for t in tiles:
        L = lp.get(t["id"], {})
        out_tiles.append(dict({k: v for k, v in t.items() if not k.startswith("_")},
                              lp={slug: [round(v, 4) for v in L[exp]] for slug, exp, *_ in MODELS if exp in L}))
    data = dict(
        models=[dict(slug=s, exp=e, name=n, short=sh, draws=d) for s, e, n, sh, d in MODELS],
        seeds=[1234, 1235, 1236, 1237, 1238],
        frames=frames, tiles=out_tiles,
    )
    with open(f"{OUT}/data/claridi_data.js", "w") as f:
        f.write("/* generated by build_site.py */\nwindow.CLARIDI = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("wrote data/claridi_data.js", os.path.getsize(f"{OUT}/data/claridi_data.js") // 1024, "KB")


if __name__ == "__main__":
    main()
