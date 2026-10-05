"""Visual review of the spatially blocked 5-fold split (model_design_exp_split).

Stitches the 256px crops back into their whole-slide frames (one mosaic per frame x scale x
modality), overlays bands / fold partitions / units, and computes balance + boundary stats.

  python review_spatial/make_split_viz.py
  -> review_spatial/split_viz.html  +  review_spatial/split_viz/*.jpg  +  split_viz/stats.json
Read-only with respect to the split CSV, manifest and images.
"""
import csv, json, os, html
from bisect import bisect_left
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = "/local/emir/ClariDi"
SPLIT = f"{ROOT}/data/splits/model_design_exp_split.csv"
MANIFEST = f"{ROOT}/data/bbdm256/manifest.csv"
IMG = f"{ROOT}/data/bbdm256/train"
OUT_DIR = f"{ROOT}/review_spatial/split_viz"
OUT_HTML = f"{ROOT}/review_spatial/split_viz.html"
LONG_SIDE = 1100          # display mosaic long side (px)
N = 5
SLIVER = 0.03


def val_band(k):
    return k + 1 if k + 1 < N else k - 1


def part_of(b, k):
    return "test" if b == k else "val" if b == val_band(k) else "train"


def load():
    man = {}
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            man[r["tile_id"]] = r
    tiles = []
    with open(SPLIT, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("band", "x", "y", "w", "h"): r[k] = int(r[k])
            r["masked"] = r["masked"].strip() == "True"
            m = man[r["tile_id"]]
            r["A"], r["B"] = m["input_filename"], m["target_filename"]
            tiles.append(r)
    return tiles


# ---------------------------------------------------------------- compressed cell grid
def cell_grid(ts):
    """Coordinate-compressed grid over all tile edges; labels each cell with the tile covering it
    (both scales; 5x5 and 10x10 crops cover the same pixels so any covering tile gives the unit)."""
    xs = sorted({t["x"] for t in ts} | {t["x"] + t["w"] for t in ts})
    ys = sorted({t["y"] for t in ts} | {t["y"] + t["h"] for t in ts})
    unit = np.full((len(ys) - 1, len(xs) - 1), -1, int)
    band = np.full_like(unit, -1)
    uidx = {u: i for i, u in enumerate(sorted({t["unit"] for t in ts}))}
    # draw big tiles first so 10x10 labels win on slivers (they are the finer grid)
    for t in sorted(ts, key=lambda t: -t["w"] * t["h"]):
        i0, i1 = bisect_left(ys, t["y"]), bisect_left(ys, t["y"] + t["h"])
        j0, j1 = bisect_left(xs, t["x"]), bisect_left(xs, t["x"] + t["w"])
        unit[i0:i1, j0:j1] = uidx[t["unit"]]
        band[i0:i1, j0:j1] = t["band"]
    return np.array(xs), np.array(ys), unit, band, uidx


def edges(xs, ys, lab):
    """Yield (a, b, length, segment) for every cell edge between labels a != b (-1 = empty/outside)."""
    ny, nx = lab.shape
    P = np.pad(lab, 1, constant_values=-1)
    for i in range(ny + 1):          # horizontal edges at y = ys[i]
        for j in range(nx):
            a, b = P[i, j + 1], P[i + 1, j + 1]
            if a != b:
                yield a, b, xs[j + 1] - xs[j], (xs[j], ys[i], xs[j + 1], ys[i])
    for i in range(ny):              # vertical edges at x = xs[j]
        for j in range(nx + 1):
            a, b = P[i + 1, j], P[i + 1, j + 1]
            if a != b:
                yield a, b, ys[i + 1] - ys[i], (xs[j], ys[i], xs[j], ys[i + 1])


def seg_path(segs):
    return "".join(f"M{x0} {y0}L{x1} {y1}" for x0, y0, x1, y1 in segs)


def region_paths(xs, ys, lab, labels):
    """One path per label, built from horizontal runs of cells."""
    out = {}
    for L in labels:
        d = []
        for i in range(lab.shape[0]):
            j = 0
            while j < lab.shape[1]:
                if lab[i, j] == L:
                    j0 = j
                    while j < lab.shape[1] and lab[i, j] == L: j += 1
                    d.append(f"M{xs[j0]} {ys[i]}H{xs[j]}V{ys[i+1]}H{xs[j0]}Z")
                else:
                    j += 1
        out[int(L)] = "".join(d)
    return out


def cell_area(xs, ys, lab, L):
    a = np.outer(np.diff(ys), np.diff(xs))
    return float(a[lab == L].sum())


# ---------------------------------------------------------------- alternative partition
def recursive_blocks(units, n):
    """Compact 2D alternative: recursive bisection of units along the longer bbox axis,
    cutting at tile-count proportion. Returns list of n lists of units."""
    if n == 1: return [units]
    tiles = [t for u in units for t in u]
    W = max(t["x"] + t["w"] for t in tiles) - min(t["x"] for t in tiles)
    H = max(t["y"] + t["h"] for t in tiles) - min(t["y"] for t in tiles)
    ax, sp = ("x", "w") if W >= H else ("y", "h")
    def c(u):
        a = sum(t["w"] * t["h"] for t in u)
        return sum((t[ax] + t[sp] / 2) * t["w"] * t["h"] for t in u) / a
    us = sorted(units, key=c)
    n1 = n // 2; tot = len(tiles); cum = 0; cut = len(us)
    for i, u in enumerate(us):
        if cum + len(u) / 2 >= tot * n1 / n: cut = i; break
        cum += len(u)
    cut = max(1, min(len(us) - 1, cut)) if len(us) > 1 else 1
    return recursive_blocks(us[:cut], n1) + recursive_blocks(us[cut:], n - n1) if len(us) > 1 else \
        [us] + [[] for _ in range(n - 1)]


def contact_stats(ts, band_of, w10):
    """Per fold: test perimeter shared with train / val / empty (native px), and fraction of
    test area within one 10x10-tile width (w10) of train pixels (raster at 1/16)."""
    xs, ys, unit, _, _ = cell_grid(ts)
    lab = np.full_like(unit, -1)
    # band label per cell from the band_of mapping (unit -> band)
    uinv = {i: u for u, i in cell_grid(ts)[4].items()}
    for (i, j), u in np.ndenumerate(unit):
        if u >= 0: lab[i, j] = band_of[uinv[u]]
    E = defaultdict(float)
    for a, b, L, _ in edges(xs, ys, lab):
        E[(a, b)] += L; E[(b, a)] += L
    s = 16
    x0, y0 = xs[0], ys[0]
    Wr, Hr = int(np.ceil((xs[-1] - x0) / s)), int(np.ceil((ys[-1] - y0) / s))
    R = np.full((Hr, Wr), -1, int)
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            if lab[i, j] >= 0:
                R[(ys[i] - y0) // s:(ys[i + 1] - y0) // s + 1, (xs[j] - x0) // s:(xs[j + 1] - x0) // s + 1] = lab[i, j]
    res = []
    for k in range(N):
        v = val_band(k); tr = [b for b in range(N) if b not in (k, v)]
        te_tr = sum(E[(k, b)] for b in tr); te_v = E[(k, v)]; te_e = E[(k, -1)]
        area = cell_area(xs, ys, lab, k)
        train_mask = np.isin(R, tr); test_mask = R == k
        if test_mask.sum() and train_mask.sum():
            dist = ndimage.distance_transform_edt(~train_mask) * s
            near = float((dist[test_mask] <= w10).mean())
            touch = float((dist[test_mask] <= s * 1.5).mean())
        else:
            near = touch = 0.0
        res.append(dict(fold=k, test_area=area, te_train=te_tr, te_val=te_v, te_empty=te_e,
                        near_train=near, n_test=sum(1 for t in ts if band_of[t["unit"]] == k)))
    return res


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tiles = load()
    frames = defaultdict(list)
    for t in tiles: frames[t["frame"]].append(t)
    fdata, stats = [], {"frames": {}}
    geo_flags = []
    for f in sorted(frames):
        ts = frames[f]
        x0 = min(t["x"] for t in ts); y0 = min(t["y"] for t in ts)
        x1 = max(t["x"] + t["w"] for t in ts); y1 = max(t["y"] + t["h"] for t in ts)
        W, H = x1 - x0, y1 - y0
        s = LONG_SIDE / max(W, H)
        dw, dh = round(W * s), round(H * s)
        # mosaics (+ grayscale float copies for geometry check)
        gray = {}
        for sc in ("5x5", "10x10"):
            for mod in ("B", "A"):
                can = Image.new("RGB", (dw, dh), (24, 26, 30))
                g = np.full((dh, dw), np.nan, np.float32)
                for t in ts:
                    if t["scale"] != sc: continue
                    px0, py0 = round((t["x"] - x0) * s), round((t["y"] - y0) * s)
                    px1, py1 = round((t["x"] + t["w"] - x0) * s), round((t["y"] + t["h"] - y0) * s)
                    im = Image.open(f"{IMG}/{mod}/{t[mod]}").convert("RGB").resize(
                        (max(1, px1 - px0), max(1, py1 - py0)), Image.LANCZOS)
                    can.paste(im, (px0, py0))
                    if mod == "B":
                        g[py0:py1, px0:px1] = np.asarray(im.convert("L"), np.float32)[:py1 - py0, :px1 - px0]
                can.save(f"{OUT_DIR}/{f}_{sc}_{mod}.jpg", quality=80, optimize=True)
                if mod == "B": gray[sc] = g
        # geometry check: every 5x5 crop vs the 10x10 mosaic underneath it
        geo = []
        for t in ts:
            if t["scale"] != "5x5": continue
            px0, py0 = round((t["x"] - x0) * s), round((t["y"] - y0) * s)
            px1, py1 = round((t["x"] + t["w"] - x0) * s), round((t["y"] + t["h"] - y0) * s)
            a = gray["5x5"][py0:py1, px0:px1]; b = gray["10x10"][py0:py1, px0:px1]
            m = ~np.isnan(a) & ~np.isnan(b)
            cov = float(m.mean())
            if cov < 0.2 or a[m].std() < 1 or b[m].std() < 1:
                continue
            r = float(np.corrcoef(a[m], b[m])[0, 1])
            geo.append(dict(tile=t["tile_id"], cov=round(cov, 2), r=round(r, 3)))
            if r < 0.8: geo_flags.append(dict(frame=f, tile=t["tile_id"], cov=round(cov, 2), r=round(r, 3)))
        # overlays
        xs, ys, unit, band, uidx = cell_grid(ts)
        unit_segs = [seg for a, b, _, seg in edges(xs, ys, unit)]
        band_segs = [seg for a, b, _, seg in edges(xs, ys, band) if a >= 0 and b >= 0]
        regions = region_paths(xs, ys, band, range(N))
        # per-band stats
        bstats = []
        long_ax = "x" if x1 >= y1 else "y"   # same rule as spatial_split.py (extent from frame origin)
        for b in range(N):
            bt = [t for t in ts if t["band"] == b]
            area = cell_area(xs, ys, band, b)
            if bt:
                lo = min(t[long_ax] for t in bt); hi = max(t[long_ax] + t["w" if long_ax == "x" else "h"] for t in bt)
            else:
                lo = hi = 0
            bstats.append(dict(band=b, n=len(bt),
                               n5=sum(t["scale"] == "5x5" for t in bt), n10=sum(t["scale"] == "10x10" for t in bt),
                               masked=sum(t["masked"] for t in bt), area_mpx=round(area / 1e6, 2),
                               extent=int(hi - lo), units=len({t["unit"] for t in bt})))
        usz = Counter(t["unit"] for t in ts)
        w10 = max(t["w"] for t in ts if t["scale"] == "10x10")
        w5 = max(t["w" if long_ax == "x" else "h"] for t in ts if t["scale"] == "5x5")
        # contact stats: current strips vs compact 2D recursive blocks (same units)
        units = defaultdict(list)
        for t in ts: units[t["unit"]].append(t)
        cur = contact_stats(ts, {u: v[0]["band"] for u, v in units.items()}, w10)
        blocks = recursive_blocks(list(units.values()), N)
        alt_map = {u[0]["unit"]: bi for bi, blk in enumerate(blocks) for u in blk}
        alt = contact_stats(ts, alt_map, w10)
        stats["frames"][f] = dict(tissue=ts[0]["tissue"], specimen=ts[0]["specimen"], W=W, H=H,
                                  long_axis=long_ax, w5_long=w5, bands=bstats,
                                  unit_sizes=sorted(usz.values(), reverse=True), contact=cur,
                                  contact_alt2d=alt, alt2d_counts=[sum(len(u) for u in blk) for blk in blocks],
                                  geo=geo)
        fdata.append(dict(
            frame=f, tissue=ts[0]["tissue"], specimen=ts[0]["specimen"], x0=x0, y0=y0, W=W, H=H,
            long_axis=long_ax, w5=w5,
            tiles=[dict(id=t["tile_id"], s=t["scale"], m=int(t["masked"]), u=t["unit"].split("_")[-1],
                        b=t["band"], x=t["x"], y=t["y"], w=t["w"], h=t["h"], A=t["A"], B=t["B"])
                   for t in ts],
            unitPath=seg_path(unit_segs), bandPath=seg_path(band_segs), regions=regions,
            bstats=bstats, nunits=len(usz), maxunit=max(usz.values()),
            contact=cur, contact_alt=alt, alt_counts=stats["frames"][f]["alt2d_counts"],
            geo_min=min((g["r"] for g in geo), default=None), geo_n=len(geo),
            geo_med=float(np.median([g["r"] for g in geo])) if geo else None))
        print(f"{f}: {len(ts)} tiles, {len(usz)} units (max {max(usz.values())}), "
              f"geo r min={fdata[-1]['geo_min']} n={len(geo)}")
    # fold totals
    folds = []
    for k in range(N):
        c = Counter(part_of(t["band"], k) for t in tiles)
        folds.append(dict(fold=k, test=c["test"], val=c["val"], train=c["train"]))
    stats["folds"] = folds; stats["geo_flags"] = geo_flags
    json.dump(stats, open(f"{OUT_DIR}/stats.json", "w"), indent=1)
    page = TEMPLATE.replace("__DATA__", json.dumps(dict(frames=fdata, folds=folds, geo_flags=geo_flags),
                                                   separators=(",", ":")))
    open(OUT_HTML, "w").write(page)
    print("folds", folds)
    print("geo flags", geo_flags)
    print("wrote", OUT_HTML)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial Split Review</title>
<style>
:root{--bg:#121417;--panel:#1b1e23;--panel2:#22262c;--ink:#e6e8eb;--mute:#9aa3ad;--line:#333941;--accent:#7fb3ff;--warn:#ffb454;--bad:#ff6b6b;
 --b0:#E69F00;--b1:#56B4E9;--b2:#009E73;--b3:#F0E442;--b4:#CC79A7;--test:#D55E00;--val:#F0E442;--train:#0072B2}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f5f6f8;--panel:#fff;--panel2:#eef0f3;--ink:#1d2228;--mute:#5d6670;--line:#d5d9de;--accent:#1f5fbf;--warn:#a85d00;--bad:#c62828}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:17px;margin:0 0 6px;font-weight:600}
h2{font-size:15px;margin:28px 0 8px}
.wrap{max-width:1500px;margin:0 auto;padding:0 16px 40px}
.ctl{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{background:var(--panel);color:var(--ink);border:0;border-right:1px solid var(--line);padding:4px 10px;font:inherit;cursor:pointer}
.seg button:last-child{border-right:0}
.seg button.on{background:var(--accent);color:#0b1220;font-weight:600}
.lbl{color:var(--mute);font-size:12px;margin-right:4px}
label.ck{display:inline-flex;gap:4px;align-items:center;color:var(--mute);font-size:13px}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12.5px;margin-top:6px;color:var(--mute)}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px;border:1px solid #0006}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px;margin-top:14px}
@media (max-width:520px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;min-width:0}
.card h3{margin:0 0 4px;font-size:14px;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.card h3 small{color:var(--mute);font-weight:400}
.meta{color:var(--mute);font-size:12px;margin-bottom:6px}
svg.mos{width:100%;height:auto;display:block;background:#181a1e;border-radius:6px}
.tile{fill:transparent;stroke-width:1;vector-effect:non-scaling-stroke;cursor:crosshair}
.tile:hover{stroke:#fff!important;stroke-width:2.5}
.unit{fill:none;stroke:#fff;stroke-opacity:.75;stroke-width:1.6;vector-effect:non-scaling-stroke;stroke-dasharray:4 3}
.bandb{fill:none;stroke:#fff;stroke-width:3;vector-effect:non-scaling-stroke}
.mini{display:flex;gap:2px;margin-top:6px}
.mini div{flex:1;text-align:center;font-size:11px;border-radius:3px;padding:1px 0;color:#111}
#tip{position:fixed;pointer-events:none;z-index:50;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px;font-size:12px;display:none;max-width:290px;box-shadow:0 6px 20px #0008}
#tip img{width:128px;height:128px;border-radius:4px;margin-top:4px}
table{border-collapse:collapse;font-size:12.5px;width:100%}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}
th,td{padding:4px 7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--mute);font-weight:500;position:sticky;top:0;background:var(--panel)}
td.l,th.l{text-align:left}
td.flag{color:var(--bad);font-weight:600}
td.warn{color:var(--warn)}
.note{color:var(--mute);font-size:12.5px;max-width:980px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;font-size:11px;background:var(--panel)}
</style></head><body>
<header><div class="wrap" style="padding:0">
<h1>Spatial 5-fold split &mdash; stitched frames</h1>
<div class="ctl">
 <span><span class="lbl">View</span><span class="seg" id="view"></span></span>
 <span><span class="lbl">Scale</span><span class="seg" id="scale"><button data-v="10x10">10&times;10</button><button data-v="5x5">5&times;5</button></span></span>
 <span><span class="lbl">Image</span><span class="seg" id="mod"><button data-v="B">GT stained</button><button data-v="A">condition</button></span></span>
 <label class="ck"><input type="checkbox" id="units" checked>units</label>
 <label class="ck"><input type="checkbox" id="bandb" checked>band edges</label>
 <label class="ck">fill <input type="range" id="op" min="0" max="80" value="30" style="width:90px"></label>
 <span class="lbl"><kbd>b</kbd> bands <kbd>0</kbd>&ndash;<kbd>4</kbd> fold <kbd>s</kbd> scale <kbd>c</kbd> image <kbd>u</kbd> units</span>
</div>
<div class="legend" id="legend"></div>
</div></header>
<div class="wrap">
<div class="grid" id="grid"></div>
<h2>Fold totals</h2><div class="tw"><table id="foldtab"></table></div>
<h2>Per frame &times; band</h2>
<p class="note">Tiles = all crops (5&times;5 + 10&times;10). Area = union of crop footprints in native Mpx. Extent = band length along the frame's long axis in 5&times;5-crop widths (&lt;1.5 = thin strip). Flags: <span style="color:var(--bad)">red</span> = band empty or &lt;50% of the frame's mean band size; <span style="color:var(--warn)">amber</span> = &gt;150% or &lt;70%.</p>
<div class="tw"><table id="bandtab"></table></div>
<h2>Test-band contact with training data</h2>
<p class="note">Per frame and fold: the test band's perimeter shared with train vs val vs empty background (native px), and the fraction of test area within one 10&times;10-crop width of a training pixel. &ldquo;2D&rdquo; = same units regrouped into 5 compact blocks by recursive bisection (a hypothetical alternative, same tile budget per block &plusmn; unit granularity).</p>
<div class="tw"><table id="contab"></table></div>
<h2>Geometry check</h2>
<p class="note">Each 5&times;5 crop was correlated (grayscale GT, display resolution) with the 10&times;10 mosaic beneath it. r near 1 means the two grids are registered; r &lt; 0.8 is flagged for visual inspection (low r also arises on low-contrast / out-of-focus tissue).</p>
<div class="tw"><table id="geotab"></table></div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const N = 5, BC = ["var(--b0)","var(--b1)","var(--b2)","var(--b3)","var(--b4)"];
const PC = {test:"var(--test)", val:"var(--val)", train:"var(--train)"};
const S = {view:"bands", scale:"10x10", mod:"B", units:true, bandb:true, op:30};
const valBand = k => k + 1 < N ? k + 1 : k - 1;
const part = (b, k) => b === k ? "test" : b === valBand(k) ? "val" : "train";
const colorOf = b => S.view === "bands" ? BC[b] : PC[part(b, +S.view)];
const NS = "http://www.w3.org/2000/svg";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function el(tag, attrs, parent){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }

// controls
const vEl = document.getElementById("view");
["bands",0,1,2,3,4].forEach(v => { const b = document.createElement("button"); b.dataset.v = v; b.textContent = v === "bands" ? "bands" : "fold " + v; vEl.appendChild(b); });
function seg(id, key){ document.getElementById(id).addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S[key] = b.dataset.v; render(); }); }
seg("view","view"); seg("scale","scale"); seg("mod","mod");
document.getElementById("units").onchange = e => { S.units = e.target.checked; render(); };
document.getElementById("bandb").onchange = e => { S.bandb = e.target.checked; render(); };
document.getElementById("op").oninput = e => { S.op = +e.target.value; render(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  if ("01234".includes(e.key)) S.view = e.key;
  else if (e.key === "b") S.view = "bands";
  else if (e.key === "s") S.scale = S.scale === "5x5" ? "10x10" : "5x5";
  else if (e.key === "c") S.mod = S.mod === "B" ? "A" : "B";
  else if (e.key === "u") { S.units = !S.units; document.getElementById("units").checked = S.units; }
  else return;
  render();
});

// cards
const grid = document.getElementById("grid"), tip = document.getElementById("tip");
const cards = D.frames.map(F => {
  const c = document.createElement("div"); c.className = "card";
  const cnt = F.bstats.map(b => b.n);
  c.innerHTML = `<h3><span>${esc(F.frame)} <small>specimen ${esc(F.specimen)} &middot; ${esc(F.tissue)}</small></span><small>${F.tiles.length} tiles &middot; ${F.nunits} units (largest ${F.maxunit}) &middot; long axis ${F.long_axis}</small></h3>`;
  const svg = el("svg", {class:"mos", viewBox:`${F.x0} ${F.y0} ${F.W} ${F.H}`, preserveAspectRatio:"xMidYMid meet"});
  const defs = el("defs", {}, svg);
  const pat = el("pattern", {id:"hatch-"+F.frame, patternUnits:"userSpaceOnUse", width: F.W/60, height: F.W/60, patternTransform:"rotate(45)"}, defs);
  el("rect", {width: F.W/120, height: F.W/60, fill:"#fff", "fill-opacity":".55"}, pat);
  const img = el("image", {x:F.x0, y:F.y0, width:F.W, height:F.H, preserveAspectRatio:"none"}, svg);
  const regs = el("g", {}, svg); const regEls = [];
  for (let b = 0; b < N; b++) regEls.push(el("path", {d: F.regions[b] || "", "fill-rule":"nonzero"}, regs));
  const tg = el("g", {}, svg); const tEls = [];
  F.tiles.forEach(t => {
    const r = el("rect", {class:"tile", x:t.x, y:t.y, width:t.w, height:t.h}, tg);
    r._t = t; tEls.push(r);
    if (t.m) { r._h = el("rect", {x:t.x, y:t.y, width:t.w, height:t.h, fill:`url(#hatch-${F.frame})`, "pointer-events":"none"}, tg); }
  });
  const bp = el("path", {class:"bandb", d:F.bandPath, "pointer-events":"none"}, svg);
  const up = el("path", {class:"unit", d:F.unitPath, "pointer-events":"none"}, svg);
  svg.addEventListener("mousemove", e => {
    const r = e.target; if (!r._t) { tip.style.display = "none"; return; }
    const t = r._t, p = S.view === "bands" ? "" : ` &middot; <b style="color:${PC[part(t.b,+S.view)]}">${part(t.b,+S.view)}</b>`;
    tip.innerHTML = `<b>${esc(t.id)}</b><br>${esc(F.frame)} &middot; unit ${esc(t.u)} &middot; band ${t.b}${p}<br>${t.s}${t.m ? " &middot; <b>masked</b>" : ""} &middot; ${esc(F.tissue)}<br>box ${t.x},${t.y} ${t.w}&times;${t.h}<br>`+
      `<img src="../data/bbdm256/train/A/${encodeURIComponent(t.A)}" alt=""> <img src="../data/bbdm256/train/B/${encodeURIComponent(t.B)}" alt="">`;
    tip.style.display = "block";
    const x = Math.min(e.clientX + 14, innerWidth - 300), y = Math.min(e.clientY + 14, innerHeight - 220);
    tip.style.left = x + "px"; tip.style.top = y + "px";
  });
  svg.addEventListener("mouseleave", () => tip.style.display = "none");
  c.appendChild(svg);
  const meta = document.createElement("div"); meta.className = "mini"; c.appendChild(meta);
  grid.appendChild(c);
  return {F, img, regEls, tEls, bp, up, meta};
});

function legend(){
  const L = document.getElementById("legend");
  const items = S.view === "bands" ? BC.map((c,i) => [c, "band " + i]) :
    [[PC.test, `test (band ${S.view})`], [PC.val, `val (band ${valBand(+S.view)})`], [PC.train, "train (other bands)"]];
  L.innerHTML = items.map(([c,t]) => `<span><span class="sw" style="background:${c}"></span>${t}</span>`).join("") +
    `<span><span class="sw" style="background:repeating-linear-gradient(45deg,#fff9 0 2px,transparent 2px 5px)"></span>masked crop</span>` +
    `<span>white solid = band boundary &middot; white dashed = unit outline &middot; outlined rects = crops of the selected scale</span>`;
}
function render(){
  document.querySelectorAll(".seg button").forEach(b => {
    const k = b.parentElement.id; b.classList.toggle("on", String(S[k]) === b.dataset.v);
  });
  legend();
  for (const C of cards) {
    const F = C.F;
    C.img.setAttribute("href", `split_viz/${F.frame}_${S.scale}_${S.mod}.jpg`);
    C.regEls.forEach((p, b) => { p.setAttribute("fill", colorOf(b)); p.setAttribute("fill-opacity", S.op / 100); });
    C.tEls.forEach(r => {
      const on = r._t.s === S.scale;
      r.style.display = on ? "" : "none"; if (r._h) r._h.style.display = on ? "" : "none";
      r.style.stroke = colorOf(r._t.b); r.style.strokeOpacity = .9;
    });
    C.up.style.display = S.units ? "" : "none"; C.bp.style.display = S.bandb ? "" : "none";
    C.meta.innerHTML = F.bstats.map(b => `<div style="background:${colorOf(b.band)}" title="band ${b.band}: ${b.n} tiles">${S.view==="bands"?"b"+b.band:part(b.band,+S.view)[0]+b.band}: ${b.n}</div>`).join("");
  }
}

// tables
(function(){
  const ft = document.getElementById("foldtab");
  ft.innerHTML = `<tr><th class="l">fold</th><th>test band</th><th>val band</th><th>test</th><th>val</th><th>train</th></tr>` +
    D.folds.map(f => `<tr><td class="l">${f.fold}</td><td>${f.fold}</td><td>${valBand(f.fold)}</td><td>${f.test}</td><td>${f.val}</td><td>${f.train}</td></tr>`).join("");
  const bt = document.getElementById("bandtab");
  let h = `<tr><th class="l">frame</th><th class="l">tissue</th>` + [0,1,2,3,4].map(b => `<th>b${b} tiles (5|10)</th><th>masked</th><th>Mpx</th><th>extent</th>`).join("") + `<th>units</th></tr>`;
  const tot = [0,0,0,0,0].map(() => ({n:0,brain:0,heart:0,n5:0,n10:0,m:0,a:0}));
  for (const F of D.frames) {
    const mean = F.tiles.length / N;
    h += `<tr><td class="l">${esc(F.frame)}</td><td class="l">${esc(F.tissue)}</td>`;
    for (const b of F.bstats) {
      const cls = (b.n === 0 || b.n < .5 * mean) ? "flag" : (b.n > 1.5 * mean || b.n < .7 * mean) ? "warn" : "";
      const ext = b.extent / F.w5;
      h += `<td class="${cls}">${b.n} (${b.n5}|${b.n10})</td><td>${b.masked}</td><td>${b.area_mpx}</td><td class="${b.n && ext < 1.5 ? "warn" : ""}">${b.n ? ext.toFixed(1) : "&ndash;"}</td>`;
      const T = tot[b.band]; T.n += b.n; T[F.tissue] += b.n; T.n5 += b.n5; T.n10 += b.n10; T.m += b.masked; T.a += b.area_mpx;
    }
    h += `<td>${F.nunits}</td></tr>`;
  }
  h += `<tr><td class="l"><b>all</b></td><td class="l"></td>` + tot.map(T => `<td><b>${T.n}</b> (${T.n5}|${T.n10})<br><small>brain ${T.brain} / heart ${T.heart}</small></td><td>${T.m}</td><td>${T.a.toFixed(1)}</td><td></td>`).join("") + `<td></td></tr>`;
  bt.innerHTML = h;
  // per fold per frame partition counts appended as another block
  let p = `<tr><th class="l">frame</th>` + [0,1,2,3,4].map(k => `<th>f${k} test</th><th>val</th><th>train</th>`).join("") + `</tr>`;
  for (const F of D.frames) {
    p += `<tr><td class="l">${esc(F.frame)}</td>`;
    for (let k = 0; k < N; k++) {
      const c = {test:0,val:0,train:0}; F.tiles.forEach(t => c[part(t.b,k)]++);
      p += `<td class="${c.test===0?"flag":""}">${c.test}</td><td class="${c.val===0?"flag":""}">${c.val}</td><td>${c.train}</td>`;
    }
    p += `</tr>`;
  }
  const t2 = document.createElement("table"); t2.innerHTML = p; t2.style.marginTop = "0";
  const wrap = document.createElement("div"); wrap.className = "tw"; wrap.style.marginTop = "12px"; wrap.appendChild(t2);
  bt.parentElement.after(wrap);
  // contact
  const ct = document.getElementById("contab");
  let c = `<tr><th class="l">frame</th>` + [0,1,2,3,4].map(k => `<th>f${k} test&ndash;train / test&ndash;val edge</th><th>near-train (strip | 2D)</th>`).join("") + `</tr>`;
  const agg = [0,1,2,3,4].map(() => ({tr:0,v:0,a:0,near:0,nearAlt:0,w:0}));
  for (const F of D.frames) {
    c += `<tr><td class="l">${esc(F.frame)}</td>`;
    F.contact.forEach((r, k) => {
      const a = F.contact_alt[k];
      c += `<td>${(r.te_train/1000).toFixed(1)}k / ${(r.te_val/1000).toFixed(1)}k</td><td>${(100*r.near_train).toFixed(0)}% | ${(100*a.near_train).toFixed(0)}%</td>`;
      const G = agg[k]; G.tr += r.te_train; G.v += r.te_val; G.near += r.near_train * r.test_area; G.nearAlt += a.near_train * a.test_area; G.a += r.test_area; G.w += a.test_area;
    });
    c += `</tr>`;
  }
  c += `<tr><td class="l"><b>area-weighted</b></td>` + agg.map(G => `<td><b>${(G.tr/1000).toFixed(0)}k / ${(G.v/1000).toFixed(0)}k</b></td><td><b>${(100*G.near/G.a).toFixed(0)}% | ${(100*G.nearAlt/G.w).toFixed(0)}%</b></td>`).join("") + `</tr>`;
  ct.innerHTML = c;
  const gt = document.getElementById("geotab");
  gt.innerHTML = `<tr><th class="l">frame</th><th>5x5 crops checked</th><th>median r</th><th>min r</th></tr>` +
    D.frames.map(F => `<tr><td class="l">${esc(F.frame)}</td><td>${F.geo_n}</td><td>${F.geo_med==null?"&ndash;":F.geo_med.toFixed(3)}</td><td class="${F.geo_min!=null&&F.geo_min<.8?"flag":""}">${F.geo_min==null?"&ndash;":F.geo_min.toFixed(3)}</td></tr>`).join("") +
    (D.geo_flags.length ? D.geo_flags.map(g => `<tr><td class="l flag" colspan="4">r&lt;0.8: ${esc(g.tile)} (${esc(g.frame)}) r=${g.r} coverage ${g.cov}</td></tr>`).join("") : `<tr><td class="l" colspan="4">No 5&times;5 crop with r &lt; 0.8.</td></tr>`);
})();
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
