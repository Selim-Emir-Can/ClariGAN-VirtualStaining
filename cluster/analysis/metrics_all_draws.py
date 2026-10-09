"""Predefined aggregate over ALL five draws (no best-of-n, no example selection), spatial split v1 or v2.

Protocol (fixed before looking at the numbers, Oct 7 2026):
  * Every one of the 753 tiles is a held-out prediction (tested once across the 5 folds), and every
    tile has 5 independent draws per method (seeds 1234-1238). All 5 draws are scored.
    The GANs (cWGAN, pix2pix; v2 only) give one output per tile (gen0, a single seeded inference pass; pix2pix
    keeps dropout on at inference, cWGAN runs in train mode, so neither is strictly deterministic): they are
    scored with the same per-specimen aggregate on that single output and reported without an SD.
  * Metrics per draw vs C&SF at 256 px: LPIPS (AlexNet; from lpips_folds_0_1_2_3_4.json), PSNR, SSIM
    (RGB, data_range 255), and colour error |G/(R+G) of output - G/(R+G) of C&SF| (tile means).
  * Per draw j: score = mean over tiles within each specimen, then the unweighted mean over the
    11 specimens (specimen-macro, so large specimens do not dominate). This gives 5 numbers per
    method; we report their mean (= the draw-averaged score) and the SD across draws.
  * Secondary: tile-micro mean (all tiles equally weighted), per tissue, per specimen.
  * Sensitivity: the same without specimen D part 0 (its fold assignment was degenerate in split v1; in v2 it
    is banded per tile with per-fold exclusion of overlapping 6x6/10x10 tiles, kept as a sensitivity check).
  * Variants marked (oracle) / (extra input) use information unavailable to plain UA -> C&SF.
Writes analysis/metrics_all_draws.json (per tile, per draw) and analysis/metrics_all_draws.md
(--v2: analysis/v2/, deliverables_spatial_v2, split v2; methods without a complete set of outputs are skipped).
  python analysis/metrics_all_draws.py [--v2]
"""
import csv, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

R = "/local/emir/ClariDi"; V2 = "--v2" in sys.argv
D = f"{R}/deliverables_spatial_v2" if V2 else f"{R}/deliverables_spatial"
A = f"{R}/analysis/v2" if V2 else f"{R}/analysis"
SPLIT = f"{R}/data/splits/model_design_exp_split_v2.csv" if V2 else f"{R}/data/splits/model_design_exp_split.csv"
EXPS = [("sp_stock_vqgan", "Vanilla L-BBDM"), ("sp_primary", "Ours (L-BBDM)"),
        ("sp_specimen_cond", "+ specimen label (oracle)"), ("sp_refA_stained", "+ A stained refs (extra input)"),
        ("sp_refB_unstained", "+ B unstained ctx")]
if V2: EXPS += [("pixel_space", "Pixel-space BBDM"), ("trainable_encoder", "Ours + trainable encoder"),
                ("cwgan", "cWGAN (1 draw)"), ("pix2pix", "pix2pix (1 draw)")]
NDRAW = {e: 1 if e in ("cwgan", "pix2pix") else 5 for e, _ in EXPS}
METRICS = [("lpips", "LPIPS ↓", 3), ("psnr", "PSNR ↑", 2), ("ssim", "SSIM ↑", 3), ("green_err", r"\|Δ green share\| ↓", 3)]


def load(p):
    return np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS), dtype=np.float64)


def green(a):
    return a[..., 1].mean() / max(a[..., 0].mean() + a[..., 1].mean(), 1e-6)


def score_tile(job):
    tid, fold, gt_path = job
    gt = load(gt_path); g0 = green(gt); out = {}
    for e, _ in EXPS:
        rs = []
        for j in range(NDRAW[e]):
            x = load(f"{D}/{e}/samples/fold_{fold}/{tid}_gen{j}.png")
            rs.append(dict(psnr=peak_signal_noise_ratio(gt, x, data_range=255),
                           ssim=structural_similarity(gt, x, channel_axis=2, data_range=255),
                           green_err=abs(green(x) - g0)))
        out[e] = rs
    return tid, out


def main():
    man = {r["tile_id"]: r for r in csv.DictReader(open(f"{R}/data/bbdm256/manifest.csv", newline=""))}
    man = {k.strip(): {a.strip(): b.strip() for a, b in v.items()} for k, v in man.items()}
    global EXPS
    split = {r["tile_id"]: r for r in csv.DictReader(open(SPLIT))}
    lp = json.load(open(f"{A}/lpips_folds_0_1_2_3_4.json"))
    have = [e for e, _ in EXPS if all(e in lp[t] for t in split)]
    if V2: EXPS = [x for x in EXPS if x[0] in have]       # incomplete methods are left out, not partially scored
    assert len(have) == len(EXPS), f"LPIPS missing for {[e for e, _ in EXPS if e not in have]}"
    print("methods:", [e for e, _ in EXPS])
    jobs = [(t, int(split[t]["band"]), f"{R}/data/bbdm256/train/B/{man[t]['target_filename']}") for t in sorted(split)]
    with ProcessPoolExecutor(48) as ex:
        res = dict(ex.map(score_tile, jobs, chunksize=4))
    tiles = {}
    for t, out in res.items():
        for e, _ in EXPS:
            for j in range(NDRAW[e]): out[e][j]["lpips"] = lp[t][e][j]
        tiles[t] = dict(specimen=split[t]["specimen"], tissue=split[t]["tissue"], frame=split[t]["frame"],
                        scale=split[t]["scale"], fold=int(split[t]["band"]), draws=out)
    json.dump(tiles, open(f"{A}/metrics_all_draws.json", "w"))

    def agg(sel, macro=True):
        """per method, metric: 5 per-draw scores (specimen-macro or tile-micro over the selected tiles)"""
        res = {}
        for e, _ in EXPS:
            for m, *_ in METRICS:
                per_draw = []
                for j in range(NDRAW[e]):
                    if macro:
                        specs = sorted({tiles[t]["specimen"] for t in sel})
                        per_draw.append(np.mean([np.mean([tiles[t]["draws"][e][j][m] for t in sel if tiles[t]["specimen"] == s]) for s in specs]))
                    else:
                        per_draw.append(np.mean([tiles[t]["draws"][e][j][m] for t in sel]))
                res[e, m] = per_draw
        return res

    def cell(v, d):
        return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"
    L = [f"# ClariDi spatial split ({'v2' if V2 else 'v1'}): all five draws, predefined aggregate", "",
         "Protocol: see the docstring of analysis/metrics_all_draws.py. Cells are mean ± SD across the 5 draws of the "
         "specimen-macro mean (each draw scored separately; no best-of-n). 256 px, vs C&SF."
         + (" GANs (cWGAN, pix2pix): one output per tile (single seeded inference pass), so a single value, no SD." if V2 else ""), ""]
    def table(title, sel, macro=True):
        a = agg(sel, macro)
        L.append(f"## {title} (n = {len(sel)} tiles, {len({tiles[t]['specimen'] for t in sel})} specimens)"); L.append("")
        L.append("| method | " + " | ".join(h for _, h, _ in METRICS) + " |"); L.append("|---|" + "---|" * len(METRICS))
        for e, name in EXPS:
            L.append(f"| {name} | " + " | ".join(cell(a[e, m], d) for m, _, d in METRICS) + " |")
        L.append("")
    allt = sorted(tiles); noD0 = [t for t in allt if tiles[t]["frame"] != "D_p0"]
    table("All tiles, specimen-macro (primary)", allt)
    table("All tiles, tile-micro", allt, macro=False)
    table("Without D part 0, specimen-macro (sensitivity)", noD0)
    for tis in ("brain", "heart"):
        table(f"{tis.capitalize()}, specimen-macro", [t for t in allt if tiles[t]["tissue"] == tis])
    L.append("## Per specimen: LPIPS ↓, mean ± SD across draws"); L.append("")
    L.append("| specimen | n | " + " | ".join(n for _, n in EXPS) + " |"); L.append("|---|---|" + "---|" * len(EXPS))
    for s in sorted({v["specimen"] for v in tiles.values()}):
        sel = [t for t in allt if tiles[t]["specimen"] == s]; a = agg(sel, macro=False)
        L.append(f"| {s} | {len(sel)} | " + " | ".join(cell(a[e, "lpips"], 3) for e, _ in EXPS) + " |")
    open(f"{A}/metrics_all_draws.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
