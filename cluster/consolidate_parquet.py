"""Build the consolidated deliverable layout: ONE parquet + 3 side files per experiment.

Rationale: one parquet per (experiment, fold) meant up to 99 sample files plus ~11 seeds files
per experiment, so every HF sync was commit-heavy (we hit the 128-commits/hour limit once).
Consolidated it is 4 files per experiment, 9 experiments -> ~36 files + ceilings + root files.

  python consolidate_parquet.py --dry           # show what would be built
  python consolidate_parquet.py                 # build deliverables/parquet_merged/<exp>/...
  python consolidate_parquet.py --only-complete # only experiments with all 11 folds
  python consolidate_parquet.py --v2 [--only-complete]  # split v2: deliverables_spatial_v2/parquet -> parquet_merged, 5 folds

Output per experiment under deliverables/parquet_merged/<exp>/:
  samples.parquet   all folds; rows sorted (fold, tile_id, gen_idx); ~1 row group per fold so a
                    reader can pull one fold cheaply; PNG bytes stored uncompressed (already PNG)
  config.yaml       copied verbatim
  timing.csv        per-fold timing rows concatenated (one header)
  seeds.json        {"fold_<k>": <contents of seeds_fold_<k>.json>, ...}
"""
import argparse, glob, json, os, re, shutil, sys
import pyarrow as pa, pyarrow.parquet as pq

ROOT = "/local/emir/ClariDi"
DELIV = os.path.join(ROOT, "deliverables")
PQ = os.path.join(DELIV, "parquet")            # per-fold packer output (source of truth)
OUT = os.path.join(DELIV, "parquet_merged")    # consolidated upload root
EXPERIMENTS = ["claridi_primary", "claridi_stock_vqgan", "trainable_encoder", "pixel_space",
               "pix2pix", "cwgan", "unet_l1", "claridi_primary_seed5678", "claridi_primary_seed9012"]
NFOLD = 11


def fold_of(p):
    return int(re.search(r"fold_(\d+)", os.path.basename(p)).group(1))


def build(exp, dry=False, only_complete=False):
    files = sorted(glob.glob(os.path.join(PQ, exp, "samples", "fold_*.parquet")), key=fold_of)
    if not files:
        return None
    folds = [fold_of(f) for f in files]
    if only_complete and len(folds) != NFOLD:
        print(f"  {exp:26s} {len(folds)}/{NFOLD} folds — skipped (--only-complete)"); return None
    src_mb = sum(os.path.getsize(f) for f in files) / 1e6
    if dry:
        print(f"  {exp:26s} {len(folds):2d} folds {src_mb:7.1f} MB -> {exp}/samples.parquet + 3 side files")
        return None

    od = os.path.join(OUT, exp); os.makedirs(od, exist_ok=True)

    # --- samples.parquet
    t = pa.concat_tables([pq.read_table(f) for f in files])
    idx = pa.compute.sort_indices(t, sort_keys=[("fold", "ascending"), ("tile_id", "ascending"),
                                                ("gen_idx", "ascending")])
    t = t.take(idx)
    rgs = max(64, t.num_rows // NFOLD)      # ~one row group per fold; floor keeps tiny GAN files sane
    out = os.path.join(od, "samples.parquet")
    pq.write_table(t, out, compression="none", row_group_size=rgs)

    # --- config.yaml (verbatim)
    cfg = os.path.join(DELIV, exp, "config.yaml")
    if os.path.exists(cfg): shutil.copyfile(cfg, os.path.join(od, "config.yaml"))

    # --- timing.csv (concatenate per-fold csvs, one header, fold order)
    tims = sorted(glob.glob(os.path.join(DELIV, exp, "timing_fold_*.csv")), key=fold_of)
    if tims:
        with open(os.path.join(od, "timing.csv"), "w", newline="") as f:
            hdr = None
            for tp in tims:
                lines = open(tp).read().splitlines()
                if not lines: continue
                if hdr is None: hdr = lines[0]; f.write(hdr + "\n")
                f.write("\n".join(lines[1:]) + "\n")

    # --- seeds.json (all folds in one dict)
    seeds = {}
    for sp in sorted(glob.glob(os.path.join(DELIV, exp, "seeds_fold_*.json")), key=fold_of):
        if os.path.getsize(sp) > 0:
            seeds[f"fold_{fold_of(sp)}"] = json.load(open(sp))
    if seeds:
        json.dump(seeds, open(os.path.join(od, "seeds.json"), "w"), indent=1)

    got = os.path.getsize(out) / 1e6
    tiles = len(set(t.column("tile_id").to_pylist()))
    print(f"  {exp:26s} {len(folds):2d} folds, {t.num_rows:5d} rows, {tiles:3d} tiles -> "
          f"samples.parquet {got:.1f} MB ({pq.ParquetFile(out).num_row_groups} row groups), "
          f"seeds.json x{len(seeds)}, timing.csv x{len(tims)}", flush=True)
    return od


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--only-complete", action="store_true")
    p.add_argument("--v2", action="store_true", help="spatial split v2 deliverables (5 folds)")
    a = p.parse_args()
    global DELIV, PQ, OUT, EXPERIMENTS, NFOLD
    if a.v2:
        DELIV = os.path.join(ROOT, "deliverables_spatial_v2"); PQ = os.path.join(DELIV, "parquet")
        OUT = os.path.join(DELIV, "parquet_merged"); NFOLD = 5
        EXPERIMENTS = ["sp_primary", "sp_stock_vqgan", "sp_specimen_cond", "sp_refA_stained", "sp_refB_unstained",
                       "pixel_space", "trainable_encoder", "cwgan", "pix2pix"]
    made = [m for e in EXPERIMENTS if (m := build(e, a.dry, a.only_complete))]
    if not a.dry:
        print(f"\n{len(made)} experiments consolidated under {OUT}")


if __name__ == "__main__":
    main()
