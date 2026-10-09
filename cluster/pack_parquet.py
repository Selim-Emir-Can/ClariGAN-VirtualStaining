"""Pack deliverable PNGs into the parquet layout pinned in the HF repo README
(SelimEmirCan/claridi-results), optionally uploading.

  python pack_parquet.py --experiment claridi_primary --fold 0 [--upload]
  python pack_parquet.py --experiment claridi_primary --all [--upload]     # every fold on disk
  python pack_parquet.py --ceilings [--upload]                            # both VQGAN ceilings
  python pack_parquet.py --static [--upload]      # fold_assignments.*, manifest.csv, data_256/
  split v2: python pack_parquet.py --root deliverables_spatial_v2 --out_root deliverables_spatial_v2/parquet \
              --split_csv data/splits/model_design_exp_split_v2.csv --experiment sp_primary --all

samples parquet: <deliverables>/parquet/<experiment>/samples/fold_<k>.parquet, one row per
(tile, generation): experiment, fold(int32), tile_id, specimen, tissue, scale, masked(bool),
gen_idx(int32), seed(int32), png(binary, bare 256x256 PNG bytes unchanged).
Also mirrored: <experiment>/config.yaml, <experiment>/seeds_fold_<k>.json, and
<experiment>/timing.csv = concatenation of the timing_fold_<k>.csv present so far.
ceilings: ceilings/vqgan_finetuned.parquet, ceilings/vqgan_stock.parquet (tile_id, png).
"""
import argparse, csv, glob, io, json, os, re, sys
import pyarrow as pa, pyarrow.parquet as pq

ROOT = "/local/emir/ClariDi"
DELIV = os.path.join(ROOT, "deliverables")
REPO = "SelimEmirCan/claridi-results"
MANIFEST = os.path.join(ROOT, "data/bbdm256/manifest.csv")
FOLDS = os.path.join(ROOT, "results/fold_assignments.csv")

def manifest():
    return {r["tile_id"]: r for r in csv.DictReader(open(MANIFEST))}

def test_specimens(fold, scheme_csv=FOLDS):
    for r in csv.DictReader(open(scheme_csv)):
        if int(r["fold"]) == fold and r["partition"] == "test":
            return set(r["specimens"].split())
    raise SystemExit(f"fold {fold} not in {scheme_csv}")

def split_test(fold, split_csv):
    """tile ids whose role in this fold is 'test' (spatial split CSV with role_f<k> columns)"""
    return {r["tile_id"] for r in csv.DictReader(open(split_csv)) if r[f"role_f{fold}"] == "test"}

def pack_fold(experiment, fold, root, out_root, fold_csv=FOLDS, split_csv=None):
    exp = os.path.join(root, experiment)
    samp = os.path.join(exp, "samples", f"fold_{fold}")
    pngs = sorted(glob.glob(os.path.join(samp, "*_gen*.png")))
    if not pngs:
        print(f"{experiment} fold {fold}: no PNGs under {samp}, skipped", flush=True); return None
    man = manifest()
    test = split_test(fold, split_csv) if split_csv else test_specimens(fold, fold_csv)
    seeds_p = os.path.join(exp, f"seeds_fold_{fold}.json")
    seeds = json.load(open(seeds_p)) if os.path.exists(seeds_p) else {}
    gen_seed = {int(k): int(v) for k, v in seeds.get("gen_seed", {}).items()}
    rx = re.compile(r"^(.*)_gen(\d+)\.png$")
    cols = {k: [] for k in ["experiment", "fold", "tile_id", "specimen", "tissue", "scale", "masked", "gen_idx", "seed", "png"]}
    for p in pngs:
        m = rx.match(os.path.basename(p)); t, j = m.group(1), int(m.group(2))
        r = man.get(t); assert r is not None, f"{t} not in manifest"
        if split_csv: assert t in test, f"{t} is not a test tile of fold {fold} in {split_csv}"
        else: assert r["specimen"] in test, f"{t}: specimen {r['specimen']} is not a test specimen of fold {fold} ({sorted(test)})"
        cols["experiment"].append(experiment); cols["fold"].append(fold); cols["tile_id"].append(t)
        cols["specimen"].append(r["specimen"]); cols["tissue"].append(r["tissue"]); cols["scale"].append(r["scale"])
        cols["masked"].append(r["masked"] == "True"); cols["gen_idx"].append(j)
        cols["seed"].append(gen_seed.get(j, 1234 + j)); cols["png"].append(open(p, "rb").read())
    schema = pa.schema([("experiment", pa.string()), ("fold", pa.int32()), ("tile_id", pa.string()),
                        ("specimen", pa.string()), ("tissue", pa.string()), ("scale", pa.string()),
                        ("masked", pa.bool_()), ("gen_idx", pa.int32()), ("seed", pa.int32()), ("png", pa.binary())])
    table = pa.table(cols, schema=schema)
    out_dir = os.path.join(out_root, experiment, "samples"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"fold_{fold}.parquet")
    pq.write_table(table, out, compression="none")   # PNG bytes are already compressed
    n_t, n_g = len(set(cols["tile_id"])), len(set(cols["gen_idx"]))
    print(f"{experiment} fold {fold}: {n_t} tiles x {n_g} gens, {table.num_rows} rows -> {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
    # side files
    side = []
    cfg = os.path.join(exp, "config.yaml")
    if os.path.exists(cfg): side.append((cfg, f"{experiment}/config.yaml"))
    if os.path.exists(seeds_p): side.append((seeds_p, f"{experiment}/seeds_fold_{fold}.json"))
    tims = sorted(glob.glob(os.path.join(exp, "timing_fold_*.csv")), key=lambda p: int(re.search(r"_(\d+)\.csv$", p).group(1)))
    if tims:
        tpath = os.path.join(out_root, experiment, "timing.csv"); hdr = None
        with open(tpath, "w", newline="") as f:
            for tp in tims:
                lines = open(tp).read().splitlines()
                if hdr is None: hdr = lines[0]; f.write(hdr + "\n")
                f.write("\n".join(lines[1:]) + "\n")
        side.append((tpath, f"{experiment}/timing.csv"))
    return [(out, f"{experiment}/samples/fold_{fold}.parquet")] + side

def pack_ceilings(out_root):
    files = []; man = manifest()
    for variant, d in (("vqgan_finetuned", "finetuned"), ("vqgan_stock", "stock")):
        pngs = sorted(glob.glob(os.path.join(DELIV, "ceilings", d, "*.png")))
        ids = [os.path.basename(p)[:-4] for p in pngs]
        assert all(t in man for t in ids), "ceiling tile not in manifest"
        table = pa.table({"tile_id": pa.array(ids, pa.string()), "png": pa.array([open(p, "rb").read() for p in pngs], pa.binary())})
        os.makedirs(os.path.join(out_root, "ceilings"), exist_ok=True)
        out = os.path.join(out_root, "ceilings", f"{variant}.parquet"); pq.write_table(table, out, compression="none")
        print(f"ceilings {variant}: {len(ids)} tiles -> {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
        files.append((out, f"ceilings/{variant}.parquet"))
    return files

def upload(files, folders=()):
    from huggingface_hub import HfApi
    api = HfApi()
    for local, remote in files:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO, repo_type="dataset",
                        commit_message=f"add {remote}")
        print(f"  uploaded {remote} ({os.path.getsize(local)/1e6:.1f} MB)", flush=True)
    for local, remote in folders:
        api.upload_folder(folder_path=local, path_in_repo=remote, repo_id=REPO, repo_type="dataset",
                          commit_message=f"add {remote}/")
        print(f"  uploaded folder {remote}/", flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DELIV, help="deliverables root holding <experiment>/samples/fold_<k>/")
    ap.add_argument("--out_root", default=os.path.join(DELIV, "parquet"))
    ap.add_argument("--experiment"); ap.add_argument("--fold", type=int); ap.add_argument("--all", action="store_true")
    ap.add_argument("--fold_csv", default=FOLDS, help="fold table to validate test specimens against (grouped-5 pixel run uses its own)")
    ap.add_argument("--split_csv", help="spatial split CSV (role_f<k>): validate each tile is a test tile of its fold")
    ap.add_argument("--ceilings", action="store_true"); ap.add_argument("--static", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    files, folders = [], []
    if a.experiment:
        if a.all:
            sd = os.path.join(a.root, a.experiment, "samples")
            folds = sorted(int(d.split("_")[1]) for d in os.listdir(sd)) if os.path.isdir(sd) else []
        elif a.fold is not None: folds = [a.fold]
        else: sys.exit("pass --fold k or --all")
        for k in folds:
            r = pack_fold(a.experiment, k, a.root, a.out_root, a.fold_csv, a.split_csv)
            if r: files += r
    if a.ceilings: files += pack_ceilings(a.out_root)
    if a.static:
        for fn in ("fold_assignments.csv", "fold_assignments.json", "fold_assignments.txt"):
            files.append((os.path.join(ROOT, "results", fn), fn))
        files.append((MANIFEST, "manifest.csv"))
        folders.append((os.path.join(ROOT, "data/bbdm256"), "data_256"))
    if a.upload and (files or folders): upload(files, folders)

if __name__ == "__main__":
    main()
