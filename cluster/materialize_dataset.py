"""Download SelimEmirCan/claridi (private HF dataset) and write it to the BBDM
path layout:  <out>/train/A/<input_filename>  and  <out>/train/B/<target_filename>.
Also writes <out>/manifest.csv with tile_id, specimen, piece, part, tissue, scale,
masked, row, col, input_filename, target_filename so the splitter never parses names.
"""
import argparse, io, os, sys
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/local/emir/ClariDi/data/bbdm")
ap.add_argument("--cache", default="/local/emir/ClariDi/data/hf_cache")
args = ap.parse_args()

local = snapshot_download("SelimEmirCan/claridi", repo_type="dataset",
                          cache_dir=args.cache, allow_patterns=["data/*.parquet"])
shards = sorted(os.listdir(os.path.join(local, "data")))
print("shards:", shards, flush=True)

A = os.path.join(args.out, "train", "A"); B = os.path.join(args.out, "train", "B")
os.makedirs(A, exist_ok=True); os.makedirs(B, exist_ok=True)
meta_cols = ["tile_id", "piece", "specimen", "part", "tissue", "scale", "masked",
             "row", "col", "input_filename", "target_filename"]
rows = []
n = 0
for sh in shards:
    pf = pq.ParquetFile(os.path.join(local, "data", sh))
    for batch in pf.iter_batches(batch_size=32):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            fin, ftg = d["input_filename"][i], d["target_filename"][i]
            assert "/" not in fin and "/" not in ftg
            for col, fn, dst in (("input_image", fin, A), ("target_image", ftg, B)):
                img = d[col][i]
                raw = img["bytes"] if isinstance(img, dict) else img
                assert raw, f"empty image for {fn}"
                p = os.path.join(dst, fn)
                if not os.path.exists(p):
                    with open(p, "wb") as f: f.write(raw)
            rows.append({c: d[c][i] for c in meta_cols})
            n += 1
    print(f"{sh}: cumulative {n} tiles", flush=True)

import csv
with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=meta_cols); w.writeheader(); w.writerows(rows)
assert n == len(set(r["tile_id"] for r in rows)) == len(set(r["input_filename"] for r in rows))
print(f"wrote {n} pairs, {len(os.listdir(A))} in A, {len(os.listdir(B))} in B")
print("MATERIALIZE_DONE")
