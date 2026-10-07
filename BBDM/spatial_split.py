"""Spatially blocked 5-fold split for ClariDi ("model_design_exp_split").

Every specimen is represented in train, val and test of every fold; what is held out is a
spatial band of each specimen, not a whole specimen.

Geometry (verified on pixels, correlation ~1.00 on shared regions): each piece image is cut
into two non-overlapping grids. A tile's box in its piece frame is (col*w, row*h, w, h) with
its native size w x h. A 5x5 crop covers a 2x2 block of 10x10 crops (specimen D's "5x5" grid
is really 6x6, so its crops straddle 10x10 cells). Z is D's second half and shares D's
part-1 frame; Hpart1 is H's second half with its own frame. Masked tiles are extra cells in
the same grids.

Unit of assignment: a connected component of the cross-scale pixel-overlap graph (overlaps
thinner than 3% of a tile side are rounding slivers and are ignored). A 5x5 crop and every
10x10 crop inside it therefore always land in the same partition: no cross-scale leakage and
no shared pixels between partitions. Units are ordered by centroid along the frame's longer
axis and cut into 5 contiguous bands of ~equal tile count. Fold k: test = band k,
val = the neighbouring band (k+1, or 3 for k=4), train = the other three bands.

Non-nested frames (v2, Oct 2026): where the two grids are not nested (D part 0, 6x6 vs 10x10)
the overlap graph collapses into one or two giant components, which made whole bands empty.
A frame whose largest component holds more than 1/5 of its tiles is therefore banded tile by
tile (block = tile) and purged per fold: a val tile that overlaps a test tile, and a train tile
that overlaps a test or val tile, is excluded from that fold (role "excluded"). Every tile is
still tested exactly once and no partitions share pixels. Per-fold roles are written as
role_f0..role_f4; `unit` stays the overlap component (used to keep references off a tile's own
pixels), `block` is the unit of band assignment. v2 also measures the long axis on the tissue
extent (v1 used the frame origin, which picked the short axis for D part 0 and J).

  python spatial_split.py --raw_root /local/emir/ClariDi/data/bbdm \
      --out /local/emir/ClariDi/data/splits/model_design_exp_split.csv
"""
import argparse, csv, os
from collections import defaultdict, Counter
from PIL import Image

N_BANDS = 5
SLIVER = 0.03


def val_band(k, n=N_BANDS):
    return k + 1 if k + 1 < n else k - 1


def frame_of(r):
    if r["piece"] == "Z":
        return f"{r['specimen']}_p1"
    return f"{r['specimen']}_p{1 if r['part'] == '1' else 0}"


def overlap(a, b):
    """Overlap rectangle size of two (x, y, w, h) boxes, ignoring rounding slivers."""
    ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    tol_x = SLIVER * min(a[2], b[2]); tol_y = SLIVER * min(a[3], b[3])
    return ox > tol_x and oy > tol_y


def build(manifest, raw_root):
    rows = list(csv.DictReader(open(manifest)))
    for r in rows:
        w, h = Image.open(os.path.join(raw_root, "train", "B", r["target_filename"])).size
        r["x"], r["y"], r["w"], r["h"] = int(r["col"]) * w, int(r["row"]) * h, w, h
        r["frame"] = frame_of(r)
    by = defaultdict(list)
    for r in rows: by[r["frame"]].append(r)
    for f, ts in by.items():
        box = lambda t: (t["x"], t["y"], t["w"], t["h"])
        parent = list(range(len(ts)))
        def find(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                if overlap(box(ts[i]), box(ts[j])): parent[find(i)] = find(j)
        units = defaultdict(list)
        for i, t in enumerate(ts): units[find(i)].append(t)
        # tissue extent (v2: v1 measured from the frame origin, which picked the short axis for D part 0 and J)
        W = max(t["x"] + t["w"] for t in ts) - min(t["x"] for t in ts)
        H = max(t["y"] + t["h"] for t in ts) - min(t["y"] for t in ts)
        ax, span = ("x", "w") if W >= H else ("y", "h")
        def centre(u):   # area-weighted centroid along the long axis
            a = sum(t["w"] * t["h"] for t in u)
            return sum((t[ax] + t[span] / 2) * t["w"] * t["h"] for t in u) / a
        n = len(ts)
        for ui, u in enumerate(sorted(units.values(), key=centre)):
            for t in u: t["unit"] = f"{f}_u{ui}"
        nested = max(len(u) for u in units.values()) <= n / N_BANDS
        blocks = list(units.values()) if nested else [[t] for t in ts]
        cum = 0
        for bi, u in enumerate(sorted(blocks, key=centre)):
            band = min(N_BANDS - 1, int(N_BANDS * (cum + len(u) / 2) / n))
            for t in u: t["band"], t["block"], t["nested"] = band, f"{f}_b{bi}", nested
            cum += len(u)
        for k in range(N_BANDS):
            role = {id(t): "test" if t["band"] == k else "val" if t["band"] == val_band(k) else "train" for t in ts}
            box = lambda t: (t["x"], t["y"], t["w"], t["h"])
            for drop, against in (("val", ("test",)), ("train", ("test", "val"))):
                for t in ts:
                    if role[id(t)] == drop and any(role[id(o)] in against and overlap(box(t), box(o)) for o in ts):
                        role[id(t)] = "excluded"
            for t in ts: t[f"role_f{k}"] = role[id(t)]
    return rows


def role(r, k):
    """Role of a split-CSV row in fold k (v1 CSVs have no role columns)."""
    if f"role_f{k}" in r and r[f"role_f{k}"]:
        return r[f"role_f{k}"]
    return "test" if int(r["band"]) == k else "val" if int(r["band"]) == val_band(k) else "train"


def assert_spatial_no_leakage(rows):
    """Per fold: no pixel overlap (beyond slivers) between tiles of different partitions,
    every tile in exactly one partition, test bands cover every tile exactly once."""
    by = defaultdict(list)
    for r in rows: by[r["frame"]].append(r)
    for k in range(N_BANDS):
        part = lambda r: role(r, k)
        for f, ts in by.items():
            for i, a in enumerate(ts):
                for b in ts[i + 1:]:
                    if "excluded" not in (part(a), part(b)) and part(a) != part(b) and overlap(
                            (int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"])),
                            (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))):
                        raise AssertionError(f"LEAKAGE fold {k}: {a['tile_id']} ({part(a)}) overlaps "
                                             f"{b['tile_id']} ({part(b)})")
    for r in rows:
        assert sum(role(r, k) == "test" for k in range(N_BANDS)) == 1, f"{r['tile_id']} not tested exactly once"
    ids = [r["tile_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate tile ids"
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="native-resolution data (train/A, train/B, manifest.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = build(os.path.join(a.raw_root, "manifest.csv"), a.raw_root)
    assert_spatial_no_leakage(rows)
    cols = ["tile_id", "specimen", "tissue", "scale", "masked", "frame", "unit", "block", "band",
            *[f"role_f{k}" for k in range(N_BANDS)], "x", "y", "w", "h"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} tiles -> {a.out}; leakage assertion PASSED for all {N_BANDS} folds")
    print("fold  test  val  train  excl   (test per specimen)")
    for k in range(N_BANDS):
        c = Counter(role(r, k) for r in rows)
        ps = Counter(r["specimen"] for r in rows if role(r, k) == "test")
        print(f"{k:>4} {c['test']:>5} {c['val']:>4} {c['train']:>6} {c['excluded']:>5}   " + " ".join(f"{s}:{ps[s]}" for s in sorted(ps)))
    for f in sorted({r["frame"] for r in rows if not r["nested"]}):
        print(f"non-nested frame {f} (tile-level bands, purged):  test/val/train/excluded per fold  " + "  ".join(
            "/".join(str(sum(role(r, k) == p for r in rows if r["frame"] == f)) for p in ("test", "val", "train", "excluded"))
            for k in range(N_BANDS)))
