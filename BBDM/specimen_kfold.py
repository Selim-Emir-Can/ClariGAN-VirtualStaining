"""Specimen-grouped cross-validation splits for ClariDi.

Replaces the patch-level ``stratified_kfold_85_5_10`` (which leaked every specimen
into every fold's train AND test partition).  Here the unit of assignment is the
physical tissue specimen (the ``specimen`` column of the dataset manifest): every
patch of a specimen, at both crop scales, masked or not, follows its specimen into
exactly one partition.  Augmentation is applied on the fly by the training dataset
only, so it never crosses partitions either.

Grouping rule (see dataset card): group by ``specimen``, never by ``piece``.
``Z`` is the second half of ``D``; ``Hpart1`` is the second half of ``H``.  The
manifest's ``specimen`` column already encodes that, so no filename parsing happens
here and the stale ``Z -> H`` merge of the old code is not reproduced.
"""
import csv
import json
import os
from collections import Counter, defaultdict

TISSUES = ("brain", "heart")


def load_manifest(manifest_csv, data_root):
    """Read the sidecar manifest and attach absolute (input, target) paths."""
    records = []
    with open(manifest_csv, newline="") as f:
        for r in csv.DictReader(f):
            r = dict(r)
            r["input_path"] = os.path.join(data_root, "train", "A", r["input_filename"])
            r["target_path"] = os.path.join(data_root, "train", "B", r["target_filename"])
            records.append(r)
    if not records:
        raise ValueError(f"empty manifest: {manifest_csv}")
    missing = [r["input_path"] for r in records if not os.path.exists(r["input_path"])] + \
              [r["target_path"] for r in records if not os.path.exists(r["target_path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest files missing on disk, e.g. {missing[0]}")
    return records


def _specimens_by_tissue(records):
    out = defaultdict(set)
    for r in records:
        out[r["tissue"]].add(r["specimen"])
    return {t: sorted(s) for t, s in out.items()}


def _next_same_tissue(specimen, pool, by_tissue):
    """Deterministic validation pick: the next specimen (cyclically, in sorted order)
    of the same tissue as ``specimen`` that is still in ``pool``."""
    tissue = next(t for t, s in by_tissue.items() if specimen in s)
    ring = by_tissue[tissue]
    i = ring.index(specimen)
    for k in range(1, len(ring) + 1):
        cand = ring[(i + k) % len(ring)]
        if cand in pool:
            return cand
    # no same-tissue specimen left (cannot happen with >=2 per tissue); fall back
    return sorted(pool)[0]


def make_folds(records, scheme="loso", n_folds=5):
    """Return a list of folds. Each fold is a dict with

        fold, test_specimens, val_specimens, train_specimens  (lists of str)
        train, val, test   (lists of (input_path, target_path) tuples)

    scheme="loso":     leave-one-specimen-out, one fold per specimen (11 folds).
    scheme="grouped":  n_folds folds, each holding out ~11/n_folds whole specimens,
                       brain/heart balanced by round-robin over sorted specimens.
    Validation is always ONE whole specimen carved from the training specimens
    (same tissue as the first test specimen, next in sorted order), never a slice
    of the test specimen.
    """
    by_tissue = _specimens_by_tissue(records)
    all_specimens = sorted({r["specimen"] for r in records})

    if scheme == "loso":
        test_sets = [[s] for s in all_specimens]
    elif scheme == "grouped":
        test_sets = [[] for _ in range(n_folds)]
        i = 0
        for t in sorted(by_tissue):          # brain first, then heart
            for s in by_tissue[t]:
                test_sets[i % n_folds].append(s)
                i += 1
    else:
        raise ValueError(scheme)

    folds = []
    for fold_idx, test_specs in enumerate(test_sets):
        pool = set(all_specimens) - set(test_specs)
        val_spec = _next_same_tissue(test_specs[0], pool, by_tissue)
        val_specs = [val_spec]
        train_specs = sorted(pool - set(val_specs))

        def pairs(specs):
            return [(r["input_path"], r["target_path"]) for r in records if r["specimen"] in specs]

        fold = {
            "fold": fold_idx,
            "test_specimens": sorted(test_specs),
            "val_specimens": val_specs,
            "train_specimens": train_specs,
            "train": pairs(set(train_specs)),
            "val": pairs(set(val_specs)),
            "test": pairs(set(test_specs)),
        }
        assert_no_leakage(fold, records)
        folds.append(fold)
    return folds


def assert_no_leakage(fold, records):
    """Fail loudly if any specimen (or any file) appears in more than one partition."""
    parts = {"train": fold["train_specimens"], "val": fold["val_specimens"], "test": fold["test_specimens"]}
    for a in parts:
        for b in parts:
            if a < b:
                overlap = set(parts[a]) & set(parts[b])
                if overlap:
                    raise AssertionError(
                        f"LEAKAGE in fold {fold['fold']}: specimen(s) {sorted(overlap)} appear in both {a} and {b}")
    # file-level check, derived independently of the specimen lists
    spec_of = {r["input_path"]: r["specimen"] for r in records}
    for name in parts:
        seen = {spec_of[p] for p, _ in fold[name]}
        if seen != set(parts[name]):
            raise AssertionError(
                f"LEAKAGE in fold {fold['fold']}: files in '{name}' come from specimens {sorted(seen)}, "
                f"expected {sorted(parts[name])}")
    files = [p for name in parts for p, _ in fold[name]]
    if len(files) != len(set(files)):
        raise AssertionError(f"LEAKAGE in fold {fold['fold']}: a file is listed in more than one partition")
    if len(files) != len(records):
        raise AssertionError(f"fold {fold['fold']} covers {len(files)} files, manifest has {len(records)}")
    return True


def fold_table(folds, records):
    """Rows: fold, partition, specimens, n_patches, n_5x5, n_10x10, n_masked, tissues."""
    by_path = {r["input_path"]: r for r in records}
    rows = []
    for f in folds:
        for part in ("train", "val", "test"):
            rs = [by_path[p] for p, _ in f[part]]
            rows.append({
                "fold": f["fold"],
                "partition": part,
                "specimens": " ".join(f[f"{part}_specimens"]),
                "n_patches": len(rs),
                "n_5x5": sum(r["scale"] == "5x5" for r in rs),
                "n_10x10": sum(r["scale"] == "10x10" for r in rs),
                "n_masked": sum(str(r["masked"]).lower() == "true" for r in rs),
                "n_brain": sum(r["tissue"] == "brain" for r in rs),
                "n_heart": sum(r["tissue"] == "heart" for r in rs),
            })
    return rows


def format_fold_table(rows):
    cols = ["fold", "partition", "specimens", "n_patches", "n_5x5", "n_10x10", "n_masked", "n_brain", "n_heart"]
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(w[c]) for c in cols), "  ".join("-" * w[c] for c in cols)]
    for r in rows:
        lines.append("  ".join(str(r[c]).ljust(w[c]) for c in cols))
    return "\n".join(lines)


def save_fold_assignments(folds, records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = fold_table(folds, records)
    with open(os.path.join(out_dir, "fold_assignments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "fold_assignments.json"), "w") as f:
        json.dump([{k: v for k, v in fd.items() if k.endswith("_specimens") or k == "fold"} |
                   {f"{p}_files": [os.path.basename(a) for a, _ in fd[p]] for p in ("train", "val", "test")}
                   for fd in folds], f, indent=1)
    with open(os.path.join(out_dir, "fold_assignments.txt"), "w") as f:
        f.write(format_fold_table(rows) + "\n")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="dir containing train/A, train/B")
    ap.add_argument("--manifest", default=None, help="default: <data_root>/manifest.csv")
    ap.add_argument("--scheme", choices=["loso", "grouped"], default="loso")
    ap.add_argument("--n_folds", type=int, default=5, help="only for --scheme grouped")
    ap.add_argument("--out", default=None, help="where to write fold_assignments.{csv,json,txt}")
    a = ap.parse_args()
    recs = load_manifest(a.manifest or os.path.join(a.data_root, "manifest.csv"), a.data_root)
    print(f"{len(recs)} patches, specimens: {dict(Counter(r['specimen'] for r in recs))}")
    folds = make_folds(recs, a.scheme, a.n_folds)
    rows = fold_table(folds, recs)
    print(format_fold_table(rows))
    for f in folds:
        assert_no_leakage(f, recs)
    print(f"\nleakage assertion passed for all {len(folds)} folds ({a.scheme})")
    if a.out:
        save_fold_assignments(folds, recs, a.out)
        print("saved to", a.out)
