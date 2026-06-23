"""K-fold orchestrator for cWGAN baseline.

Uses the SAME stratified k-fold splits as BBDM/k-fold_validation.py (the
stratified_kfold_85_5_10 function is duplicated below verbatim to avoid
pulling in BBDM's heavy training imports) AND the SAME augmentation pipeline
(BBDM's CustomAlignedDataset, wrapped by bbdm_aligned_dataset.py inside the
cwgan repo), then drives the upstream cwgan train.py / test.py via subprocess.

Writes per-fold list files of (R1_path, R3_path) pairs and runs the GAN
training with:
    --dataset_mode bbdm_aligned --dataroot <list_file>
    --val_dataroot <val_list_file>   # for per-epoch val L1 logging (no checkpoint selection)

Custom dataset class:
    baselines/cwgan/data/bbdm_aligned_dataset.py

NOTE on flags: cwgan is an older fork of pytorch-CycleGAN-and-pix2pix and uses
--niter / --niter_decay instead of --n_epochs / --n_epochs_decay.
"""

import os
import re
import sys
import subprocess
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split


# -------- paths --------
CWGAN_DIR = r"C:\Users\ammic\Desktop\ClariGAN-DL\baselines\cwgan"
DATASET_TRAIN_DIR = r"C:\Users\ammic\Desktop\BBDM-kfold\train"  # has A/ and B/ subdirs
FOLD_LISTS_DIR = os.path.join(CWGAN_DIR, "fold_lists")

# -------- training hyperparameters --------
K = 10
SEED = 42
NITER = 25             # initial-lr epochs (matches pix2pix's n_epochs)
NITER_DECAY = 25       # linearly-decay epochs (total = 50, matching BBDM)
LOAD_SIZE = 256
CROP_SIZE = 256
BATCH_SIZE = 4         # cwgan's example script uses 4; keep that default
GPU_IDS = "0"
NETG = "unet_256"
NGF = 144              # generator features. Default 64 -> ~54M G params.
                       # Param scaling ~ ngf^2: ngf=144 -> ~273M G (target: BBDM ~288M trainable).
                       # Verify on first run from the "[Network G] Total number of parameters" line.


def stratified_kfold_85_5_10(train_dir, k=10, seed=42):
    """Duplicated VERBATIM from BBDM/k-fold_validation.py to keep splits
    bit-identical without depending on the BBDM training stack."""
    input_dir = os.path.join(train_dir, 'A')
    gt_dir = os.path.join(train_dir, 'B')

    r1_files = [f for f in os.listdir(input_dir) if f.startswith("R1") and os.path.isfile(os.path.join(input_dir, f))]
    data = []

    pattern = re.compile(r'R1-([A-Za-z]+(?:part\d+)?).*?_(5x5|10x10)')

    for filename in r1_files:
        match = pattern.search(filename)
        if not match:
            continue
        letter = match.group(1)
        crop = match.group(2)
        label = f"{letter}_{crop}"
        r3_filename = filename.replace("R1", "R3", 1)
        r1_path = os.path.join(input_dir, filename)
        r3_path = os.path.join(gt_dir, r3_filename)
        if os.path.exists(r3_path):
            data.append({"r1": r1_path, "r3": r3_path, "letter": letter, "crop": crop, "original_label": label})

    original_labels = [item['original_label'] for item in data]
    label_counts = Counter(original_labels)

    for item in data:
        label = item['original_label']
        letter = item['letter']
        crop = item['crop']
        count = label_counts[label]
        if count < k:
            if letter == "P":
                item["letter"] = "D"
            elif letter == "Z":
                item["letter"] = "H"
            elif crop == "5x5":
                counterpart_label = f"{letter}_10x10"
                if label_counts.get(counterpart_label, 0) >= k:
                    item["crop"] = "10x10"
        item["label"] = f"{item['letter']}_{item['crop']}"

    final_labels = [item["label"] for item in data]
    pairs = [(item["r1"], item["r3"]) for item in data]

    final_counts = Counter(final_labels)
    too_small = {lbl: c for lbl, c in final_counts.items() if c < k}
    if too_small:
        raise ValueError(
            f"After merging, the following classes still have fewer than {k} samples:\n" +
            "\n".join([f"{lbl}: {c}" for lbl, c in too_small.items()])
        )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    all_indices = list(skf.split(pairs, final_labels))

    folds = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(all_indices):
        trainval_data = [pairs[i] for i in trainval_idx]
        trainval_labels = [final_labels[i] for i in trainval_idx]
        test_data = [pairs[i] for i in test_idx]
        val_ratio = 1 / 18  # ~5%
        train_data, val_data, _, _ = train_test_split(
            trainval_data, trainval_labels,
            test_size=val_ratio, stratify=trainval_labels, random_state=seed,
        )
        folds.append({"fold": fold_idx, "train": train_data, "val": val_data, "test": test_data})
    return folds


def write_list_file(pairs, fold_num, phase, lists_dir):
    path = os.path.join(lists_dir, f"fold_{fold_num}_{phase}.txt")
    with open(path, "w") as f:
        for r1, r3 in pairs:
            f.write(f"{r1},{r3}\n")
    print(f"  wrote {path}  ({len(pairs)} pairs)")
    return path


def run_train(fold_idx, train_list, val_list):
    """Train cwgan on the train list. val_list, if non-empty, is passed via
    --val_dataroot so train.py logs per-epoch val L1 (no checkpoint selection)."""
    name = f"clarigan_cwgan_fold_{fold_idx}"
    cmd = [
        sys.executable, "train.py",
        "--dataroot", train_list,
        "--val_dataroot", val_list,
        "--name", name,
        "--model", "pix2pix",          # cwgan reuses pix2pix model class with WGAN losses
        "--dataset_mode", "bbdm_aligned",
        "--direction", "AtoB",
        "--netG", NETG,
        "--ngf", str(NGF),
        "--load_size", str(LOAD_SIZE),
        "--crop_size", str(CROP_SIZE),
        "--batch_size", str(BATCH_SIZE),
        "--niter", str(NITER),
        "--niter_decay", str(NITER_DECAY),
        "--gpu_ids", GPU_IDS,
        "--input_nc", "3",
        "--output_nc", "3",
        # cwgan's example script's other choices (per scripts/train_pix2pix.sh):
        "--init_type", "kaiming",
        "--norm", "none",
    ]
    print(f"\n=== TRAIN fold {fold_idx} ===\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=CWGAN_DIR)


def run_test(fold_idx, list_path, phase):
    """Run inference. `phase` becomes part of results dir (val_latest / test_latest)."""
    name = f"clarigan_cwgan_fold_{fold_idx}"
    cmd = [
        sys.executable, "test.py",
        "--dataroot", list_path,
        "--name", name,
        "--model", "pix2pix",
        "--dataset_mode", "bbdm_aligned",
        "--direction", "AtoB",
        "--netG", NETG,
        "--ngf", str(NGF),
        "--load_size", str(LOAD_SIZE),
        "--crop_size", str(CROP_SIZE),
        "--gpu_ids", GPU_IDS,
        "--input_nc", "3",
        "--output_nc", "3",
        "--norm", "none",
        "--phase", phase,
        "--num_test", "10000",
    ]
    print(f"\n=== {phase.upper()} fold {fold_idx} ===\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=CWGAN_DIR)


def main():
    print(f"computing {K}-fold stratified splits from {DATASET_TRAIN_DIR}")
    splits = stratified_kfold_85_5_10(DATASET_TRAIN_DIR, k=K, seed=SEED)

    os.makedirs(FOLD_LISTS_DIR, exist_ok=True)
    print(f"writing fold list files to {FOLD_LISTS_DIR}")

    # Same loop structure as BBDM/k-fold_validation.py:
    #   fold_num = fold['fold']
    #   train_set = fold['train']; val_set = fold['val']; test_set = fold['test']
    #
    # GAN convention: no val-based checkpoint selection. val is held out and
    # per-epoch val L1 is just LOGGED during training (val_log.txt in
    # checkpoints dir) so we can inspect overfitting curves. Final model =
    # last epoch.
    for fold in splits:
        fold_num = fold['fold']
        train_set = fold['train']
        val_set = fold['val']
        test_set = fold['test']

        train_list = write_list_file(train_set, fold_num, "train", FOLD_LISTS_DIR)
        val_list = write_list_file(val_set, fold_num, "val", FOLD_LISTS_DIR)
        test_list = write_list_file(test_set, fold_num, "test", FOLD_LISTS_DIR)

        run_train(fold_num, train_list, val_list)
        run_test(fold_num, test_list, phase="test")


if __name__ == "__main__":
    main()
