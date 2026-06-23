"""K-fold orchestrator for pix2pix baseline.

Uses the SAME stratified k-fold splits as BBDM/k-fold_validation.py (imported
directly) AND the SAME augmentation pipeline (BBDM's CustomAlignedDataset),
then drives the upstream pix2pix train.py / test.py via subprocess.

Writes per-fold list files of (R1_path, R3_path) pairs and runs the GAN
training with:
    --dataset_mode bbdm_aligned --dataroot <list_file>

Custom dataset class:
    baselines/pytorch-CycleGAN-and-pix2pix/data/bbdm_aligned_dataset.py
"""

import os
import sys
import subprocess
import importlib.util


# -------- paths --------
BBDM_DIR = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM"
PIX2PIX_DIR = r"C:\Users\ammic\Desktop\ClariGAN-DL\baselines\pytorch-CycleGAN-and-pix2pix"
DATASET_TRAIN_DIR = r"C:\Users\ammic\Desktop\BBDM-kfold\train"  # has A/ and B/ subdirs
FOLD_LISTS_DIR = os.path.join(PIX2PIX_DIR, "fold_lists")

# -------- training hyperparameters --------
K = 10
SEED = 42
N_EPOCHS = 25          # initial-lr epochs
N_EPOCHS_DECAY = 25    # linearly-decay epochs (so total = 50, matching BBDM)
LOAD_SIZE = 256        # no resize (data already 256x256)
CROP_SIZE = 256
BATCH_SIZE = 1         # canonical pix2pix
GPU_IDS = "0"
NETG = "unet_256"


def load_bbdm_kfold_fn():
    """Import stratified_kfold_85_5_10 from BBDM/k-fold_validation.py without
    triggering its __main__ block."""
    sys.path.insert(0, BBDM_DIR)
    spec = importlib.util.spec_from_file_location(
        "bbdm_kfold", os.path.join(BBDM_DIR, "k-fold_validation.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stratified_kfold_85_5_10


def write_list_file(pairs, fold_num, phase, lists_dir):
    path = os.path.join(lists_dir, f"fold_{fold_num}_{phase}.txt")
    with open(path, "w") as f:
        for r1, r3 in pairs:
            f.write(f"{r1},{r3}\n")
    print(f"  wrote {path}  ({len(pairs)} pairs)")
    return path


def run_train(fold_idx, train_list, val_list):
    """Train pix2pix on the train list. val_list, if non-empty, is passed via
    --val_dataroot so train.py logs per-epoch val L1 (no checkpoint selection)."""
    name = f"clarigan_pix2pix_fold_{fold_idx}"
    cmd = [
        sys.executable, "train.py",
        "--dataroot", train_list,
        "--val_dataroot", val_list,
        "--name", name,
        "--model", "pix2pix",
        "--dataset_mode", "bbdm_aligned",
        "--direction", "AtoB",
        "--netG", NETG,
        "--load_size", str(LOAD_SIZE),
        "--crop_size", str(CROP_SIZE),
        "--batch_size", str(BATCH_SIZE),
        "--n_epochs", str(N_EPOCHS),
        "--n_epochs_decay", str(N_EPOCHS_DECAY),
        "--gpu_ids", GPU_IDS,
        "--input_nc", "3",
        "--output_nc", "3",
    ]
    print(f"\n=== TRAIN fold {fold_idx} ===\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PIX2PIX_DIR)


def run_test(fold_idx, list_path, phase):
    """Run inference. `phase` becomes part of results dir (val_latest / test_latest)
    so the val and test outputs don't clobber each other."""
    name = f"clarigan_pix2pix_fold_{fold_idx}"
    cmd = [
        sys.executable, "test.py",
        "--dataroot", list_path,
        "--name", name,
        "--model", "pix2pix",
        "--dataset_mode", "bbdm_aligned",
        "--direction", "AtoB",
        "--netG", NETG,
        "--load_size", str(LOAD_SIZE),
        "--crop_size", str(CROP_SIZE),
        "--gpu_ids", GPU_IDS,
        "--input_nc", "3",
        "--output_nc", "3",
        "--phase", phase,
        # test.py defaults: serial_batches True, no_flip True, num_test 50.
        # Bump num_test to cover all images in any fold (~76 test, ~38 val).
        "--num_test", "10000",
    ]
    print(f"\n=== {phase.upper()} fold {fold_idx} ===\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PIX2PIX_DIR)


def main():
    stratified_kfold_85_5_10 = load_bbdm_kfold_fn()
    print(f"computing {K}-fold stratified splits from {DATASET_TRAIN_DIR}")
    splits = stratified_kfold_85_5_10(DATASET_TRAIN_DIR, k=K, seed=SEED)

    os.makedirs(FOLD_LISTS_DIR, exist_ok=True)
    print(f"writing fold list files to {FOLD_LISTS_DIR}")

    # Same loop structure as BBDM/k-fold_validation.py:
    #   fold_num = fold['fold']
    #   train_set = fold['train']; val_set = fold['val']; test_set = fold['test']
    #
    # GAN convention: no val-based checkpoint selection (adversarial training
    # has no clean monotone val signal). Instead, val is held out and per-epoch
    # val L1 is just LOGGED during training (val_log.txt in checkpoints dir)
    # so we can inspect overfitting curves. Final model = last epoch.
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
