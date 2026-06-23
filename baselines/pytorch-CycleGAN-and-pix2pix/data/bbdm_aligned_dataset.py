"""BBDM-style aligned dataset for pix2pix.

Wraps BBDM's CustomAlignedDataset (from BBDM/datasets/custom.py) so the GAN
baseline uses the IDENTICAL augmentation pipeline as BBDM training:
  - ConsistentTransform (same random params applied to both A and B):
      horizontal flip, vertical flip, rotation (180 deg), translation,
      random resized crop, resize
  - 8x oversampling per epoch during training, no augmentation at val/test
  - Normalize to [-1, 1] (matches pix2pix default)

Reads the same list file as list_aligned_dataset.py:
    /path/to/A.png,/path/to/B.png   (one pair per line)

Use:
    --dataset_mode bbdm_aligned --dataroot /path/to/fold_N_train.txt
"""

import os
import sys
import types

from data.base_dataset import BaseDataset

# Add BBDM to sys.path so we can import its dataset module + its `Register`/
# `datasets.base` dependencies. Resolved relative to this file:
#   .../baselines/pytorch-CycleGAN-and-pix2pix/data/bbdm_aligned_dataset.py
#   .../BBDM/
BBDM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "BBDM")
)
if BBDM_DIR not in sys.path:
    sys.path.insert(0, BBDM_DIR)

from datasets.custom import CustomAlignedDataset  # noqa: E402


class BbdmAlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # opt.dataroot is the path to the list file for this phase
        list_path = opt.dataroot
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"list file not found: {list_path}")
        pairs = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split(",")
                pairs.append((a.strip(), b.strip()))

        # Construct the namespace BBDM's CustomAlignedDataset expects.
        # dataset_path is unused when split_set is provided.
        cfg = types.SimpleNamespace(
            image_size=opt.crop_size,
            dataset_path="",
            flip=True,         # only consulted when stage == 'train'
            to_normal=True,    # normalize to [-1, 1] (matches pix2pix)
        )

        # stage controls BBDM's augmentation + 8x oversampling.
        # train -> augmented 8x; val/test -> no augmentation, length = len(pairs).
        stage = "train" if opt.isTrain else opt.phase
        self._bbdm = CustomAlignedDataset(cfg, stage=stage, split_set=pairs)

    def __len__(self):
        return len(self._bbdm)

    def __getitem__(self, index):
        # BBDM returns ((img_ori, stem_ori), (img_cond, stem_cond))
        #   ori = R3 = target  (B in pix2pix AtoB)
        #   cond = R1 = input  (A in pix2pix AtoB)
        (img_ori, stem_ori), (img_cond, stem_cond) = self._bbdm[index]

        if self.opt.direction == "BtoA":
            A, B = img_ori, img_cond
            A_path, B_path = stem_ori, stem_cond
        else:
            A, B = img_cond, img_ori
            A_path, B_path = stem_cond, stem_ori
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}
