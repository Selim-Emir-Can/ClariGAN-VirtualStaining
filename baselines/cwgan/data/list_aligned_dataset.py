"""List-based aligned dataset.

Reads paired (A, B) image paths from a list file, where each line is:
    /path/to/A.png,/path/to/B.png

Unlike the standard aligned dataset, this does NOT require concatenated A|B
images on disk. Lets us reuse separate A/ and B/ folders (e.g. the BBDM
ClariGAN-stratified data layout) without duplicating data.

Use:
    --dataset_mode list_aligned --dataroot /path/to/fold_0_train.txt
"""

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image


class ListAlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # opt.dataroot is the path to the list file (.txt) for this phase.
        # For pix2pix train/test cycles, you can also point each phase at a
        # different list file via --dataroot directly.
        list_path = opt.dataroot
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"list file not found: {list_path}")
        self.pairs = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split(",")
                self.pairs.append((a.strip(), b.strip()))
        if opt.max_dataset_size is not None and opt.max_dataset_size != float("inf"):
            self.pairs = self.pairs[: int(opt.max_dataset_size)]

        assert self.opt.load_size >= self.opt.crop_size
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

    def __getitem__(self, index):
        A_path, B_path = self.pairs[index]
        A = Image.open(A_path).convert("RGB")
        B = Image.open(B_path).convert("RGB")

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return len(self.pairs)
