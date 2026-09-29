"""Specimen class labels for specimen-conditioned L-BBDM.

Label = index of the physical specimen (A..K, 11 classes); index 11 is the "null" label used
for classifier-free label dropout during training and for unconditional sampling (e.g. an
unseen specimen). Filenames look like R1-A_row0_col2_5x5[.png]; Z is D's second half and
Hpart1 is H's second half, so they map to D and H (same rule as the dataset's specimen column).
"""
import os, re
import torch

SPECIMENS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
NULL_LABEL = len(SPECIMENS)            # 11
NUM_CLASSES = len(SPECIMENS) + 1       # 12 = 11 specimens + null
_ALIAS = {"Z": "D", "Hpart1": "H"}
_PAT = re.compile(r"^(?:R\d+-)?([A-Za-z0-9]+?)_row\d+_col\d+")


def specimen_of(name):
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    m = _PAT.match(stem)
    if not m:
        raise ValueError(f"cannot parse specimen from tile name {name!r}")
    piece = m.group(1)
    return _ALIAS.get(piece, piece)


def names_to_labels(names, device=None):
    return torch.tensor([SPECIMENS.index(specimen_of(n)) for n in names], dtype=torch.long, device=device)
