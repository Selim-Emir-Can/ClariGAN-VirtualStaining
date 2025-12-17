# import os
# import shutil
# import numpy as np
# import csv
# from PIL import Image
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from torchvision.models.inception import inception_v3
# import torch
# import torchvision.transforms as T
# from scipy.linalg import sqrtm

# def load_image(path, size=(299, 299)):
#     try:
#         img = Image.open(path).convert('RGB').resize(size, Image.BICUBIC)
#         return img
#     except Exception as e:
#         print(f"⚠️ Error loading {path}: {e}")
#         return None

# def compute_psnr(img1, img2):
#     return psnr(np.array(img1), np.array(img2), data_range=255)

# def is_image_file(fname):
#     return fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))

# def extract_number_from_output(filename):
#     parts = filename.split('_')
#     if len(parts) >= 2 and parts[1].split('.')[0].isdigit():
#         return parts[1].split('.')[0]
#     return None

# def find_matching_visualization(vis_folder, number):
#     for fname in os.listdir(vis_folder):
#         if fname.startswith(f"titled_plot_{number}") and is_image_file(fname):
#             return os.path.join(vis_folder, fname)
#     return None

# def select_best_images(parent_dir, output_dir, psnr_thresh=None):
#     """
#     If psnr_thresh is not None, samples whose best candidate PSNR < psnr_thresh are skipped.
#     """
#     gt_dir = os.path.join(parent_dir, 'ground_truth')
#     candidate_root = os.path.join(parent_dir, '200')
#     vis_root = os.path.join(parent_dir, 'titled_plots')
#     os.makedirs(output_dir, exist_ok=True)

#     gt_files = sorted([f for f in os.listdir(gt_dir) if is_image_file(f)])
#     candidate_folders = sorted([f for f in os.listdir(candidate_root) if os.path.isdir(os.path.join(candidate_root, f))])

#     if len(gt_files) != len(candidate_folders):
#         print(f"❗ Mismatch: {len(gt_files)} GT images, {len(candidate_folders)} candidate folders.")
#         return None

#     metrics = []
#     skipped_count = 0

#     for gt_file, folder_name in zip(gt_files, candidate_folders):
#         gt_path = os.path.join(gt_dir, gt_file)
#         folder_path = os.path.join(candidate_root, folder_name)
#         vis_folder = os.path.join(vis_root, folder_name)

#         gt_img = load_image(gt_path)
#         if gt_img is None:
#             print(f"⚠️ Could not load GT image: {gt_file}")
#             continue

#         best_score = -np.inf
#         best_candidate_fname = None
#         best_candidate_number = None

#         # iterate candidates
#         for fname in sorted(os.listdir(folder_path)):
#             if not is_image_file(fname) or not fname.startswith("output_"):
#                 continue

#             number = extract_number_from_output(fname)
#             if number is None:
#                 continue

#             cand_path = os.path.join(folder_path, fname)
#             cand_img = load_image(cand_path)
#             if cand_img is None:
#                 continue

#             psnr_score = compute_psnr(cand_img, gt_img)
#             if psnr_score > best_score:
#                 best_score = psnr_score
#                 best_candidate_fname = fname
#                 best_candidate_number = number

#         # if no valid candidates found, skip
#         if best_candidate_number is None:
#             print(f"⚠️ No valid candidates in {folder_path}; skipping.")
#             skipped_count += 1
#             continue

#         # threshold check
#         if (psnr_thresh is not None) and ((best_score < psnr_thresh) or (best_score >= 30)):
#             print(f"⏭️  Skipping {folder_name}: best PSNR {best_score:.2f} < threshold {psnr_thresh:.2f}")
#             skipped_count += 1
#             continue

#         # copy matching visualization
#         vis_path = find_matching_visualization(vis_folder, best_candidate_number)
#         if not vis_path:
#             print(f"⚠️ Visualization not found for {folder_name} candidate {best_candidate_number}")
#             skipped_count += 1
#             continue

#         out_path = os.path.join(output_dir, gt_file)
#         shutil.copy(vis_path, out_path)

#         metrics.append((gt_file, folder_name, best_candidate_fname, best_score))
#         print(f"✅ {folder_name} → {gt_file} (best: {best_candidate_fname}, PSNR: {best_score:.2f})")

#     # Save metrics to CSV
#     csv_path = os.path.join(output_dir, "metrics.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["GT Image", "Folder", "Best Candidate", "PSNR"])
#         writer.writerows(metrics)

#     kept = len(metrics)
#     print(f"\n📦 Kept: {kept} | ⏭️ Skipped: {skipped_count}")
#     if metrics:
#         psnrs = [row[3] for row in metrics]
#         avg_psnr = sum(psnrs) / len(psnrs)
#         print(f"📊 Fold average PSNR (kept only): {avg_psnr:.2f}")
#         return avg_psnr
#     else:
#         return None

# # Run this script to aggregate across folds
# import glob

# fold_base = r"C:\Users\ammic\Desktop\ClariGAN-DL\BrianHE_k-fold_results\Brian_v1\complete_folds"
# output_base = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\k-fold-automatic_selection\output_best_images_HE"
# PSNR_THRESHOLD = 15.0  # set what you want

# fold_dirs = sorted([d for d in os.listdir(fold_base) if os.path.isdir(os.path.join(fold_base, d))])
# fold_avg_psnrs = []

# for fold in fold_dirs:
#     print(f"\n🚀 Processing {fold}...")
#     parent_folder = os.path.join(fold_base, fold)
#     output_folder = os.path.join(output_base, fold)
#     avg_psnr = select_best_images(parent_folder, output_folder, psnr_thresh=PSNR_THRESHOLD)
#     if avg_psnr is not None:
#         fold_avg_psnrs.append((fold, avg_psnr))

# # Save overall summary
# summary_path = os.path.join(output_base, "summary.csv")
# with open(summary_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Fold", "Average PSNR"])
#     writer.writerows(fold_avg_psnrs)

# if fold_avg_psnrs:
#     overall_avg = sum(f[1] for f in fold_avg_psnrs) / len(fold_avg_psnrs)
#     print(f"\n✅ Overall average PSNR across folds: {overall_avg:.2f}")
# else:
#     print("\n⚠️ No valid folds processed.")
import os
import shutil
import numpy as np
import csv
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim  # NEW


import torch, kornia
import torch, torch.nn.functional as F
import math

def ciede2000_from_lab(Lab1: torch.Tensor, Lab2: torch.Tensor,
                       kL: float = 1.0, kC: float = 1.0, kH: float = 1.0,
                       eps: float = 1e-12) -> torch.Tensor:
    """
    Lab inputs: [N, 3, H, W] (L* in [0,100], a*, b* ~ [-128,127])
    Returns per-pixel ΔE00 map [N, H, W].
    """
    assert Lab1.shape == Lab2.shape and Lab1.shape[1] == 3, "Expect [N,3,H,W]"
    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    # Chroma
    C1 = torch.sqrt(a1*a1 + b1*b1 + eps)
    C2 = torch.sqrt(a2*a2 + b2*b2 + eps)
    Cbar = (C1 + C2) * 0.5

    # Compensation factor
    c25 = Lab1.new_tensor(25.0)
    c25_7 = c25.pow(7)
    G = 0.5 * (1.0 - torch.sqrt((Cbar.pow(7) / (Cbar.pow(7) + c25_7)).clamp_min(0)))

    # Adjusted a*
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    # Adjusted chroma
    C1p = torch.sqrt(a1p*a1p + b1*b1 + eps)
    C2p = torch.sqrt(a2p*a2p + b2*b2 + eps)

    # Adjusted hue (radians in [0, 2π))
    h1p = torch.atan2(b1, a1p); h1p = torch.where(h1p < 0, h1p + 2*math.pi, h1p)
    h2p = torch.atan2(b2, a2p); h2p = torch.where(h2p < 0, h2p + 2*math.pi, h2p)

    # Differences
    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(dhp >  math.pi, dhp - 2*math.pi, dhp)
    dhp = torch.where(dhp < -math.pi, dhp + 2*math.pi, dhp)
    # If either chroma is ~0, hue diff = 0
    dhp = torch.where((C1p*C2p) < eps, Lab1.new_zeros(dhp.shape), dhp)

    dHp = 2.0 * torch.sqrt(C1p*C2p + eps) * torch.sin(dhp * 0.5)

    # Averages
    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    habs = torch.abs(h1p - h2p)
    hbarp = (h1p + h2p) * 0.5
    hbarp = torch.where((C1p*C2p) < eps, h1p + h2p, hbarp)
    hbarp = torch.where((C1p*C2p) >= eps, torch.where(habs > math.pi,
                        hbarp + math.pi * torch.where((h1p + h2p) < 2*math.pi, 1.0, -1.0), hbarp),
                        hbarp)
    hbarp = hbarp % (2*math.pi)

    # T term
    T = (1
         - 0.17*torch.cos(hbarp - math.radians(30))
         + 0.24*torch.cos(2*hbarp)
         + 0.32*torch.cos(3*hbarp + math.radians(6))
         - 0.20*torch.cos(4*hbarp - math.radians(63)))

    # Rotation and weighting terms
    habp_deg = hbarp * (180.0 / math.pi)
    dTheta = 30.0 * torch.exp(-((habp_deg - 275.0)/25.0)**2)
    RC = 2.0 * torch.sqrt((Cbarp.pow(7) / (Cbarp.pow(7) + c25_7)).clamp_min(0))
    SL = 1.0 + (0.015 * (Lbarp - 50.0).pow(2)) / torch.sqrt(20.0 + (Lbarp - 50.0).pow(2) + eps)
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T
    RT = -torch.sin(2.0 * torch.deg2rad(dTheta)) * RC

    # Compose
    dL_ = dLp / (kL * SL + eps)
    dC_ = dCp / (kC * SC + eps)
    dH_ = dHp / (kH * SH + eps)

    dE2 = dL_*dL_ + dC_*dC_ + dH_*dH_ + RT * dC_ * dH_
    return torch.sqrt(dE2.clamp_min(0.0))

def ciede2000_from_rgb(rgb1: torch.Tensor, rgb2: torch.Tensor, **kw) -> torch.Tensor:
    """
    RGB in [0,1], NCHW. Converts to Lab and calls ΔE00.
    """
    lab1 = kornia.color.rgb_to_lab(rgb1)
    lab2 = kornia.color.rgb_to_lab(rgb2)
    return ciede2000_from_lab(lab1, lab2, **kw)

def load_image(path, size=(299, 299)):
    try:
        img = Image.open(path).convert('RGB').resize(size, Image.BICUBIC)
        return img
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        return None

def compute_psnr(img1, img2):
    # img1, img2: PIL images, uint8 [0,255]
    return psnr(np.array(img1), np.array(img2), data_range=255)

def compute_ssim(img1, img2):
    # Robust to skimage versions (channel_axis vs multichannel)
    a1, a2 = np.array(img1), np.array(img2)
    try:
        return ssim(a1, a2, data_range=255, channel_axis=-1)
    except TypeError:
        return ssim(a1, a2, data_range=255, multichannel=True)

import torch
import kornia

# def compute_cidede2000_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
#     """
#     Computes differentiable CIEDE2000 loss between two RGB images.
#     Uses ΔE00 (per-pixel perceptual difference) as the error metric.

#     Args:
#         pred (torch.Tensor): Predicted image, shape [N, 3, H, W], RGB in [0,1]
#         target (torch.Tensor): Ground-truth image, shape [N, 3, H, W], RGB in [0,1]
#         reduction (str): "mean" | "sum" | "none"
#                          - "mean": returns scalar mean ΔE00
#                          - "sum": returns total ΔE00 over all pixels
#                          - "none": returns full per-pixel ΔE00 map

#     Returns:
#         torch.Tensor: Scalar loss or per-pixel map depending on `reduction`.
#     """
#     # Sanity checks
#     assert pred.shape == target.shape, f"Shape mismatch: pred={pred.shape}, target={target.shape}"
#     assert pred.shape[1] == 3, "Images must be RGB: shape [N,3,H,W]"

#     # Convert to Lab color space
#     lab_pred = kornia.color.rgb_to_lab(pred)
#     lab_target = kornia.color.rgb_to_lab(target)

#     # Compute ΔE00 per-pixel map
#     delta_e = ciede2000_from_lab(lab_pred, lab_target)

#     # Reduce if needed
#     if reduction == "mean":
#         return delta_e.mean()
#     elif reduction == "sum":
#         return delta_e.sum()
#     elif reduction == "none":
#         return delta_e
#     else:
#         raise ValueError(f"Invalid reduction '{reduction}', choose from ['mean', 'sum', 'none']")

import torch
import numpy as np
from PIL import Image
import kornia
from typing import Union

def compute_cidede2000_loss(pred: Union[Image.Image, np.ndarray], 
                            target: Union[Image.Image, np.ndarray],
                            reduction: str = "mean") -> float:
    """
    Computes CIEDE2000 ΔE00 loss between two RGB images (PIL or NumPy).
    Lower values = better perceptual similarity.

    Args:
        pred (PIL.Image | np.ndarray): Predicted image (RGB)
        target (PIL.Image | np.ndarray): Ground-truth image (RGB)
        reduction (str): "mean" | "sum" | "none"
                         - "mean": returns scalar mean ΔE00 (default)
                         - "sum": returns total ΔE00 over all pixels
                         - "none": returns the full ΔE00 map (as np.ndarray)

    Returns:
        float | np.ndarray: Scalar loss or per-pixel ΔE00 map.
    """
    # Convert PIL → NumPy → Torch tensor in [0,1]
    if isinstance(pred, Image.Image):
        pred = np.array(pred.convert("RGB"), dtype=np.float32) / 255.0
    if isinstance(target, Image.Image):
        target = np.array(target.convert("RGB"), dtype=np.float32) / 255.0

    # Sanity check shape
    assert pred.shape == target.shape, f"Shape mismatch: pred={pred.shape}, target={target.shape}"
    assert pred.ndim == 3 and pred.shape[2] == 3, "Images must be RGB, shape [H,W,3]"

    # Convert [H,W,3] → [1,3,H,W] torch format
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
    target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)

    # Convert to Lab color space
    lab_pred = kornia.color.rgb_to_lab(pred_t)
    lab_target = kornia.color.rgb_to_lab(target_t)

    # Compute ΔE00 per-pixel difference
    delta_e = ciede2000_from_lab(lab_pred, lab_target)

    # Reduce result
    if reduction == "mean":
        return delta_e.mean().item()
    elif reduction == "sum":
        return delta_e.sum().item()
    elif reduction == "none":
        return delta_e.squeeze(0).cpu().numpy()
    else:
        raise ValueError(f"Invalid reduction '{reduction}', choose from ['mean', 'sum', 'none']")


def is_image_file(fname):
    return fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))

def extract_number_from_output(filename):
    parts = filename.split('_')
    if len(parts) >= 2 and parts[1].split('.')[0].isdigit():
        return parts[1].split('.')[0]
    return None

def find_matching_visualization(vis_folder, number):
    for fname in os.listdir(vis_folder):
        if fname.startswith(f"titled_plot_{number}") and is_image_file(fname):
            return os.path.join(vis_folder, fname)
    return None

def _pick_joint_best(candidates):
    """
    candidates: list of dicts with keys ['fname','number','psnr','ssim'].
    Strategy: rank by PSNR desc and SSIM desc; minimize rank_sum = r_psnr + r_ssim.
    Tie-break: higher PSNR, then higher SSIM.
    """
    psnr_sorted = sorted(candidates, key=lambda c: c['psnr'], reverse=True)
    ssim_sorted = sorted(candidates, key=lambda c: c['ssim'], reverse=True)
    ciedede_sorted = sorted(candidates, key=lambda c: c['ciedede'])

    psnr_rank = {c['fname']: i for i, c in enumerate(psnr_sorted)}
    ssim_rank = {c['fname']: i for i, c in enumerate(ssim_sorted)}
    ciedede_rank = {c['fname']: i for i, c in enumerate(ciedede_sorted)}

    for c in candidates:
        c['rank_sum'] = ciedede_rank[c['fname']] # psnr_rank[c['fname']] + ssim_rank[c['fname']]

    # best = min(
    #     candidates,
    #     key=lambda c: (c['rank_sum'], -c['psnr'], -c['ssim'])
    # )
    best = min(
        candidates,
        key=lambda c: (c['rank_sum'])
    )
    return best

def select_best_images(parent_dir, output_dir, psnr_thresh=None):
    """
    If psnr_thresh is not None, samples whose selected candidate PSNR < psnr_thresh are skipped.
    (Thresholding still uses PSNR; selection uses joint PSNR+SSIM rank.)
    """
    gt_dir = os.path.join(parent_dir, 'ground_truth')
    candidate_root = os.path.join(parent_dir, '200')
    vis_root = os.path.join(parent_dir, 'titled_plots')
    os.makedirs(output_dir, exist_ok=True)

    gt_files = sorted([f for f in os.listdir(gt_dir) if is_image_file(f)])
    candidate_folders = sorted([f for f in os.listdir(candidate_root) if os.path.isdir(os.path.join(candidate_root, f))])

    if len(gt_files) != len(candidate_folders):
        print(f"❗ Mismatch: {len(gt_files)} GT images, {len(candidate_folders)} candidate folders.")
        return None

    metrics = []
    skipped_count = 0

    for gt_file, folder_name in zip(gt_files, candidate_folders):
        gt_path = os.path.join(gt_dir, gt_file)
        folder_path = os.path.join(candidate_root, folder_name)
        vis_folder = os.path.join(vis_root, folder_name)

        gt_img = load_image(gt_path)
        if gt_img is None:
            print(f"⚠️ Could not load GT image: {gt_file}")
            continue

        # Gather metrics for all candidates in this folder
        cand_stats = []
        for fname in sorted(os.listdir(folder_path)):
            if not is_image_file(fname) or not fname.startswith("output_"):
                continue

            number = extract_number_from_output(fname)
            if number is None:
                continue

            cand_path = os.path.join(folder_path, fname)
            cand_img = load_image(cand_path)
            if cand_img is None:
                continue

            psnr_score = compute_psnr(cand_img, gt_img)
            ssim_score = compute_ssim(cand_img, gt_img)
            ciedede_score = compute_cidede2000_loss(cand_img, gt_img, "mean")

            cand_stats.append({
                'fname': fname,
                'number': number,
                'psnr': psnr_score,
                'ssim': ssim_score,
                'ciedede': ciedede_score
            })

        if not cand_stats:
            print(f"⚠️ No valid candidates in {folder_path}; skipping.")
            skipped_count += 1
            continue

        # Joint selection by PSNR+SSIM rank
        best = _pick_joint_best(cand_stats)
        best_psnr = best['psnr']
        best_ssim = best['ssim']
        best_candidate_fname = best['fname']
        best_candidate_number = best['number']

        # Threshold (kept as PSNR-based, same logic as before)
        if False: #  (psnr_thresh is not None) and ((best_psnr < psnr_thresh) or (best_psnr >= 30)):
            print(f"⏭️  Skipping {folder_name}: best PSNR {best_psnr:.2f} < threshold {psnr_thresh:.2f}")
            skipped_count += 1
            continue

        # Copy matching visualization
        vis_path = find_matching_visualization(vis_folder, best_candidate_number)
        if not vis_path:
            print(f"⚠️ Visualization not found for {folder_name} candidate {best_candidate_number}")
            skipped_count += 1
            continue

        out_path = os.path.join(output_dir, gt_file)
        shutil.copy(vis_path, out_path)

        metrics.append((gt_file, folder_name, best_candidate_fname, best_psnr, best_ssim, best['rank_sum']))
        print(f"✅ {folder_name} → {gt_file} "
              f"(best: {best_candidate_fname}, PSNR: {best_psnr:.2f}, SSIM: {best_ssim:.4f}, rank_sum: {best['rank_sum']})")

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GT Image", "Folder", "Best Candidate", "PSNR", "SSIM", "RankSum"])
        writer.writerows(metrics)

    kept = len(metrics)
    print(f"\n📦 Kept: {kept} | ⏭️ Skipped: {skipped_count}")
    if metrics:
        psnrs = [row[3] for row in metrics]
        ssims = [row[4] for row in metrics]
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_ssim = sum(ssims) / len(ssims)
        print(f"📊 Fold averages (kept only): PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        return avg_psnr, avg_ssim
    else:
        return None

# ----------------------------
# Run this script to aggregate across folds
# ----------------------------
import glob

fold_base = r"C:\Users\ammic\Desktop\ClariGAN-DL\BrianHE_k-fold_results"
output_base = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\k-fold-automatic_selection\output_best_images_HEnew"
PSNR_THRESHOLD = 15.0  # set what you want

fold_dirs = sorted([d for d in os.listdir(fold_base) if os.path.isdir(os.path.join(fold_base, d))])
fold_avg_stats = []  # (fold, avg_psnr, avg_ssim)

for fold in fold_dirs:
    print(f"\n🚀 Processing {fold}...")
    parent_folder = os.path.join(fold_base, fold)
    output_folder = os.path.join(output_base, fold)
    result = select_best_images(parent_folder, output_folder, psnr_thresh=PSNR_THRESHOLD)
    if result is not None:
        avg_psnr, avg_ssim = result
        fold_avg_stats.append((fold, avg_psnr, avg_ssim))

# Save overall summary
summary_path = os.path.join(output_base, "summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Average PSNR", "Average SSIM"])
    writer.writerows(fold_avg_stats)

if fold_avg_stats:
    overall_avg_psnr = sum(f[1] for f in fold_avg_stats) / len(fold_avg_stats)
    overall_avg_ssim = sum(f[2] for f in fold_avg_stats) / len(fold_avg_stats)
    print(f"\n✅ Overall averages across folds: PSNR={overall_avg_psnr:.2f}, SSIM={overall_avg_ssim:.4f}")
else:
    print("\n⚠️ No valid folds processed.")
