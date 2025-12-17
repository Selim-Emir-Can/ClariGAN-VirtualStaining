# import csv
# import sys
# import os
# import shutil
# from datetime import datetime
# from pathlib import Path
# from typing import Optional, List, Tuple, Dict

# import numpy as np
# import torch
# import kornia
# from PIL import Image, ImageDraw, ImageFont, ImageTk
# import tkinter as tk

# # ==============================
# # SET YOUR DIRECTORIES HERE
# # ==============================
# # INPUT_DIR has three subdirs: titled_plots, 200, ground_truth
# INPUT_DIR = r"C:\Users\ammic\Desktop\ClariGAN-DL\BrianHE_k-fold_results\HE400\fold_0"
# OUTPUT_DIR = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\k-fold-manual_selection"

# IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
# CSV_COLUMNS = [
#     "timestamp",
#     "folder_path",
#     "selected_index",
#     "selected_filename",
#     "source_path",
#     "saved_as",
#     "ciede2000",
#     "status",  # <— NEW
# ]


# # -------------------------------
# # Utility: file/dir helpers
# # -------------------------------
# def is_image_file(p: Path) -> bool:
#     return p.suffix.lower() in IMAGE_EXTS

# def find_image_files(folder: Path) -> List[Path]:
#     files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
#     files.sort()
#     return files

# def extract_number_from_titled_plot(filename: str) -> Optional[str]:
#     # expects "titled_plot_<number>.*" (robust if extra underscores exist)
#     name = Path(filename).stem
    
#     part = name.split("_")[2]

#     if part.isdigit():
#         return part
#     return None

# def find_candidate_image(candidate_dir: Path, number: str) -> Optional[Path]:
#     # find file like "output_<number>.<ext>"
#     if not candidate_dir.is_dir():
#         return None
#     for p in sorted(candidate_dir.iterdir()):
#         if p.is_file() and is_image_file(p) and p.stem.startswith(f"output_{number}"):
#             return p
#     return None

# def find_ground_truth_for_folder(gt_dir: Path, folder_name: str) -> Optional[Path]:
#     """
#     Heuristics:
#       1) exact stem match
#       2) stem contained in folder_name or vice versa
#       3) if only one image in gt_dir, use it
#     """
#     if not gt_dir.is_dir():
#         return None
#     images = [p for p in gt_dir.iterdir() if p.is_file() and is_image_file(p)]
#     if not images:
#         return None
#     for p in images:
#         if p.stem == folder_name:
#             return p
#     for p in images:
#         if (p.stem in folder_name) or (folder_name in p.stem):
#             return p
#     if len(images) == 1:
#         return images[0]
#     return None

# # -------------------------------
# # CSV helpers (resume + schema)
# # -------------------------------
# def ensure_csv_with_schema(csv_path: Path, columns: List[str]) -> None:
#     if not csv_path.exists():
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=columns)
#             writer.writeheader()
#         return

#     with open(csv_path, "r", encoding="utf-8", newline="") as f:
#         reader = csv.reader(f)
#         rows = list(reader)

#     if not rows:
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=columns)
#             writer.writeheader()
#         return

#     old_header = rows[0]
#     if old_header == columns:
#         return

#     old_idx = {name: i for i, name in enumerate(old_header)}
#     dict_rows = []
#     for r in rows[1:]:
#         d = {}
#         for name in columns:
#             if name in old_idx and old_idx[name] < len(r):
#                 d[name] = r[old_idx[name]]
#             else:
#                 d[name] = ""
#         dict_rows.append(d)

#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=columns)
#         writer.writeheader()
#         writer.writerows(dict_rows)

# def load_processed_folders(csv_path: Path) -> set:
#     processed = set()
#     if csv_path.exists():
#         with open(csv_path, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 processed.add(row["folder_path"])
    
#     return processed

# def append_csv_row(csv_path: Path, data: Dict[str, str]) -> None:
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
#         writer.writerow(data)

# def append_skip_row(csv_path: Path, folder_path: Path) -> None:
#     append_csv_row(csv_path, {
#         "timestamp": datetime.now().isoformat(timespec="seconds"),
#         "folder_path": str(folder_path),
#         "selected_index": "",
#         "selected_filename": "",
#         "source_path": "",
#         "saved_as": "",
#         "ciede2000": "",
#         "status": "skipped",
#     })


# def remove_last_csv_row_for_folder(csv_path: Path, folder_path: Path) -> Optional[Dict[str, str]]:
#     if not csv_path.exists():
#         return None
#     with open(csv_path, "r", encoding="utf-8", newline="") as f:
#         rows = list(csv.DictReader(f))
#     idx = None
#     for i in range(len(rows) - 1, -1, -1):
#         if rows[i]["folder_path"] == str(folder_path):
#             idx = i
#             break
#     if idx is None:
#         return None
#     removed = rows.pop(idx)
#     with open(csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
#         writer.writeheader()
#         writer.writerows(rows)
#     return removed

# # -------------------------------
# # Metrics: CIEDE2000 (ΔE00)
# # -------------------------------
# def load_rgb_native(path: Path) -> Optional[Image.Image]:
#     try:
#         return Image.open(path).convert("RGB")
#     except Exception as e:
#         print(f"⚠️ Error loading {path}: {e}")
#         return None

# def ciede2000_from_lab(Lab1: torch.Tensor, Lab2: torch.Tensor,
#                        kL: float = 1.0, kC: float = 1.0, kH: float = 1.0,
#                        eps: float = 1e-12) -> torch.Tensor:
#     assert Lab1.shape == Lab2.shape and Lab1.shape[1] == 3, "Expect [N,3,H,W]"
#     L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
#     L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

#     C1 = torch.sqrt(a1*a1 + b1*b1 + eps)
#     C2 = torch.sqrt(a2*a2 + b2*b2 + eps)
#     Cbar = (C1 + C2) * 0.5

#     c25_7 = (Lab1.new_tensor(25.0)).pow(7)
#     G = 0.5 * (1.0 - torch.sqrt((Cbar.pow(7) / (Cbar.pow(7) + c25_7)).clamp_min(0)))

#     a1p = (1.0 + G) * a1
#     a2p = (1.0 + G) * a2

#     C1p = torch.sqrt(a1p*a1p + b1*b1 + eps)
#     C2p = torch.sqrt(a2p*a2p + b2*b2 + eps)

#     h1p = torch.atan2(b1, a1p); h1p = torch.where(h1p < 0, h1p + 2*np.pi, h1p)
#     h2p = torch.atan2(b2, a2p); h2p = torch.where(h2p < 0, h2p + 2*np.pi, h2p)

#     dLp = L2 - L1
#     dCp = C2p - C1p

#     dhp = h2p - h1p
#     dhp = torch.where(dhp >  np.pi, dhp - 2*np.pi, dhp)
#     dhp = torch.where(dhp < -np.pi, dhp + 2*np.pi, dhp)
#     dhp = torch.where((C1p*C2p) < eps, Lab1.new_zeros(dhp.shape), dhp)

#     dHp = 2.0 * torch.sqrt(C1p*C2p + eps) * torch.sin(dhp * 0.5)

#     Lbarp = 0.5 * (L1 + L2)
#     Cbarp = 0.5 * (C1p + C2p)

#     habs = torch.abs(h1p - h2p)
#     hbarp = (h1p + h2p) * 0.5
#     hbarp = torch.where((C1p*C2p) < eps, h1p + h2p, hbarp)
#     hbarp = torch.where((C1p*C2p) >= eps, torch.where(habs > np.pi,
#                         hbarp + np.pi * torch.where((h1p + h2p) < 2*np.pi, 1.0, -1.0), hbarp),
#                         hbarp)
#     hbarp = hbarp % (2*np.pi)

#     T = (1
#          - 0.17*torch.cos(hbarp - np.deg2rad(30))
#          + 0.24*torch.cos(2*hbarp)
#          + 0.32*torch.cos(3*hbarp + np.deg2rad(6))
#          - 0.20*torch.cos(4*hbarp - np.deg2rad(63)))

#     habp_deg = hbarp * (180.0 / np.pi)
#     dTheta = 30.0 * torch.exp(-((habp_deg - 275.0)/25.0)**2)
#     RC = 2.0 * torch.sqrt((Cbarp.pow(7) / (Cbarp.pow(7) + c25_7)).clamp_min(0))
#     SL = 1.0 + (0.015 * (Lbarp - 50.0).pow(2)) / torch.sqrt(20.0 + (Lbarp - 50.0).pow(2) + eps)
#     SC = 1.0 + 0.045 * Cbarp
#     SH = 1.0 + 0.015 * Cbarp * T
#     RT = -torch.sin(2.0 * torch.deg2rad(dTheta)) * RC

#     dL_ = dLp / (kL * SL + eps)
#     dC_ = dCp / (kC * SC + eps)
#     dH_ = dHp / (kH * SH + eps)

#     dE2 = dL_*dL_ + dC_*dC_ + dH_*dH_ + RT * dC_ * dH_
#     return torch.sqrt(dE2.clamp_min(0.0))

# def compute_ciede2000(pred_img: Image.Image, gt_img: Image.Image) -> Optional[float]:
#     """
#     Computes ΔE00 at native resolution. If sizes differ, pred is resized to GT size (bicubic).
#     """
#     try:
#         if pred_img.size != gt_img.size:
#             pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

#         pred = np.asarray(pred_img.convert("RGB"), dtype=np.float32) / 255.0
#         gt   = np.asarray(gt_img.convert("RGB"), dtype=np.float32) / 255.0

#         pred_t = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0)
#         gt_t   = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0)

#         lab_pred = kornia.color.rgb_to_lab(pred_t)
#         lab_gt   = kornia.color.rgb_to_lab(gt_t)
#         delta = ciede2000_from_lab(lab_pred, lab_gt)
#         return float(delta.mean().item())
#     except Exception as e:
#         print(f"⚠️ ΔE00 failed: {e}")
#         return None

# # -------------------------------
# # Build annotated stack per folder
# # -------------------------------
# def build_annotated_stack_and_scores(
#     vis_folder: Path, candidate_folder: Path, gt_path: Path
# ) -> Tuple[Optional[Image.Image], List[Optional[float]], List[Image.Image], List[Path]]:
#     """
#     Returns:
#       stacked_image (PIL, annotated),
#       scores: [ΔE00 for each of 5 images],
#       per_image_annotated: list of 5 PIL images with overlay,
#       vis_paths: the original titled_plot paths in order 0..4
#     """
#     vis_paths = find_image_files(vis_folder)
#     if len(vis_paths) != 5:
#         print(f"[SKIP] {vis_folder} (found {len(vis_paths)} images, expected 5).")
#         return None, [], [], []

#     vis_paths.sort()
#     gt_img = load_rgb_native(gt_path)
#     if gt_img is None:
#         print(f"[WARN] Could not load GT: {gt_path}")
#         return None, [], [], []

#     try:
#         font = ImageFont.truetype("arial.ttf", 24)
#     except:
#         font = ImageFont.load_default()

#     scores: List[Optional[float]] = []
#     annotated_list: List[Image.Image] = []

#     for idx, vp in enumerate(vis_paths):
#         # Find matching candidate by number
#         num = extract_number_from_titled_plot(vp.name)
#         score = None
        
#         if num is not None:
#             cand_path = find_candidate_image(candidate_folder, num)
#             if cand_path:
#                 cand_img = load_rgb_native(cand_path)
#                 if cand_img is not None:
#                     score = compute_ciede2000(cand_img, gt_img)

#         scores.append(score)

#         # Build display image (annotate titled_plot)
#         try:
#             im = Image.open(vp).convert("RGB")
#         except Exception as e:
#             print(f"⚠️ Error opening {vp}: {e}")
#             return None, [], [], []

#         im = im.copy()
#         draw = ImageDraw.Draw(im, "RGBA")

#         # index box (top-left)
#         draw.rectangle([0, 0, 70, 50], fill=(0, 0, 0, 128))
#         draw.text((12, 10), str(idx), fill=(255,255,255,255), font=font)

#         # ΔE00 box (top-right)
#         text = f"ΔE00: {score:.2f}" if (score is not None) else "ΔE00: N/A"
#         bbox = draw.textbbox((0,0), text, font=font)
#         tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#         pad = 8
#         draw.rectangle([im.width - tw - 2*pad, 0, im.width, th + 2*pad], fill=(0,0,0,128))
#         draw.text((im.width - tw - pad, pad), text, fill=(255,255,255,255), font=font)

#         annotated_list.append(im)

#     # Stack vertically
#     total_h = sum(i.height for i in annotated_list)
#     stacked = Image.new("RGB", (annotated_list[0].width, total_h))
#     y = 0
#     for im in annotated_list:
#         stacked.paste(im, (0, y))
#         y += im.height

#     return stacked, scores, annotated_list, vis_paths

# # -------------------------------
# # Main App
# # -------------------------------
# def main():
#     root_dir = Path(INPUT_DIR).expanduser().resolve()
#     output_dir = Path(OUTPUT_DIR).expanduser().resolve()
#     output_dir.mkdir(parents=True, exist_ok=True)

#     csv_path = output_dir / "summary.csv"
#     ensure_csv_with_schema(csv_path, CSV_COLUMNS)
#     processed_folders = load_processed_folders(csv_path)

#     # Validate expected subfolders inside INPUT_DIR
#     titled_root = root_dir / "titled_plots"
#     candidate_root = root_dir / "200"
#     gt_dir = root_dir / "ground_truth"

#     if not (titled_root.is_dir() and candidate_root.is_dir() and gt_dir.is_dir()):
#         print("[ERROR] INPUT_DIR must contain 'titled_plots', '200', and 'ground_truth' subfolders.")
#         print(f"titled_plots: {titled_root}\n200: {candidate_root}\nground_truth: {gt_dir}")
#         sys.exit(1)

#     # Case folders live under titled_plots
#     case_folders = [p for p in sorted(titled_root.iterdir()) if p.is_dir()]
#     if not case_folders:
#         print(f"[WARN] No case subfolders found in {titled_root}")
#         sys.exit(0)

#     # Tk setup
#     root = tk.Tk()
#     root.title("Select (0-4) | s=skip | b=back | q=quit")
#     screen_w = root.winfo_screenwidth()
#     screen_h = root.winfo_screenheight()
#     root.geometry(f"{min(1400, screen_w-60)}x{min(screen_h-60, screen_h-60)}")

#     label = tk.Label(root)
#     label.pack(expand=True, fill="both")

#     selection_var = tk.StringVar(value="")
#     history = []  # [{'folder', 'selected_idx', 'dst_path'}]

#     def on_key(event):
#         ch = event.char.lower()
#         if ch in ['0','1','2','3','4','s','q','b']:
#             selection_var.set(ch)

#     root.bind("<Key>", on_key)

#     def show_and_pick(tp_case_folder: Path):
#         case_name = tp_case_folder.name
#         gt_path = find_ground_truth_for_folder(gt_dir, case_name)
#         if gt_path is None:
#             print(f"[SKIP] Ground-truth not found for case '{case_name}'.")
#             append_skip_row(csv_path, tp_case_folder)   # NEW
#             return "skipped", None

#         candidate_folder = candidate_root / case_name
#         if not candidate_folder.is_dir():
#             print(f"[SKIP] Candidate folder not found: {candidate_folder}")
#             append_skip_row(csv_path, tp_case_folder)   # NEW
#             return "skipped", None

#         stacked, scores, annotated_list, vis_paths = build_annotated_stack_and_scores(
#             tp_case_folder, candidate_folder, gt_path
#         )
#         if stacked is None:
#             append_skip_row(csv_path, tp_case_folder)   # NEW
#             return "skipped", None

#         # Fit stack to 90% screen height (fast BILINEAR)
#         max_h = int(screen_h * 0.9)
#         if stacked.height > max_h:
#             scale = max_h / stacked.height
#             display_img = stacked.resize((int(stacked.width * scale), max_h), Image.BILINEAR)
#         else:
#             display_img = stacked

#         tk_img = ImageTk.PhotoImage(display_img)
#         label.configure(image=tk_img)
#         label.image = tk_img
#         root.update_idletasks()
#         root.update()

#         print(f"\nCase: {case_name}")
#         print("Keys: 0..4 = select | s = skip | b = back (undo last save) | q = quit")

#         selection_var.set("")
#         root.wait_variable(selection_var)
#         choice = selection_var.get()

#         if choice == 'q':
#             return "quit", None
#         if choice == 'b':
#             return "back", None
#         if choice == 's':
#             print("[SKIP] Skipped this case.")
#             append_skip_row(csv_path, tp_case_folder)   # already logging skip
#             return "skipped", None

#         if choice in ['0','1','2','3','4']:
#             sel_idx = int(choice)
#             sel_vis_path = vis_paths[sel_idx]
#             sel_score = scores[sel_idx]
#             out_name = f"{case_name}__idx{sel_idx}__{sel_vis_path.name}"
#             dst = output_dir / out_name
#             try:
#                 annotated_list[sel_idx].save(dst, quality=95)
#             except Exception as e:
#                 print(f"[ERROR] Failed to save {dst}: {e}")
#                 append_skip_row(csv_path, tp_case_folder)   # NEW
#                 return "skipped", None

#             row = {
#                 "timestamp": datetime.now().isoformat(timespec="seconds"),
#                 "folder_path": str(tp_case_folder),   # store titled_plots/<case> path for resume
#                 "selected_index": str(sel_idx),
#                 "selected_filename": sel_vis_path.name,
#                 "source_path": str(sel_vis_path),
#                 "saved_as": str(dst),
#                 "ciede2000": (f"{sel_score:.6f}" if sel_score is not None else ""),
#                 "status": "selected",
#             }
#             append_csv_row(csv_path, row)
#             print(f"[SAVED] {case_name} -> idx {sel_idx} | ΔE00={row['ciede2000']} → {dst.name}")
#             return "saved", {"folder": tp_case_folder, "selected_idx": sel_idx, "dst_path": dst}

#         print("[WARN] Unknown key. Treating as skip.")
#         append_skip_row(csv_path, tp_case_folder)   # NEW
#         return "skipped", None

#     i = 0
#     try:
#         processed = load_processed_folders(csv_path)

#         # PULL CSV TO INPUT DIRECTORY
#         processed = list(processed)
#         for i, processed_path in enumerate(processed):
#             processed[i] = os.path.join(INPUT_DIR, os.path.basename(processed_path))
#         processed = set(processed)

#         while i < len(case_folders):
#             tp_case_folder = case_folders[i]
#             if str(tp_case_folder) in processed:
#                 print(f"[SKIP] Already processed: {tp_case_folder.name}")
#                 i += 1
#                 continue

#             status, info = show_and_pick(tp_case_folder)

#             if status == "quit":
#                 break

#             if status == "back":
#                 if not history:
#                     print("[INFO] Nothing to go back to.")
#                     continue
#                 last = history.pop()
#                 undo_folder = last["folder"]
#                 removed = remove_last_csv_row_for_folder(csv_path, undo_folder)
#                 try:
#                     if last["dst_path"].exists():
#                         last["dst_path"].unlink()
#                 except Exception as e:
#                     print(f"[WARN] Could not delete {last['dst_path']}: {e}")
#                 processed.discard(str(undo_folder))
#                 try:
#                     i = case_folders.index(undo_folder)
#                 except ValueError:
#                     pass
#                 print(f"[UNDO] Reopened: {undo_folder.name}")
#                 continue

#             if status == "saved":
#                 history.append(info)
#                 processed.add(str(tp_case_folder))
#                 i += 1
#                 continue

#             if status == "skipped":
#                 i += 1
#                 continue

#     finally:
#         try:
#             root.destroy()
#         except:
#             pass

#     print(f"\n[DONE] Session complete. CSV at: {csv_path}")

# if __name__ == "__main__":
#     main()


import csv
import sys
import os
import shutil
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import kornia
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk

# ==============================
# SET YOUR DIRECTORIES HERE
# ==============================
# INPUT_DIR has three subdirs: titled_plots, 200, ground_truth
INPUT_DIR = r"C:\Users\ammic\Desktop\ClariGAN-DL\BrianHE_k-fold_results\HE400\fold_0"
OUTPUT_DIR = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\k-fold-manual_selection"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
CSV_COLUMNS = [
    "timestamp",
    "folder_path",
    "selected_index",
    "selected_filename",
    "source_path",
    "saved_as",
    "ciede2000",
    "status",  # <— NEW
]


# -------------------------------
# Utility: file/dir helpers
# -------------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def find_image_files(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
    files.sort()
    return files

def extract_number_from_titled_plot(filename: str) -> Optional[str]:
    # expects "titled_plot_<number>.*" (robust if extra underscores exist)
    name = Path(filename).stem
    part = name.split("_")[2]
    if part.isdigit():
        return part
    return None

def find_candidate_image(candidate_dir: Path, number: str) -> Optional[Path]:
    # find file like "output_<number>.<ext>"
    if not candidate_dir.is_dir():
        return None
    for p in sorted(candidate_dir.iterdir()):
        if p.is_file() and is_image_file(p) and p.stem.startswith(f"output_{number}"):
            return p
    return None

def find_ground_truth_for_folder(gt_dir: Path, folder_name: str) -> Optional[Path]:
    """
    Heuristics:
      1) exact stem match
      2) stem contained in folder_name or vice versa
      3) if only one image in gt_dir, use it
    """
    if not gt_dir.is_dir():
        return None
    images = [p for p in gt_dir.iterdir() if p.is_file() and is_image_file(p)]
    if not images:
        return None
    for p in images:
        if p.stem == folder_name:
            return p
    for p in images:
        if (p.stem in folder_name) or (folder_name in p.stem):
            return p
    if len(images) == 1:
        return images[0]
    return None

# -------------------------------
# CSV helpers (resume + schema)
# -------------------------------
def ensure_csv_with_schema(csv_path: Path, columns: List[str]) -> None:
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        return

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        return

    old_header = rows[0]
    if old_header == columns:
        return

    old_idx = {name: i for i, name in enumerate(old_header)}
    dict_rows = []
    for r in rows[1:]:
        d = {}
        for name in columns:
            if name in old_idx and old_idx[name] < len(r):
                d[name] = r[old_idx[name]]
            else:
                d[name] = ""
        dict_rows.append(d)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(dict_rows)

def load_processed_folders(csv_path: Path) -> set:
    processed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["folder_path"])
    return processed

def append_csv_row(csv_path: Path, data: Dict[str, str]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(data)

def append_skip_row(csv_path: Path, folder_path: Path) -> None:
    append_csv_row(csv_path, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "folder_path": str(folder_path),
        "selected_index": "",
        "selected_filename": "",
        "source_path": "",
        "saved_as": "",
        "ciede2000": "",
        "status": "skipped",
    })

def remove_last_csv_row_for_folder(csv_path: Path, folder_path: Path) -> Optional[Dict[str, str]]:
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    idx = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i]["folder_path"] == str(folder_path):
            idx = i
            break
    if idx is None:
        return None
    removed = rows.pop(idx)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return removed

# -------------------------------
# Metrics: CIEDE2000 (ΔE00)
# -------------------------------
def load_rgb_native(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        return None

def ciede2000_from_lab(Lab1: torch.Tensor, Lab2: torch.Tensor,
                       kL: float = 1.0, kC: float = 1.0, kH: float = 1.0,
                       eps: float = 1e-12) -> torch.Tensor:
    assert Lab1.shape == Lab2.shape and Lab1.shape[1] == 3, "Expect [N,3,H,W]"
    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    C1 = torch.sqrt(a1*a1 + b1*b1 + eps)
    C2 = torch.sqrt(a2*a2 + b2*b2 + eps)
    Cbar = (C1 + C2) * 0.5

    c25_7 = (Lab1.new_tensor(25.0)).pow(7)
    G = 0.5 * (1.0 - torch.sqrt((Cbar.pow(7) / (Cbar.pow(7) + c25_7)).clamp_min(0)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = torch.sqrt(a1p*a1p + b1*b1 + eps)
    C2p = torch.sqrt(a2p*a2p + b2*b2 + eps)

    h1p = torch.atan2(b1, a1p); h1p = torch.where(h1p < 0, h1p + 2*np.pi, h1p)
    h2p = torch.atan2(b2, a2p); h2p = torch.where(h2p < 0, h2p + 2*np.pi, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(dhp >  np.pi, dhp - 2*np.pi, dhp)
    dhp = torch.where(dhp < -np.pi, dhp + 2*np.pi, dhp)
    dhp = torch.where((C1p*C2p) < eps, Lab1.new_zeros(dhp.shape), dhp)

    dHp = 2.0 * torch.sqrt(C1p*C2p + eps) * torch.sin(dhp * 0.5)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    habs = torch.abs(h1p - h2p)
    hbarp = (h1p + h2p) * 0.5
    hbarp = torch.where((C1p*C2p) < eps, h1p + h2p, hbarp)
    hbarp = torch.where((C1p*C2p) >= eps, torch.where(habs > np.pi,
                        hbarp + np.pi * torch.where((h1p + h2p) < 2*np.pi, 1.0, -1.0), hbarp),
                        hbarp)
    hbarp = hbarp % (2*np.pi)

    T = (1
         - 0.17*torch.cos(hbarp - np.deg2rad(30))
         + 0.24*torch.cos(2*hbarp)
         + 0.32*torch.cos(3*hbarp + np.deg2rad(6))
         - 0.20*torch.cos(4*hbarp - np.deg2rad(63)))

    habp_deg = hbarp * (180.0 / np.pi)
    dTheta = 30.0 * torch.exp(-((habp_deg - 275.0)/25.0)**2)
    RC = 2.0 * torch.sqrt((Cbarp.pow(7) / (Cbarp.pow(7) + c25_7)).clamp_min(0))
    SL = 1.0 + (0.015 * (Lbarp - 50.0).pow(2)) / torch.sqrt(20.0 + (Lbarp - 50.0).pow(2) + eps)
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T
    RT = -torch.sin(2.0 * torch.deg2rad(dTheta)) * RC

    dL_ = dLp / (kL * SL + eps)
    dC_ = dCp / (kC * SC + eps)
    dH_ = dHp / (kH * SH + eps)

    dE2 = dL_*dL_ + dC_*dC_ + dH_*dH_ + RT * dC_ * dH_
    return torch.sqrt(dE2.clamp_min(0.0))

def compute_ciede2000(pred_img: Image.Image, gt_img: Image.Image) -> Optional[float]:
    """
    Computes ΔE00 at native resolution. If sizes differ, pred is resized to GT size (bicubic).
    """
    try:
        if pred_img.size != gt_img.size:
            pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

        pred = np.asarray(pred_img.convert("RGB"), dtype=np.float32) / 255.0
        gt   = np.asarray(gt_img.convert("RGB"), dtype=np.float32) / 255.0

        pred_t = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0)
        gt_t   = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0)

        lab_pred = kornia.color.rgb_to_lab(pred_t)
        lab_gt   = kornia.color.rgb_to_lab(gt_t)
        delta = ciede2000_from_lab(lab_pred, lab_gt)
        return float(delta.mean().item())
    except Exception as e:
        print(f"⚠️ ΔE00 failed: {e}")
        return None

# -------------------------------
# Build annotated stack per folder
# -------------------------------
def build_annotated_stack_and_scores(
    vis_folder: Path, candidate_folder: Path, gt_path: Path
) -> Tuple[Optional[Image.Image], List[Optional[float]], List[Image.Image], List[Path]]:
    """
    Returns:
      stacked_image (PIL, annotated),
      scores: [ΔE00 for each of 5 images],
      per_image_annotated: list of 5 PIL images with overlay,
      vis_paths: the original titled_plot paths in order 0..4
    """
    vis_paths = find_image_files(vis_folder)
    if len(vis_paths) != 5:
        print(f"[SKIP] {vis_folder} (found {len(vis_paths)} images, expected 5).")
        return None, [], [], []

    vis_paths.sort()
    gt_img = load_rgb_native(gt_path)
    if gt_img is None:
        print(f"[WARN] Could not load GT: {gt_path}")
        return None, [], [], []

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    scores: List[Optional[float]] = []
    annotated_list: List[Image.Image] = []

    for idx, vp in enumerate(vis_paths):
        # Find matching candidate by number
        num = extract_number_from_titled_plot(vp.name)
        score = None

        if num is not None:
            cand_path = find_candidate_image(candidate_folder, num)
            if cand_path:
                cand_img = load_rgb_native(cand_path)
                if cand_img is not None:
                    score = compute_ciede2000(cand_img, gt_img)

        scores.append(score)

        # Build display image (annotate titled_plot)
        try:
            im = Image.open(vp).convert("RGB")
        except Exception as e:
            print(f"⚠️ Error opening {vp}: {e}")
            return None, [], [], []

        im = im.copy()
        draw = ImageDraw.Draw(im, "RGBA")

        # index box (top-left)
        draw.rectangle([0, 0, 70, 50], fill=(0, 0, 0, 128))
        draw.text((12, 10), str(idx), fill=(255,255,255,255), font=font)

        # ΔE00 box (top-right)
        text = f"ΔE00: {score:.2f}" if (score is not None) else "ΔE00: N/A"
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 8
        draw.rectangle([im.width - tw - 2*pad, 0, im.width, th + 2*pad], fill=(0,0,0,128))
        draw.text((im.width - tw - pad, pad), text, fill=(255,255,255,255), font=font)

        annotated_list.append(im)

    # Stack vertically
    total_h = sum(i.height for i in annotated_list)
    stacked = Image.new("RGB", (annotated_list[0].width, total_h))
    y = 0
    for im in annotated_list:
        stacked.paste(im, (0, y))
        y += im.height

    return stacked, scores, annotated_list, vis_paths

# -------------------------------
# Main App
# -------------------------------
def main():
    root_dir = Path(INPUT_DIR).expanduser().resolve()
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "summary.csv"
    ensure_csv_with_schema(csv_path, CSV_COLUMNS)
    processed_folders = load_processed_folders(csv_path)

    # Validate expected subfolders inside INPUT_DIR
    titled_root = root_dir / "titled_plots"
    candidate_root = root_dir / "200"
    gt_dir = root_dir / "ground_truth"

    if not (titled_root.is_dir() and candidate_root.is_dir() and gt_dir.is_dir()):
        print("[ERROR] INPUT_DIR must contain 'titled_plots', '200', and 'ground_truth' subfolders.")
        print(f"titled_plots: {titled_root}\n200: {candidate_root}\nground_truth: {gt_dir}")
        sys.exit(1)

    # Case folders live under titled_plots
    case_folders = [p for p in sorted(titled_root.iterdir()) if p.is_dir()]
    if not case_folders:
        print(f"[WARN] No case subfolders found in {titled_root}")
        sys.exit(0)

    # Tk setup
    root = tk.Tk()
    root.title("Select (0-4) | s=skip | b=back | q=quit")
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{min(1400, screen_w-60)}x{min(screen_h-60, screen_h-60)}")

    label = tk.Label(root)
    label.pack(expand=True, fill="both")

    selection_var = tk.StringVar(value="")
    history = []  # [{'folder', 'selected_idx', 'dst_path'}]

    def on_key(event):
        ch = event.char.lower()
        if ch in ['0','1','2','3','4','s','q','b']:
            selection_var.set(ch)

    root.bind("<Key>", on_key)

    def show_and_pick(tp_case_folder: Path):
        case_name = tp_case_folder.name
        gt_path = find_ground_truth_for_folder(gt_dir, case_name)
        if gt_path is None:
            print(f"[SKIP] Ground-truth not found for case '{case_name}'.")
            append_skip_row(csv_path, tp_case_folder)   # NEW
            return "skipped", None

        candidate_folder = candidate_root / case_name
        if not candidate_folder.is_dir():
            print(f"[SKIP] Candidate folder not found: {candidate_folder}")
            append_skip_row(csv_path, tp_case_folder)   # NEW
            return "skipped", None

        stacked, scores, annotated_list, vis_paths = build_annotated_stack_and_scores(
            tp_case_folder, candidate_folder, gt_path
        )
        if stacked is None:
            append_skip_row(csv_path, tp_case_folder)   # NEW
            return "skipped", None

        # Fit stack to 90% screen height (fast BILINEAR)
        max_h = int(screen_h * 0.9)
        if stacked.height > max_h:
            scale = max_h / stacked.height
            display_img = stacked.resize((int(stacked.width * scale), max_h), Image.BILINEAR)
        else:
            display_img = stacked

        tk_img = ImageTk.PhotoImage(display_img)
        label.configure(image=tk_img)
        label.image = tk_img
        root.update_idletasks()
        root.update()

        print(f"\nCase: {case_name}")
        print("Keys: 0..4 = select | s = skip | b = back (undo last save) | q = quit")

        selection_var.set("")
        root.wait_variable(selection_var)
        choice = selection_var.get()

        if choice == 'q':
            return "quit", None
        if choice == 'b':
            return "back", None
        if choice == 's':
            print("[SKIP] Skipped this case.")
            append_skip_row(csv_path, tp_case_folder)   # already logging skip
            return "skipped", None

        if choice in ['0','1','2','3','4']:
            sel_idx = int(choice)
            sel_vis_path = vis_paths[sel_idx]
            sel_score = scores[sel_idx]
            out_name = f"{case_name}__idx{sel_idx}__{sel_vis_path.name}"
            dst = output_dir / out_name
            try:
                annotated_list[sel_idx].save(dst, quality=95)
            except Exception as e:
                print(f"[ERROR] Failed to save {dst}: {e}")
                append_skip_row(csv_path, tp_case_folder)   # NEW
                return "skipped", None

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "folder_path": str(tp_case_folder),   # store titled_plots/<case> path for resume
                "selected_index": str(sel_idx),
                "selected_filename": sel_vis_path.name,
                "source_path": str(sel_vis_path),
                "saved_as": str(dst),
                "ciede2000": (f"{sel_score:.6f}" if sel_score is not None else ""),
                "status": "selected",
            }
            append_csv_row(csv_path, row)
            print(f"[SAVED] {case_name} -> idx {sel_idx} | ΔE00={row['ciede2000']} → {dst.name}")
            return "saved", {"folder": tp_case_folder, "selected_idx": sel_idx, "dst_path": dst}

        print("[WARN] Unknown key. Treating as skip.")
        append_skip_row(csv_path, tp_case_folder)   # NEW
        return "skipped", None

    # === Random sampling params ===
    RNG_SEED = 42          # Set to an int for reproducibility, or None to vary each run
    SAMPLE_SIZE = None     # e.g., 200 to process a random subset; or None for "all unprocessed"

    try:
        processed = load_processed_folders(csv_path)

        # Map processed paths to current INPUT_DIR base (your original PULL logic)
        processed_list = list(processed)
        for j, processed_path in enumerate(processed_list):
            processed_list[j] = os.path.join(INPUT_DIR, os.path.basename(processed_path))
        processed = set(processed_list)

        # Build UNPROCESSED list
        unprocessed = [p for p in case_folders if str(p) not in processed]

        if not unprocessed:
            print("[INFO] Nothing left to process. All case folders are already in CSV.")
            print(f"[DONE] Session complete. CSV at: {csv_path}")
            return

        # Randomize (and optionally sample subset)
        if RNG_SEED is not None:
            random.seed(RNG_SEED)
        if SAMPLE_SIZE is not None and SAMPLE_SIZE > 0:
            k = min(SAMPLE_SIZE, len(unprocessed))
            unprocessed = random.sample(unprocessed, k)
        else:
            random.shuffle(unprocessed)

        # Iterate the randomized queue
        queue = unprocessed[:]
        idx = 0
        while idx < len(queue):
            tp_case_folder = queue[idx]

            status, info = show_and_pick(tp_case_folder)

            if status == "quit":
                break

            if status == "back":
                if not history:
                    print("[INFO] Nothing to go back to.")
                    continue
                last = history.pop()
                undo_folder = last["folder"]
                removed = remove_last_csv_row_for_folder(csv_path, undo_folder)
                try:
                    if last["dst_path"].exists():
                        last["dst_path"].unlink()
                except Exception as e:
                    print(f"[WARN] Could not delete {last['dst_path']}: {e}")

                # Make sure this folder is considered unprocessed again.
                processed.discard(str(undo_folder))

                # Reopen the same item by moving idx to its position (or inserting it)
                try:
                    reopen_pos = queue.index(undo_folder)
                    idx = reopen_pos
                except ValueError:
                    queue.insert(idx, undo_folder)
                print(f"[UNDO] Reopened: {undo_folder.name}")
                continue

            if status == "saved":
                history.append(info)
                processed.add(str(tp_case_folder))
                idx += 1
                continue

            if status == "skipped":
                # Treat skipped as processed for future runs; remove this line to revisit skips.
                processed.add(str(tp_case_folder))
                idx += 1
                continue

    finally:
        try:
            root.destroy()
        except:
            pass

    print(f"\n[DONE] Session complete. CSV at: {csv_path}")

if __name__ == "__main__":
    main()
