import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os


from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import torch

class ConsistentTransform:
    def __init__(self, image_size, p=0.0, augmentations=True):
        self.image_size = image_size
        self.augmentations = augmentations
        self.params = None  # To store the transformation parameters
        
        # Define the augmentations
        if augmentations:
            self.augmentation_transform = {
                "horizontal_flip": transforms.RandomHorizontalFlip(p=p),
                "vertical_flip": transforms.RandomVerticalFlip(p=0.5),
                "rotation": transforms.RandomRotation(degrees=180),
                "translation": transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                "resized_crop": transforms.RandomResizedCrop(
                    size=self.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                )
            }
        
        # Default resizing transformation
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            # transforms.ToTensor()
        ])
    
    def save_params(self, image):
        """Apply transformations to the first image and save the parameters."""
        if self.augmentations:
            # Choose random crop size (80% to 100% of original size)
            crop_height = int(self.image_size[0] * random.uniform(0.8, 1.0))
            crop_width = int(self.image_size[1] * random.uniform(0.8, 1.0))

            # Choose a valid top-left corner so the crop stays inside image bounds
            top = random.randint(0, self.image_size[0] - crop_height)
            left = random.randint(0, self.image_size[1] - crop_width)

            # Save parameters for each augmentation
            self.params = {
                "horizontal_flip": random.random() < 0.5,
                "vertical_flip": random.random() < 0.5,
                "rotation": random.uniform(-180, 180),
                "translation": {
                    "translate_x": random.uniform(-0.05, 0.05),
                    "translate_y": random.uniform(-0.05, 0.05)
                },
                "resized_crop": {
                    "top": top,
                    "left": left,
                    "height": crop_height,
                    "width": crop_width
                }
            }
            return self.apply_saved_params(image)
        else:
            return self.resize_transform(image)
    
    def apply_saved_params(self, image):
        """Apply saved transformations to the given image."""
        if self.params is None:
            raise ValueError("Transformation parameters have not been saved yet!")
        
        # Apply transformations with saved parameters
        img = image
        if self.params["horizontal_flip"]:
            img = F.hflip(img)
        if self.params["vertical_flip"]:
            img = F.vflip(img)
        img = F.rotate(img, self.params["rotation"])
        img = F.affine(
            img, 
            angle=0, 
            translate=(
                int(self.params["translation"]["translate_x"] * self.image_size[1]),
                int(self.params["translation"]["translate_y"] * self.image_size[0])
            ), 
            scale=1.0, 
            shear=0
        )
        img = F.resized_crop(
            img,
            top=self.params["resized_crop"]["top"],
            left=self.params["resized_crop"]["left"],
            height=self.params["resized_crop"]["height"],
            width=self.params["resized_crop"]["width"],
            size=(self.image_size[0], self.image_size[1])
        )
        return self.resize_transform(img)

class ConsistentFlip:
    def __init__(self, image_size, p=0.5):
        """
        Initialize the transformation pipeline with parameter saving.

        Args:
        ----
        image_size (int): The target size for resizing.
        p (float): Probability of applying RandomHorizontalFlip.
        """
        self.image_size = image_size
        self.p = p
        self.params = None  # To store the transformation parameters

    def save_params(self, image):
        """
        Save the transformation parameters for consistent transformations.
        
        Args:
        ----
        image (PIL.Image or Tensor): The first image to apply and save transformations.

        Returns:
        -------
        Tensor: Transformed image.
        """
        self.params = {
            "horizontal_flip": random.random() < self.p,  # Random flip
        }

        # Apply transformations using these parameters
        return self.apply_saved_params(image)

    def apply_saved_params(self, image):
        """
        Apply saved transformation parameters to an image.

        Args:
        ----
        image (PIL.Image or Tensor): Input image to transform.

        Returns:
        -------
        Tensor: Transformed image.
        """
        if self.params is None:
            raise ValueError("Transformation parameters have not been saved yet!")
        
        # Apply transformations using saved parameters
        img = image
        if self.params["horizontal_flip"]:
            img = F.hflip(img)
        
        # Resize and convert to tensor
        img = F.resize(img, [self.image_size[0], self.image_size[1]])
        img = F.to_tensor(img)

        return img
    


@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train', multi_modal=False, split_set=None):
        super().__init__()
        self.multi_modal = multi_modal
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        if(split_set is None):
            image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
            image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        else:
            r1_list, r3_list = zip(*split_set)

            # Convert from tuples to lists (optional, depending on what you need)
            image_paths_cond = list(r1_list)
            image_paths_ori = list(r3_list)
        
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self.augmentations = stage == 'train'

        self.imgs_ori = ImagePathDataset(image_paths=image_paths_ori, image_size=self.image_size, flip=False, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths=image_paths_cond, image_size=self.image_size, flip=False, to_normal=self.to_normal)
        # BF + DF
        if(multi_modal == True):
            image_paths_cond2nd = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/C'))
            self.imgs_cond2nd = ImagePathDataset(image_paths=image_paths_cond2nd, image_size=self.image_size, flip=False, to_normal=self.to_normal)



        self.len = 8 * len(self.imgs_ori) if stage == 'train' else len(self.imgs_ori)
        self._length = len(self.imgs_ori)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.augmentations:

            p = 0.0
            skip_aug = False
            dataset_length = self.__len__()
            flip_tresh = int(dataset_length/(2*self._length))

            flip_num = int(index/self._length)

            if self.flip and (flip_num > flip_tresh):
                p = 1.0


            # Determine if augmentation applies
            if (index < self._length) or ((index > flip_tresh*self._length) and (index > (flip_tresh+1)*self._length)):
                skip_aug = True

            if index >= self._length:
                index = index % self._length
            
            if(self.multi_modal == True):
                (img_ori, stem_ori), (img_cond, stem_cond), (img_cond2nd, stem_cond2nd) = self.imgs_ori[index], self.imgs_cond[index], self.imgs_cond2nd[index]
            else:
                (img_ori, stem_ori), (img_cond, stem_cond) = self.imgs_ori[index], self.imgs_cond[index]
                # try:
                #     assert(os.path.basename(stem_ori).replace('R3', 'R1') == os.path.basename(stem_cond))
                # except:
                #     print(os.path.basename(stem_ori), "cannot be paired with ", os.path.basename(stem_cond))
                #     raise Exception


            if(skip_aug == False):
                transform = ConsistentTransform(self.image_size, p=p, augmentations=True)
                img_ori = transform.save_params(img_ori)
                img_cond = transform.apply_saved_params(img_cond)

                if(self.multi_modal == True):
                    img_cond2nd = transform.apply_saved_params(img_cond2nd)
        else:
            if(self.multi_modal == True):
                (img_ori, stem_ori), (img_cond, stem_cond), (img_cond2nd, stem_cond2nd) = self.imgs_ori[index], self.imgs_cond[index], self.imgs_cond2nd[index]
            else:
                (img_ori, stem_ori), (img_cond, stem_cond) = self.imgs_ori[index], self.imgs_cond[index]

        if(self.multi_modal == True):
            # print(img_cond.shape, img_cond2nd.shape, img_ori.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow((img_cond.permute((1,2,0))/2+0.5))
            # plt.title("BF")
            # plt.show()
            # plt.imshow((img_cond2nd.permute((1,2,0))/2+0.5))
            # plt.title("DF")
            # plt.show()

            # Zero-pad img_ori to shape (2, 3, 256, 256)
            img_ori_padded = torch.zeros(2, 3, 256, 256)
            img_ori_padded[1] = img_ori  # Place the original image in the last position

            # Stack img_cond and img_cond2nd to shape (2, 3, 256, 256)
            img_cond_stacked = torch.stack([img_cond2nd, img_cond], dim=0)

            # print("Padded img_ori shape:", img_ori_padded.shape)  # Expected (2,3,256,256)
            # print("Stacked img_cond shape:", img_cond_stacked.shape)  # Expected (2,3,256,256)
            return (img_ori_padded, stem_ori), (img_cond_stacked, stem_cond)
        else:
            return (img_ori, stem_ori), (img_cond, stem_cond)

import os
import re
import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np

@Registers.datasets.register_with_name('balanced_aligned')
class BalancedAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train', multi_modal=False):
        super().__init__()
        self.multi_modal = multi_modal
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self.augmentations = stage == 'train'
        
        # Create datasets
        self.imgs_ori = ImagePathDataset(image_paths=image_paths_ori, image_size=self.image_size, 
                                         flip=False, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths=image_paths_cond, image_size=self.image_size, 
                                          flip=False, to_normal=self.to_normal)
        
        # BF + DF (multi-modal)
        if multi_modal:
            image_paths_cond2nd = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/C'))
            self.imgs_cond2nd = ImagePathDataset(image_paths=image_paths_cond2nd, image_size=self.image_size, 
                                                 flip=False, to_normal=self.to_normal)
        
        # Base length of the dataset (original data)
        self._length = len(self.imgs_ori)
        
        # Analyze class distribution
        self.letter_size_indices = self._group_by_letter_and_size(image_paths_ori)
        
        # Print detailed class distribution before balancing
        self._print_original_distribution()
        
        if stage == 'train':
            # Find max sizes per letter and across all letters for each size
            self.max_sizes = self._get_max_sizes()
            
            # Print max sizes info
            self._print_max_sizes()
            
            # Create balanced index list with dummy indices
            self.all_indices, self.dummy_to_class = self._create_balanced_indices()
            self.len = len(self.all_indices)
            
            # Print oversampling statistics
            self._print_oversampling_stats()
        else:
            # For validation/test, use original indices without oversampling
            self.all_indices = list(range(self._length))
            self.dummy_to_class = {}
            self.len = self._length
        
        print(f"Original dataset size: {self._length}, Effective dataset size with balancing: {self.len}")

    def _group_by_letter_and_size(self, image_paths):
        """Group image indices by letter and size."""
        # Structure: size -> letter -> list of indices
        letter_size_indices = {
            '5x5': defaultdict(list),
            '10x10': defaultdict(list)
        }
        
        # Regular expression to extract the letter
        letter_pattern = re.compile(r'R[13]-([A-Za-z](?:part\d+)?)')
        
        for idx, path in enumerate(image_paths):
            filename = os.path.basename(path)
            letter_match = letter_pattern.search(filename)
            if not letter_match:
                continue
                
            letter = letter_match.group(1)
            
            # Focus only on 5x5 and 10x10 files
            if "_5x5" in filename:
                letter_size_indices['5x5'][letter].append(idx)
            elif "_10x10" in filename:
                letter_size_indices['10x10'][letter].append(idx)
        
        return letter_size_indices
    
    def _get_max_sizes(self):
        """Get the maximum sizes for each letter and each size category."""
        max_sizes = {
            'per_letter': {
                '5x5': {},
                '10x10': {}
            },
            'global': {
                '5x5': 0,
                '10x10': 0
            }
        }
        
        # Get max size per letter
        for size in ['5x5', '10x10']:
            for letter, indices in self.letter_size_indices[size].items():
                max_sizes['per_letter'][size][letter] = len(indices)
            
            # Get global max size
            if self.letter_size_indices[size]:
                max_sizes['global'][size] = max(len(indices) for indices in self.letter_size_indices[size].values())
        
        return max_sizes
    
    def _print_original_distribution(self):
        """Print the distribution of classes by letter and size."""
        print("\nOriginal class distribution:")
        print("Size\tLetter\tCount")
        print("-" * 30)
        
        total_samples = 0
        
        # Print counts for each size-letter combination
        for size in ['5x5', '10x10']:
            size_total = 0
            
            for letter, indices in sorted(self.letter_size_indices[size].items()):
                count = len(indices)
                size_total += count
                print(f"{size}\t{letter}\t{count}")
            
            total_samples += size_total
            print(f"{size} Total: {size_total}")
            print("-" * 30)
        
        print(f"Grand Total: {total_samples}")
    
    def _print_max_sizes(self):
        """Print max sizes per letter and global max sizes."""
        print("\nMaximum sizes per letter:")
        print("Letter\t5x5\t10x10")
        print("-" * 30)
        
        # Get all unique letters across both sizes
        all_letters = set()
        for size in ['5x5', '10x10']:
            all_letters.update(self.letter_size_indices[size].keys())
        
        # Print max size for each letter
        for letter in sorted(all_letters):
            size_5x5 = self.max_sizes['per_letter']['5x5'].get(letter, 0)
            size_10x10 = self.max_sizes['per_letter']['10x10'].get(letter, 0)
            print(f"{letter}\t{size_5x5}\t{size_10x10}")
        
        print("-" * 30)
        print("Global maximum sizes across all letters:")
        print(f"5x5: {self.max_sizes['global']['5x5']}")
        print(f"10x10: {self.max_sizes['global']['10x10']}")
    
    def _print_oversampling_stats(self):
        """Print statistics about oversampling, showing per-section values with percentage increases."""
        print("\nOversampling statistics:")
        
        # Count original indices per section
        original_indices_per_section = sum(len(self.letter_size_indices[size][letter]) 
                                        for size in ['5x5', '10x10'] 
                                        for letter in self.letter_size_indices[size])
        
        # Total original indices (all sections)
        original_indices_total = original_indices_per_section * 8 if self.augmentations else original_indices_per_section
        
        # Count dummy indices by class
        dummy_by_class = defaultdict(int)
        for _, (size, letter) in self.dummy_to_class.items():
            dummy_by_class[(size, letter)] += 1
        
        # Total dummy indices
        dummy_indices = sum(dummy_by_class.values())
        
        # Total indices overall
        total_indices = original_indices_total + dummy_indices
        
        # Per-section statistics
        if self.augmentations:
            section_count = 8
            print(f"Number of sections: {section_count}")
            print(f"Original indices per section: {original_indices_per_section}")
        else:
            section_count = 1
            print("Single section dataset (no augmentations)")
        
        print(f"Total indices in dataset: {total_indices}")
        print(f"Total original indices: {original_indices_total}")
        print(f"Total dummy indices: {dummy_indices} ({dummy_indices/total_indices*100:.1f}%)")
        
        # Per-class statistics
        print("\nPer-class statistics (per section):")
        print("Size\tLetter\tOriginal\tDummy/Section\t% Increase\tTotal/Section\tFinal Total")
        print("-" * 90)
        
        for size in ['5x5', '10x10']:
            size_total_original = 0
            size_total_dummy = 0
            
            for letter in sorted(self.letter_size_indices[size].keys()):
                # Original count for this class
                original = len(self.letter_size_indices[size][letter])
                size_total_original += original
                
                # Dummy count per section
                dummy_per_section = max(0, self.max_sizes['global'][size] - original)
                size_total_dummy += dummy_per_section * section_count
                
                # Percentage increase per section
                pct_increase = (dummy_per_section / original * 100) if original > 0 else float('inf')
                
                # Total per section (original + dummy)
                total_per_section = original + dummy_per_section
                
                # Final total across all sections
                final_total = original * section_count + dummy_per_section * section_count
                
                # Only show if there's oversampling
                if dummy_per_section > 0:
                    if pct_increase == float('inf'):
                        pct_increase_str = "∞"
                    else:
                        pct_increase_str = f"{pct_increase:.1f}%"
                    
                    print(f"{size}\t{letter}\t{original}\t\t{dummy_per_section}\t\t{pct_increase_str}\t\t{total_per_section}\t\t{final_total}")
            
            # Calculate overall percentage increase for this size
            size_pct_increase = (size_total_dummy // section_count) / size_total_original * 100 if size_total_original > 0 else 0
            
            # Print size category totals
            print(f"{size} totals:\t{size_total_original}\t\t{size_total_dummy // section_count}\t\t" + 
                f"{size_pct_increase:.1f}%\t\t{size_total_original + size_total_dummy // section_count}\t\t" +
                f"{size_total_original * section_count + size_total_dummy}")
            print("-" * 90)
        
        # Balance information
        print("\nClasses balanced to (per section):")
        print("Size\tTarget Size\tLetters Balanced")
        print("-" * 60)
        
        for size in ['5x5', '10x10']:
            target = self.max_sizes['global'][size]
            balanced_letters = []
            
            for letter in sorted(self.letter_size_indices[size].keys()):
                if len(self.letter_size_indices[size][letter]) < target:
                    balanced_letters.append(letter)
            
            if balanced_letters:
                print(f"{size}\t{target}\t\t{', '.join(balanced_letters)}")
            else:
                print(f"{size}\t{target}\t\tNone (already balanced)")
    
    def _print_oversampling_stats_old(self):
        """Print statistics about oversampling."""
        print("\nOversampling statistics:")
        
        original_indices = sum(len(self.letter_size_indices[size][letter]) 
                              for size in ['5x5', '10x10'] 
                              for letter in self.letter_size_indices[size])
        
        original_indices_per_section = original_indices
        original_indices_total = original_indices * 8 if self.augmentations else original_indices
        
        # Count dummy indices by class
        dummy_by_class = defaultdict(int)
        for _, (size, letter) in self.dummy_to_class.items():
            dummy_by_class[(size, letter)] += 1
        
        dummy_indices = sum(dummy_by_class.values())
        total_indices = original_indices_total + dummy_indices
        
        print(f"Total indices: {total_indices}")
        print(f"Original indices: {original_indices_total}")
        print(f"Dummy indices: {dummy_indices} ({dummy_indices/total_indices*100:.1f}%)")
        
        print("\nDummy indices by class:")
        print("Size\tLetter\tOriginal\tDummy\tTotal\tIncrease")
        print("-" * 60)
        
        for size in ['5x5', '10x10']:
            for letter in sorted(self.letter_size_indices[size].keys()):
                original = len(self.letter_size_indices[size][letter])
                dummy_per_section = max(0, self.max_sizes['global'][size] - original)
                dummy_total = dummy_per_section * 8 if self.augmentations else dummy_per_section
                total_per_class = original * 8 + dummy_total if self.augmentations else original + dummy_total
                increase = (dummy_total / (original * 8) * 100) if self.augmentations and original > 0 else 0
                
                if dummy_total > 0:  # Only show classes that are oversampled
                    print(f"{size}\t{letter}\t{original}\t\t{dummy_total}\t{total_per_class}\t{increase:.1f}%")
        
        print("\nClasses balanced to:")
        print("Size\tTarget Size\tLetters Balanced")
        print("-" * 60)
        
        for size in ['5x5', '10x10']:
            target = self.max_sizes['global'][size]
            balanced_letters = []
            
            for letter in sorted(self.letter_size_indices[size].keys()):
                if len(self.letter_size_indices[size][letter]) < target:
                    balanced_letters.append(letter)
            
            if balanced_letters:
                print(f"{size}\t{target}\t\t{', '.join(balanced_letters)}")
            else:
                print(f"{size}\t{target}\t\tNone (already balanced)")
    
    def _create_balanced_indices(self):
        """Create a balanced list of indices with dummy indices for underrepresented classes."""
        all_indices = []
        dummy_to_class = {}
        next_dummy_index = self._length * 8  # Start dummy indices after all real indices
        
        # Process each section
        for section_id in range(8):
            section_start = section_id * self._length
            section_indices = []
            
            # Process each size category
            for size in ['5x5', '10x10']:
                global_max = self.max_sizes['global'][size]
                
                # Process each letter in this size category
                for letter, original_indices in self.letter_size_indices[size].items():
                    original_size = len(original_indices)
                    
                    # Add all original indices for this section
                    for idx in original_indices:
                        section_indices.append(section_start + idx)
                    
                    # Create dummy indices for underrepresented classes
                    if original_size < global_max:
                        # How many dummy indices to add - only for this section!
                        dummy_count = global_max - original_size
                        
                        # Create dummy indices
                        for _ in range(dummy_count):
                            # Map dummy index to class info
                            dummy_to_class[next_dummy_index] = (size, letter)
                            # Add dummy index to the section
                            section_indices.append(next_dummy_index)
                            next_dummy_index += 1
            
            # Add all indices for this section to the master list
            all_indices.extend(section_indices)
        
        # Shuffle the indices to avoid blocks of similar classes
        random.shuffle(all_indices)
        
        return all_indices, dummy_to_class
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # Get the index from our balanced index list
        mapped_index = self.all_indices[index]
        
        # Check if this is a dummy index that needs dynamic oversampling
        if mapped_index in self.dummy_to_class:
            # Get the class info for this dummy index
            size, letter = self.dummy_to_class[mapped_index]
            
            # Get a random index from this class
            indices = self.letter_size_indices[size][letter]
            if indices:
                # Randomly select an index from this class
                real_index = random.choice(indices)
                
                # Determine which section this dummy index belongs to
                section_id = min(7, (mapped_index // self._length) if mapped_index < self._length * 8 else 0) if self.augmentations else 0
            else:
                # Fallback if class has no indices
                real_index = 0
                section_id = 0
        else:
            # For real indices, calculate section and real index
            section_id = mapped_index // self._length if self.augmentations else 0
            real_index = mapped_index % self._length
        
        # Determine augmentation parameters based on section
        if self.augmentations:
            p = 0.0
            skip_aug = False
            
            # Original logic for flipping based on section ID
            flip_thresh = 1  # Default threshold at section 1
            if self.flip and (section_id > flip_thresh):
                p = 1.0
                
            # Sections 0 and sections between flip_thresh and flip_thresh+1 skip augmentation
            if (section_id == 0) or ((section_id > flip_thresh) and (section_id > (flip_thresh+1))):
                skip_aug = True
                
            if self.multi_modal:
                (img_ori, stem_ori), (img_cond, stem_cond), (img_cond2nd, stem_cond2nd) = (
                    self.imgs_ori[real_index], 
                    self.imgs_cond[real_index], 
                    self.imgs_cond2nd[real_index]
                )
            else:
                (img_ori, stem_ori), (img_cond, stem_cond) = self.imgs_ori[real_index], self.imgs_cond[real_index]
                try:
                    assert(os.path.basename(stem_ori).replace('R3', 'R1') == os.path.basename(stem_cond))
                except:
                    print(os.path.basename(stem_ori), "cannot be paired with ", os.path.basename(stem_cond))
                    raise Exception

            if not skip_aug:
                transform = ConsistentTransform(self.image_size, p=p, augmentations=True)
                img_ori = transform.save_params(img_ori)
                img_cond = transform.apply_saved_params(img_cond)

                if self.multi_modal:
                    img_cond2nd = transform.apply_saved_params(img_cond2nd)
        else:
            if self.multi_modal:
                (img_ori, stem_ori), (img_cond, stem_cond), (img_cond2nd, stem_cond2nd) = (
                    self.imgs_ori[real_index], 
                    self.imgs_cond[real_index], 
                    self.imgs_cond2nd[real_index]
                )
            else:
                (img_ori, stem_ori), (img_cond, stem_cond) = self.imgs_ori[real_index], self.imgs_cond[real_index]

        if self.multi_modal:
            # Zero-pad img_ori to shape (2, 3, 256, 256)
            img_ori_padded = torch.zeros(2, 3, 256, 256)
            img_ori_padded[1] = img_ori  # Place the original image in the last position

            # Stack img_cond and img_cond2nd to shape (2, 3, 256, 256)
            img_cond_stacked = torch.stack([img_cond2nd, img_cond], dim=0)

            return (img_ori_padded, stem_ori), (img_cond_stacked, stem_cond)
        else:
            return (img_ori, stem_ori), (img_cond, stem_cond)
@Registers.datasets.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)
