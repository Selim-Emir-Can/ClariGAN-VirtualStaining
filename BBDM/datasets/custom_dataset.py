from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

import os
import re
from PIL import Image
import numpy as np
import torch
import random
from torchvision.transforms import functional as TF

import tifffile
import matplotlib.pyplot as plt

def process_single_path(img_path):
    """Parse metadata from a single image path."""
    img_path = os.path.normpath(img_path)
    filename = os.path.basename(img_path)
    
    match = re.search(r"_s(\d+)_r(\d+)_c(\d+)_", filename)
    if match:
        slice_num = int(match.group(1))
        row = int(match.group(2))
        column = int(match.group(3))
    else:
        slice_num = -1
        row = -1
        column = -1

    return slice_num, row, column


def load_image(img_path, resolution=None):
    """Loads an image, ensures it is in RGB format, and resizes if resolution is provided."""
    try:
        if(img_path.endswith('.png')):
            image = Image.open(img_path)
        elif(img_path.endswith('.tif')):
            with tifffile.TiffFile(img_path) as tif:
                # Read the image data
                image = tif.asarray()
            assert(image.shape[2] == 3)

            image = Image.fromarray(image)           
            
        else:
            raise Exception('Unsupported image type: ', img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if resolution:
            image = image.resize(resolution, Image.Resampling.LANCZOS)
        return np.array(image)
    except Exception as e:
        print(f"Error loading image: {img_path}\nException: {e}")
        return None

from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch

import random
import torchvision.transforms as T
import torchvision.transforms.functional as F

class ConsistentTransform:
    def __init__(self, image_size, augmentations=True):
        self.image_size = image_size
        self.augmentations = augmentations
        self.params = None  # To store the transformation parameters
        
        # Define the augmentations
        if augmentations:
            self.augmentation_transform = {
                "horizontal_flip": transforms.RandomHorizontalFlip(p=0.5),
                "vertical_flip": transforms.RandomVerticalFlip(p=0.5),
                "rotation": transforms.RandomRotation(degrees=10),
                "color_jitter": transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                "resized_crop": transforms.RandomResizedCrop(
                    size=self.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                )
            }
        
        # Default resizing transformation
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
    
    def save_params(self, image):
        """Apply transformations to the first image and save the parameters."""
        if self.augmentations:
            # Save parameters for each augmentation
            self.params = {
                "horizontal_flip": random.random() < 0.5,
                "vertical_flip": random.random() < 0.5,
                "rotation": random.uniform(-10, 10),
                "color_jitter": {
                    "brightness": random.uniform(0.8, 1.2),
                    "contrast": random.uniform(0.8, 1.2),
                    "saturation": random.uniform(0.8, 1.2),
                    "hue": random.uniform(-0.1, 0.1)
                },
                "resized_crop": {
                    "top": random.uniform(0, 0.2),
                    "left": random.uniform(0, 0.2),
                    "height": int(self.image_size[0] * random.uniform(0.8, 1.0)),
                    "width": int(self.image_size[1] * random.uniform(0.8, 1.0))
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
            img = transforms.functional.hflip(img)
        if self.params["vertical_flip"]:
            img = transforms.functional.vflip(img)
        img = transforms.functional.rotate(img, self.params["rotation"])
        img = transforms.functional.adjust_brightness(img, self.params["color_jitter"]["brightness"])
        img = transforms.functional.adjust_contrast(img, self.params["color_jitter"]["contrast"])
        img = transforms.functional.adjust_saturation(img, self.params["color_jitter"]["saturation"])
        img = transforms.functional.adjust_hue(img, self.params["color_jitter"]["hue"])
        img = transforms.functional.resized_crop(
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
    

class ImagePathDatasetCustom(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, augmentations=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.augmentations = augmentations
        self.to_normal = to_normal

        # Define default resizing
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        # Augmentations for training
        self.augmentation_transform = transforms.Compose([
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ])

    def __len__(self):
        dataset_length = self._length
        
        if self.augmentations:
            dataset_length = dataset_length * 4
            
        if self.flip:
            dataset_length = dataset_length * 2
        return dataset_length
    
    def __getitem__(self, index):
        p = 0.0
        skip_aug = False
        dataset_length = self.__len__()
        flip_tresh = int(dataset_length/(2*self._length))

        flip_num = int(index/self._length)

        if self.flip and (flip_num > flip_tresh):
            p = 1.0


        # Determine if flipping applies
        if (index < self._length) or ((index > flip_tresh*self._length) and (index > (flip_tresh+1)*self._length)):
            skip_aug = True

        if index >= self._length:
            index = index % self._length

        img_path = self.image_paths[index]
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(f"Error loading image: {img_path}, {e}")
            return None

        # Ensure image is in RGB mode
        if not image.mode == 'RGB':
            image = image.convert('RGB')

        # Apply augmentations
        if self.augmentations and (skip_aug == False):
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=p),
                self.augmentation_transform
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=p),
                self.resize_transform
            ])

        # image = transform(image)

        def is_too_dark(img_tensor, threshold=0.02, dark_percent=0.95):
            """
            Returns True if at least `dark_percent` of the pixels in img_tensor have values below `threshold`.
            
            Args:
                img_tensor (torch.Tensor): A tensor image (assumed to be normalized to [0,1] or similar).
                threshold (float): The pixel intensity below which a pixel is considered dark.
                dark_percent (float): The fraction of pixels that must be dark to consider the image too dark.
            """
            # Flatten the tensor; note that this works regardless of the number of channels.
            num_dark_pixels = (img_tensor < threshold).float().sum()
            total_pixels = img_tensor.numel()
            return (num_dark_pixels / total_pixels) >= dark_percent

        def safe_transform(pil_img, transform, threshold=0.02, dark_percent=0.95, max_attempts=10):
            """
            Applies the transformation to a PIL image repeatedly until the resulting tensor is not too dark,
            or until max_attempts is reached.
            
            Args:
                pil_img (PIL.Image.Image): The original PIL image.
                transform (callable): A transformation that converts the PIL image into a torch.Tensor.
                threshold (float): The intensity below which a pixel is considered dark.
                dark_percent (float): The fraction of pixels that must be dark for the image to be rejected.
                max_attempts (int): Maximum number of attempts.
                
            Returns:
                torch.Tensor: The transformed image tensor.
            """
            # Keep a backup of the original PIL image.
            original = pil_img.copy()
            for attempt in range(max_attempts):
                transformed = transform(original)
                if not is_too_dark(transformed, threshold, dark_percent):
                    return transformed
            # If all attempts yield an image that is too dark, return the last result.
            return transformed

        # Example usage:
        # Assuming `image` is your tensor and `transform` is your transformation function.
        image = safe_transform(image, transform)


        # Normalize to [-1, 1] if required
        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        # Extract image name
        image_name = Path(img_path).stem
        return image, image_name
"""
class ImagePathDatasetCustom(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, augmentations=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.augmentations = augmentations
        self.to_normal = to_normal

        # Define default resizing
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        dataset_length = self._length
        
        if self.augmentations:
            dataset_length = dataset_length * 4
            
        if self.flip:
            dataset_length = dataset_length * 2
        return dataset_length
    
    def __getitem__(self, index):
        p = 0.0
        if(self.augmentations == False):
            if index >= self._length:
                index = index - self._length
                p = 1.0
        else:
            if index >= (4 * self._length):
                index = index - (index//self._length)*self._length
                p = 1.0

        transform = ConsistentFlip(image_size=self.image_size, p=p)
        augmentation_transform = None
        if(self.augmentations == True):
            augmentation_transform = ConsistentTransform(image_size=self.image_size, augmentations=True)
            

        img_path = self.image_paths[index]
        image_name = Path(img_path).stem

        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        # flip images if p=1.0
        image = transform.save_params(image)

        # apply random augmentations to batch
        if(self.augmentations == True):
            image = augmentation_transform.save_params(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
        
        
        return image, image_name
"""    
