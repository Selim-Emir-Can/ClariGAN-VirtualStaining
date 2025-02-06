import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class CustomImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths) * 8  # Increase the dataset size by 8x
        if self.size is not None and self.size > 0:
            self.cropper = (
                A.RandomResizedCrop(height=self.size, width=self.size, scale=(0.8, 1.0))
                if self.random_crop else A.CenterCrop(height=self.size, width=self.size)
            )
            self.final_resizer = A.Resize(height=self.size, width=self.size)  # Ensure exact output size
            self.affine_augmentations = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=0.1,  # Translation up to 10% of the image size
                    scale_limit=0.2,  # Scale between 0.8x and 1.0x (±20%)
                    rotate_limit=30,  # Rotate within a range of ±30 degrees
                    p=1.0
                ),
                self.cropper,
            ])
            self.flip_augmentation = A.Compose([
                self.cropper,
                A.HorizontalFlip(p=1.0)
            ])
        else:
            self.affine_augmentations = lambda **kwargs: kwargs
            self.flip_augmentation = lambda **kwargs: kwargs
        
        self.final_resizer = A.Resize(height=self.size, width=self.size)  # Ensure exact output size


    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, augment_idx):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # Debug: Visualize the original image
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.title("Original Image")
        # plt.imshow(image)
        # plt.axis("off")


        # Determine which augmentations to apply
        if augment_idx < len(self.labels["file_path_"]):
            # No augmentations
            augmented = {"image": image}
        elif augment_idx < 4 * len(self.labels["file_path_"]):
            # Affine augmentations + random cropping
            augmented = self.affine_augmentations(image=image)
        else:
            # Flip the image
            augmented = self.flip_augmentation(image=image)

            if augment_idx > 5 * len(self.labels["file_path_"]):
                # Affine augmentations + random cropping
                augmented = self.affine_augmentations(image=image)

        image = augmented["image"]

        # Debug: Visualize the augmented image
        # plt.subplot(1, 2, 2)
        # plt.title("Augmented Image")
        # plt.imshow(image)
        # plt.axis("off")
        # plt.show()


        # Apply resizing to all cases
        image = self.final_resizer(image=image)["image"]
        assert(image.shape == (256,256,3))

        # Normalize to (-1, 1)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, idx):
        dataset_idx = idx % len(self.labels["file_path_"])  # Original dataset index
        augment_idx = idx  # Determine augmentation behavior

        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][dataset_idx], augment_idx)

        for k in self.labels:
            example[k] = self.labels[k][dataset_idx]

        return example
    
"""
import torchvision.transforms as transforms

class ImagePathDatasetCustom(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

        # BBDM augmentations
        self.flip = True
        self.augmentations = True

        # Define default resizing
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        # Augmentations for training
        self.augmentation_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
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

        image = transform(image)


        # Extract image name
        image_name = Path(img_path).stem
        return image, image_name
"""