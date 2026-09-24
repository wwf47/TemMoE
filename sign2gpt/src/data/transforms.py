"""Video augmentation using Albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from typing import List


class VideoTransform:
    """Augment video frames with shared params across time."""

    def __init__(
        self,
        training: bool = True,
        img_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        random_shift: int = 4,
        stride: int = 2,
        max_seq_len: int = 256,
        horizontal_flip: bool = False,
        aug_strength: float = 0.2,
    ):
        self.training = training
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.random_shift = random_shift
        self.stride = stride
        self.max_seq_len = max_seq_len

        s = aug_strength
        train_augs = [
            A.Rotate(limit=5, p=0.3),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.875, 1.0), ratio=(0.9, 1.1)),
        ]
        if horizontal_flip:
            train_augs.append(A.HorizontalFlip(p=0.5))
        train_augs.extend([
            A.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s, p=0.3),
            A.ToGray(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        self.train_transform = A.Compose(train_augs)

        self.val_transform = A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __call__(self, frames: List[np.ndarray], is_valid: bool = False) -> torch.Tensor:
        if len(frames) == 0:
            raise ValueError("frames list is empty")

        transform = self.val_transform if is_valid else self.train_transform

        # Same augmentation across all frames
        additional_targets = {f"image{i}": "image" for i in range(1, len(frames))}
        transform.add_targets(additional_targets)

        aug_dict = {"image": frames[0]}
        for i, frame in enumerate(frames[1:], 1):
            aug_dict[f"image{i}"] = frame

        augmented = transform(**aug_dict)
        tensors = [augmented["image"]]
        for i in range(1, len(frames)):
            tensors.append(augmented[f"image{i}"])

        return torch.stack(tensors)
