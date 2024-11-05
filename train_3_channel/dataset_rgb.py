import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

"""
loads numpy dataset
"""


class RGBDataset(Dataset):
    def __init__(self, annotations, npy_dir, transform=None):
        self.annotations = annotations
        self.npy_dir = npy_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations["images"])

    def __getitem__(self, idx):
        # Retrieve image data
        image_info = self.annotations["images"][idx]
        img_name = image_info["file_name"]
        img_id = image_info["id"]

        # Load and format RGB image
        npy_path = os.path.join(self.npy_dir, img_name)
        data = np.load(npy_path)
        rgb_image = data[:, :, :3]  # Use only RGB channels

        # Format image as tensor
        rgb_image = torch.tensor(rgb_image).permute(2, 0, 1)  # (C, H, W) format

        # Generate mask based on target annotations
        target = self.get_target(img_id, rgb_image.shape[1:])

        # Apply transformations if defined
        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image.float(), target

    def get_target(self, image_id, img_shape):
        # Retrieve and combine all masks for the current image
        anns = [
            ann
            for ann in self.annotations["annotations"]
            if ann["image_id"] == image_id
        ]
        masks = np.zeros(img_shape, dtype=np.uint8)

        for ann in anns:
            mask = self.create_mask(ann["segmentation"], img_shape)
            masks = np.maximum(masks, mask)  # Combine masks

        return torch.tensor(masks, dtype=torch.long)

    def create_mask(self, segmentation, img_shape):
        # Initialize empty mask and fill with segmentation polygons
        mask = np.zeros(img_shape, dtype=np.uint8)
        for segment in segmentation:
            polygon = np.array(segment).reshape(-1, 2)
            mask = cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask
