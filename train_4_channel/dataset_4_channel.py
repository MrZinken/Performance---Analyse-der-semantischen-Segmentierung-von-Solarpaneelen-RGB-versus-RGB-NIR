import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

"""
loads numpy dataset
"""


class RGBNIRDataset(Dataset):
    def __init__(self, annotations, npy_dir, transform=None):
        self.annotations = annotations
        self.npy_dir = npy_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations["images"])

    def __getitem__(self, idx):
        # Get image information
        img_info = self.annotations["images"][idx]
        img_name = img_info["file_name"]
        img_id = img_info["id"]

        # Load image data from numpy file
        npy_path = os.path.join(self.npy_dir, img_name)
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"{npy_path} is missing.")

        # Load numpy data and split channels
        data = np.load(npy_path)
        rgb_data = data[:, :, :3]
        nir_data = data[:, :, 3]

        # Create fused 4-channel tensor
        fused_image = torch.cat(
            [
                torch.tensor(rgb_data).permute(2, 0, 1),
                torch.tensor(nir_data).unsqueeze(0),
            ],
            dim=0,
        )

        # Load target mask
        target = self._get_mask(img_id, rgb_data.shape[:2])

        # Apply any transformations
        if self.transform:
            fused_image = self.transform(fused_image)

        return fused_image.float(), target

    def _get_mask(self, image_id, img_shape):
        # Generate mask from annotations
        anns = [
            ann
            for ann in self.annotations["annotations"]
            if ann["image_id"] == image_id
        ]
        mask = np.zeros(img_shape, dtype=np.uint8)

        for ann in anns:
            mask = np.maximum(mask, self._draw_polygon(ann["segmentation"], img_shape))

        return torch.tensor(mask, dtype=torch.long)

    def _draw_polygon(self, segmentation, img_shape):
        # Draw polygons onto mask
        mask = np.zeros(img_shape, dtype=np.uint8)
        for segment in segmentation:
            polygon = np.array(segment).reshape(-1, 2)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask
