import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

import json

class RGBNIRDataset(Dataset):
    def __init__(self, annotations, npy_dir, transform=None):
        # Ensure annotations is a dictionary
        if isinstance(annotations, str):  # If it's a string, treat it as a file path
            with open(annotations, 'r') as f:
                self.annotations = json.load(f)
        else:  # Otherwise, assume it's already a dictionary
            self.annotations = annotations

        self.npy_dir = npy_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['images'])



    def __getitem__(self, idx):
        # Extract image information
        image_info = self.annotations['images'][idx]
        img_name = image_info['file_name']  # This is the exact file name in the annotation
        img_id = image_info['id']
        
        # Load the NumPy file using the exact file name from the annotation
        npy_path = os.path.join(self.npy_dir, img_name)
        
        # Check if the file exists
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"NumPy file {npy_path} not found.")
        
        # Load the NumPy file (assuming it has 4 channels: RGB + NIR)
        data = np.load(npy_path)
        
        # Split the RGB and NIR channels
        rgb_image = data[:, :, :3]  # First three channels are RGB
        nir_image = data[:, :, 3]   # Fourth channel is NIR
        
        # Combine into a 4-channel image (fused input)
        fused_image = torch.cat([torch.tensor(rgb_image).permute(2, 0, 1), torch.tensor(nir_image).unsqueeze(0)], dim=0)
        
        # Convert the fused image to float and scale to [0, 1]
        fused_image = fused_image.float() / 255.0
        
        # Get the target mask
        target = self.get_target(img_id, rgb_image.shape[:2])
        
        # Apply transformations (if any), skipping ToTensor as it's already a tensor
        if self.transform:
            fused_image = self.transform(fused_image)  # This should handle resizing, normalization, etc.

        return fused_image, target



    def get_target(self, image_id, img_shape):
        # Retrieve annotations (masks) for the current image
        anns = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
        masks = np.zeros(img_shape, dtype=np.uint8)
        
        for ann in anns:
            # Create mask from annotations
            mask = self.create_mask(ann['segmentation'], img_shape)
            masks = np.maximum(masks, mask)  # Merge all masks
        
        return torch.tensor(masks, dtype=torch.long)  # Target should be long tensor

    def create_mask(self, segmentation, img_shape):
        # Create an empty mask
        mask = np.zeros(img_shape, dtype=np.uint8)
        for segment in segmentation:
            polygon = np.array(segment).reshape(-1, 2)
            mask = cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask
    

