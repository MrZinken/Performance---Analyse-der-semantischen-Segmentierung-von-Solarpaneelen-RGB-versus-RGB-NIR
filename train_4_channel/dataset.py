import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class RGBNIRDataset(Dataset):
    def __init__(self, annotations, npy_dir, transform=None):
        self.annotations = annotations
        self.npy_dir = npy_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        # Extract image information
        image_info = self.annotations['images'][idx]
        img_name = image_info['file_name']
        img_id = image_info['id']
        npy_name = f"{os.path.splitext(img_name)[0]}_rgb_ir.npy"
        
        # Load the NumPy file (assuming 4 channels: RGB + NIR)
        npy_path = os.path.join(self.npy_dir, npy_name)
        data = np.load(npy_path)
        
        # Split the RGB and NIR channels
        rgb_image = data[:, :, :3]  # First three channels are RGB
        nir_image = data[:, :, 3]   # Fourth channel is NIR
        
        # Combine into a 4-channel image (fused input)
        fused_image = torch.cat([torch.tensor(rgb_image).permute(2, 0, 1), torch.tensor(nir_image).unsqueeze(0)], dim=0)
        
        # Get the target mask
        target = self.get_target(img_id, rgb_image.shape[:2])
        
        # Apply transformations (if any)
        if self.transform:
            fused_image = self.transform(fused_image)

        return fused_image.float(), target

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