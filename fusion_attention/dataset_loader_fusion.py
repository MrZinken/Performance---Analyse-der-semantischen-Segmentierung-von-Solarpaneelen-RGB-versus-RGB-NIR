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
        
        # Load the NumPy file containing RGB + NIR channels
        npy_path = os.path.join(self.npy_dir, img_name)
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"NumPy file {npy_path} not found.")
        
        data = np.load(npy_path)
        
        # Separate RGB and NIR channels
        rgb_image = data[:, :, :3]  # RGB channels
        nir_image = data[:, :, 3]   # NIR channel

        # Convert both images to tensors and permute to (C, H, W)
        rgb_image = torch.tensor(rgb_image).permute(2, 0, 1)  # (3, H, W)
        nir_image = torch.tensor(nir_image).unsqueeze(0)  # (1, H, W)

        # Stack RGB and NIR into a 4-channel image for the model
        fused_image = torch.cat([rgb_image, nir_image], dim=0)

        # Get the target mask for segmentation
        target = self.get_target(img_id, rgb_image.shape[1:])

        # Apply any transformations
        if self.transform:
            fused_image = self.transform(fused_image)

        return fused_image.float(), target

    def get_target(self, image_id, img_shape):
        # Find the annotations for this image and create the mask
        anns = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
        masks = np.zeros(img_shape, dtype=np.uint8)
        
        for ann in anns:
            mask = self.create_mask(ann['segmentation'], img_shape)
            masks = np.maximum(masks, mask)
        
        return torch.tensor(masks, dtype=torch.long) 

    def create_mask(self, segmentation, img_shape):
        # Create a binary mask for each segmented object
        mask = np.zeros(img_shape, dtype=np.uint8)
        for segment in segmentation:
            polygon = np.array(segment).reshape(-1, 2)
            mask = cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask
