import json
import numpy as np
import os
from skimage.draw import polygon


# Load the annotation file
annotation_path = '/home/kai/Documents/dataset/test/_annotations.coco.json'
with open(annotation_path, 'r') as f:
    annotations = json.load(f)

# Directory containing the numpy array files
data_dir = '/home/kai/Documents/dataset/test'

# Parameters for analysis
surrounding_width = 20

# Initialize variables for storing overall contrast information
contrast_results = {'red': [], 'green': [], 'blue': [], 'nir': []}

# Loop through each image in the annotations
for image_info in annotations['images']:
    file_name = image_info['file_name']
    image_path = os.path.join(data_dir, file_name)

    # Load the 4-channel numpy file and normalize
    img_data = np.load(image_path).astype(np.float32)
    img_data /= img_data.max(axis=(0, 1))  # Normalize per channel to [0, 1]

    # Get annotations for the current image
    image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_info['id']]
    
    # Process each annotated region
    for ann in image_annotations:
        # Create a mask for the solar panel area
        mask = np.zeros(img_data.shape[:2], dtype=bool)
        polygon_coords = np.array(ann['segmentation'][0]).reshape((-1, 2))
        rr, cc = polygon(polygon_coords[:, 1], polygon_coords[:, 0], mask.shape)
        mask[rr, cc] = True

        # Define bounds for surrounding mask based on image boundaries
        panel_coords = np.argwhere(mask)
        min_r, min_c = panel_coords.min(axis=0)
        max_r, max_c = panel_coords.max(axis=0)

        surrounding_mask = np.zeros_like(mask)
        for i in range(-surrounding_width, surrounding_width + 1):
            if min_r + i >= 0 and max_r + i < mask.shape[0]:
                surrounding_mask = np.logical_or(surrounding_mask, np.roll(mask, i, axis=0))
            if min_c + i >= 0 and max_c + i < mask.shape[1]:
                surrounding_mask = np.logical_or(surrounding_mask, np.roll(mask, i, axis=1))

        # Exclude the panel area from the surrounding mask
        surrounding_mask = np.logical_and(surrounding_mask, ~mask)

        # Calculate contrast only if the surrounding area is within bounds
        if np.any(surrounding_mask):
            for channel, name in enumerate(['red', 'green', 'blue', 'nir']):
                panel_std = np.std(img_data[mask, channel])
                surrounding_std = np.std(img_data[surrounding_mask, channel])
                contrast = panel_std - surrounding_std
                contrast_results[name].append(contrast)

# Calculate and print the average contrast per channel
for name in ['red', 'green', 'blue', 'nir']:
    avg_contrast = np.mean(contrast_results[name])
    print(f"{name.capitalize()} Channel - Average Contrast (Panel vs Surrounding StdDev): {avg_contrast:.4f}")

