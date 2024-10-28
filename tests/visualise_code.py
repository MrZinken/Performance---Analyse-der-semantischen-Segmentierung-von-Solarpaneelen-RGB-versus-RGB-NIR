import json
import numpy as np
import os
from skimage.draw import polygon
import matplotlib.pyplot as plt

# Main directory containing subfolders with annotations and numpy files
main_dir = '/home/kai/Documents/dataset_150'  # Replace this with your main directory path


# Parameters for analysis
surrounding_width = 20

# Counter to limit to 3 visualizations
image_count = 0

# Iterate through each subfolder
for subfolder in os.listdir(main_dir):
    subfolder_path = os.path.join(main_dir, subfolder)
    
    if not os.path.isdir(subfolder_path):
        continue
    
    # Locate annotation file in the subfolder
    annotation_path = os.path.join(subfolder_path, '_annotations.coco.json')
    if not os.path.exists(annotation_path):
        print(f"No annotation file found in {subfolder_path}. Skipping this folder.")
        continue

    # Load the annotation file
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Loop through each image in the annotations
    for image_info in annotations['images']:
        file_name = image_info['file_name']
        image_path = os.path.join(subfolder_path, file_name)

        # Check if the corresponding numpy file exists
        if not os.path.exists(image_path):
            print(f"NumPy file not found for {file_name} in {subfolder_path}. Skipping this image.")
            continue

        # Load the 4-channel numpy file and normalize
        img_data = np.load(image_path).astype(np.float32)
        img_data /= img_data.max(axis=(0, 1))  # Normalize per channel to [0, 1]

        # Prepare an RGB version of the image for visualization
        rgb_image = img_data[:, :, :3]

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

            # Visualize the original RGB image with overlaid masks
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Display original RGB image
            axs[0].imshow(rgb_image)
            axs[0].set_title("Original RGB Image")
            axs[0].axis('off')

            # Display RGB image with solar panel mask overlay
            axs[1].imshow(rgb_image)
            axs[1].imshow(mask, cmap='Reds', alpha=0.9)  # Red transparent mask
            axs[1].set_title("Solar Panel Mask Overlay")
            axs[1].axis('off')

            # Display RGB image with surrounding mask overlay
            axs[2].imshow(rgb_image)
            axs[2].imshow(surrounding_mask, cmap='Blues', alpha=0.9)  # Blue transparent mask
            axs[2].set_title("Surrounding Mask Overlay")
            axs[2].axis('off')

            plt.show()
            
            # Increment counter and break if 3 images have been displayed
            image_count += 1
            if image_count >= 3:
                break
        if image_count >= 3:
            break
    if image_count >= 3:
        break
