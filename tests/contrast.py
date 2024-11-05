import json
import numpy as np
import os
from skimage.draw import polygon

"""
calculates the average brightness and stdev of the mask of the annotations of the dataset as well as the sourrounding given by width
"""
# Main directory containing subfolders with annotations and numpy files
main_dir = "/home/kai/Documents/dataset"  # Replace this with your main directory path

# Parameters for analysis
surrounding_width = 20

# Initialize variables for storing overall brightness and homogeneity information across all subfolders
overall_brightness_panel = {"red": [], "green": [], "blue": [], "nir": []}
overall_brightness_surrounding = {"red": [], "green": [], "blue": [], "nir": []}
overall_brightness_diff = {"red": [], "green": [], "blue": [], "nir": []}

overall_homogeneity_panel = {"red": [], "green": [], "blue": [], "nir": []}
overall_homogeneity_surrounding = {"red": [], "green": [], "blue": [], "nir": []}
overall_homogeneity_diff = {"red": [], "green": [], "blue": [], "nir": []}

# Iterate through each subfolder
for subfolder in os.listdir(main_dir):
    subfolder_path = os.path.join(main_dir, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    # Locate annotation file in the subfolder
    annotation_path = os.path.join(subfolder_path, "_annotations.coco.json")
    if not os.path.exists(annotation_path):
        print(f"No annotation file found in {subfolder_path}. Skipping this folder.")
        continue

    # Load the annotation file
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    # Loop through each image in the annotations
    for image_info in annotations["images"]:
        file_name = image_info["file_name"]
        image_path = os.path.join(subfolder_path, file_name)

        # Check if the corresponding numpy file exists
        if not os.path.exists(image_path):
            print(
                f"NumPy file not found for {file_name} in {subfolder_path}. Skipping this image."
            )
            continue

        # Load the 4-channel numpy file and normalize
        img_data = np.load(image_path).astype(np.float32)
        img_data /= img_data.max(axis=(0, 1))  # Normalize per channel to [0, 1]

        # Get annotations for the current image
        image_annotations = [
            ann
            for ann in annotations["annotations"]
            if ann["image_id"] == image_info["id"]
        ]

        # Process each annotated region
        for ann in image_annotations:
            # Create a mask for the solar panel area
            mask = np.zeros(img_data.shape[:2], dtype=bool)
            polygon_coords = np.array(ann["segmentation"][0]).reshape((-1, 2))
            rr, cc = polygon(polygon_coords[:, 1], polygon_coords[:, 0], mask.shape)
            mask[rr, cc] = True

            # Define bounds for surrounding mask based on image boundaries
            panel_coords = np.argwhere(mask)
            min_r, min_c = panel_coords.min(axis=0)
            max_r, max_c = panel_coords.max(axis=0)

            surrounding_mask = np.zeros_like(mask)
            for i in range(-surrounding_width, surrounding_width + 1):
                if min_r + i >= 0 and max_r + i < mask.shape[0]:
                    surrounding_mask = np.logical_or(
                        surrounding_mask, np.roll(mask, i, axis=0)
                    )
                if min_c + i >= 0 and max_c + i < mask.shape[1]:
                    surrounding_mask = np.logical_or(
                        surrounding_mask, np.roll(mask, i, axis=1)
                    )

            # Exclude the panel area from the surrounding mask
            surrounding_mask = np.logical_and(surrounding_mask, ~mask)

            # Calculate metrics only if the surrounding area is within bounds
            if np.any(surrounding_mask):
                for channel, name in enumerate(["red", "green", "blue", "nir"]):
                    # Average brightness for the panel area
                    panel_mean_brightness = np.mean(img_data[mask, channel])
                    overall_brightness_panel[name].append(panel_mean_brightness)

                    # Average brightness for the surrounding area
                    surrounding_mean_brightness = np.mean(
                        img_data[surrounding_mask, channel]
                    )
                    overall_brightness_surrounding[name].append(
                        surrounding_mean_brightness
                    )

                    # Difference in brightness
                    brightness_diff = (
                        panel_mean_brightness - surrounding_mean_brightness
                    )
                    overall_brightness_diff[name].append(brightness_diff)

                    # Homogeneity (StdDev) for the panel area
                    panel_homogeneity = np.std(img_data[mask, channel])
                    overall_homogeneity_panel[name].append(panel_homogeneity)

                    # Homogeneity (StdDev) for the surrounding area
                    surrounding_homogeneity = np.std(
                        img_data[surrounding_mask, channel]
                    )
                    overall_homogeneity_surrounding[name].append(
                        surrounding_homogeneity
                    )

                    # Difference in homogeneity
                    homogeneity_diff = panel_homogeneity - surrounding_homogeneity
                    overall_homogeneity_diff[name].append(homogeneity_diff)

# Calculate and print the overall average brightness and homogeneity values and differences per channel
for name in ["red", "green", "blue", "nir"]:
    avg_panel_brightness = np.mean(overall_brightness_panel[name])
    avg_surrounding_brightness = np.mean(overall_brightness_surrounding[name])
    avg_brightness_diff = np.mean(overall_brightness_diff[name])

    avg_panel_homogeneity = np.mean(overall_homogeneity_panel[name])
    avg_surrounding_homogeneity = np.mean(overall_homogeneity_surrounding[name])
    avg_homogeneity_diff = np.mean(overall_homogeneity_diff[name])

    print(
        f"{name.capitalize()} Channel - Average Panel Brightness: {avg_panel_brightness:.4f}"
    )
    print(
        f"{name.capitalize()} Channel - Average Surrounding Brightness: {avg_surrounding_brightness:.4f}"
    )
    print(
        f"{name.capitalize()} Channel - Average Brightness Difference (Panel - Surrounding): {avg_brightness_diff:.4f}"
    )
    print(
        f"{name.capitalize()} Channel - Average Panel Homogeneity (StdDev): {avg_panel_homogeneity:.4f}"
    )
    print(
        f"{name.capitalize()} Channel - Average Surrounding Homogeneity (StdDev): {avg_surrounding_homogeneity:.4f}"
    )
    print(
        f"{name.capitalize()} Channel - Average Homogeneity Difference (Panel - Surrounding): {avg_homogeneity_diff:.4f}"
    )
    print("-" * 50)
