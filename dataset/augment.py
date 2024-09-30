import numpy as np
import os
import random
import cv2
import json
from scipy.ndimage import rotate

def augment_image(img_array, aug_config):
    # Ensure the input array has 4 channels (RGB + NIR)
    assert img_array.shape[2] == 4, "The input image must have 4 channels (RGB + NIR)"
    
    augmented_images = []
    
    # Separate RGB and NIR channels
    rgb_image = img_array[:, :, :3]
    nir_image = img_array[:, :, 3]

    if random.random() <= aug_config['rotation_prob']:
        angle = random.choice([90, 180, 270])
        rgb_rotated = rotate(rgb_image, angle, axes=(1, 0), reshape=False)
        nir_rotated = rotate(nir_image, angle, axes=(1, 0), reshape=False)
        img_rotated = np.dstack((rgb_rotated, nir_rotated))  # Stack RGB and NIR back together
        augmented_images.append(('rotation', img_rotated, angle))

    if random.random() <= aug_config['flip_prob']:
        img_flipped = np.flipud(rgb_image) if random.choice([True, False]) else np.fliplr(rgb_image)
        nir_flipped = np.flipud(nir_image) if random.choice([True, False]) else np.fliplr(nir_image)
        img_flipped = np.dstack((img_flipped, nir_flipped))  # Stack RGB and NIR back together
        augmented_images.append(('flip', img_flipped, 0))

    if random.random() <= aug_config['blur_prob']:
        ksize = random.choice([3, 5])
        rgb_blurred = cv2.GaussianBlur(rgb_image, (ksize, ksize), 0)
        img_blurred = np.dstack((rgb_blurred, nir_image))  # Blur only the RGB channels
        augmented_images.append(('blur', img_blurred, 0))

    if random.random() <= aug_config['brightness_prob']:
        img_hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        brightness_factor = random.uniform(0.7, 1.3) * aug_config['brightness_percent']
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * brightness_factor, 0, 255)
        rgb_bright = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        img_bright = np.dstack((rgb_bright, nir_image))  # Apply brightness adjustment only to the RGB channels
        augmented_images.append(('brightness', img_bright, 0))

    if random.random() <= aug_config['saturation_prob']:
        img_hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        saturation_factor = random.uniform(0.7, 1.3) * aug_config['saturation_percent']
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation_factor, 0, 255)
        rgb_saturation = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        img_saturation = np.dstack((rgb_saturation, nir_image))  # Adjust saturation on RGB only
        augmented_images.append(('saturation', img_saturation, 0))

    if random.random() <= aug_config['exposure_prob']:
        img_hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        exposure_factor = random.uniform(0.8, 1.2) * aug_config['exposure_percent']
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * exposure_factor, 0, 255)
        rgb_exposure = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        img_exposure = np.dstack((rgb_exposure, nir_image))  # Apply exposure change only to RGB
        augmented_images.append(('exposure', img_exposure, 0))

    return augmented_images


def adjust_filename(file_name, aug_type=None, aug_index=None):
    base_name, ext = os.path.splitext(file_name)
    
    # If augmentation is being applied, append augmentation type and index
    if aug_type is not None and aug_index is not None:
        return f"{base_name}_{aug_type}_aug_{aug_index}{ext}"
    
    return f"{base_name}{ext}"

def rotate_coordinates(coords, angle, img_height, img_width):
    if angle == 90:
        return [(img_height - y, x) for x, y in coords]
    elif angle == 180:
        return [(img_width - x, img_height - y) for x, y in coords]
    elif angle == 270:
        return [(y, img_width - x) for x, y in coords]
    return coords

def rotate_bbox(bbox, angle, img_height, img_width):
    x, y, w, h = bbox
    if angle == 90:
        return [y, img_width - (x + w), h, w]
    elif angle == 180:
        return [img_width - (x + w), img_height - (y + h), w, h]
    elif angle == 270:
        return [img_height - (y + h), x, h, w]
    return bbox

# Augmentation configuration
aug_config = {
    'rotation_prob': 0.3,
    'flip_prob': 0.3,
    'blur_prob': 0.3,
    'brightness_prob': 0.3,
    'brightness_percent': 1.2,
    'saturation_prob': 0.3,
    'saturation_percent': 1.1,
    'exposure_prob': 0.3,
    'exposure_percent': 1.0
}

# Paths
npy_dir = '/home/kai/Documents/dataset/train'
original_coco_file = '/home/kai/Documents/dataset/train/_annotations.coco.json'
augmented_dir = '/home/kai/Documents/augmented'
os.makedirs(augmented_dir, exist_ok=True)

# Load COCO annotations
with open(original_coco_file, 'r') as f:
    coco_data = json.load(f)

new_images = []
new_annotations = []
image_id_offset = len(coco_data['images'])
annotation_id_offset = len(coco_data['annotations'])

# Loop through images
for image in coco_data['images']:
    npy_file = image['file_name']
    file_path = os.path.join(npy_dir, npy_file)
    
    if os.path.exists(file_path):
        img_array = np.load(file_path)

        # Save the original image
        original_save_name = adjust_filename(image['file_name'])
        original_save_path = os.path.join(augmented_dir, original_save_name)
        np.save(original_save_path, img_array)

        new_image_id = image_id_offset + len(new_images) + 1
        new_image_entry = {
            'id': new_image_id,
            'license': image['license'],
            'file_name': original_save_name,
            'height': image['height'],
            'width': image['width'],
            'date_captured': image['date_captured']
        }
        new_images.append(new_image_entry)

        # Copy annotations to the new image
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image['id']:
                new_annotation = annotation.copy()
                new_annotation['image_id'] = new_image_id
                new_annotation_id = annotation_id_offset + len(new_annotations) + 1
                new_annotation['id'] = new_annotation_id
                new_annotations.append(new_annotation)

        # Augment the images
        augmented_images = augment_image(img_array, aug_config)

        for j, (aug_type, augmented_img, angle) in enumerate(augmented_images):
            aug_file_name = adjust_filename(image['file_name'], aug_type, j)
            save_path = os.path.join(augmented_dir, aug_file_name)
            np.save(save_path, augmented_img)

            new_image_id = image_id_offset + len(new_images) + 1
            new_image_entry = {
                'id': new_image_id,
                'license': image['license'],
                'file_name': aug_file_name,
                'height': image['height'],
                'width': image['width'],
                'date_captured': image['date_captured']
            }
            new_images.append(new_image_entry)

            # Process annotations for augmented images
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image['id']:
                    new_annotation_id = annotation_id_offset + len(new_annotations) + 1
                    new_segmentation = []
                    img_width = image['width']
                    img_height = image['height']

                    for segment in annotation['segmentation']:
                        coords = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
                        rotated_coords = rotate_coordinates(coords, angle, img_height, img_width)
                        new_segmentation.append([coord for point in rotated_coords for coord in point])

                    new_bbox = rotate_bbox(annotation['bbox'], angle, img_height, img_width)

                    new_annotation_entry = {
                        'id': new_annotation_id,
                        'image_id': new_image_id,
                        'category_id': annotation['category_id'],
                        'bbox': new_bbox,
                        'area': annotation['area'],
                        'segmentation': new_segmentation,
                        'iscrowd': annotation['iscrowd']
                    }
                    new_annotations.append(new_annotation_entry)

# Update and save COCO annotations
coco_data['images'].extend(new_images)
coco_data['annotations'].extend(new_annotations)

augmented_coco_file = os.path.join(augmented_dir, '_annotations.coco.json')
with open(augmented_coco_file, 'w') as f:
    json.dump(coco_data, f)

print(f"Augmented dataset saved to {augmented_coco_file}")
