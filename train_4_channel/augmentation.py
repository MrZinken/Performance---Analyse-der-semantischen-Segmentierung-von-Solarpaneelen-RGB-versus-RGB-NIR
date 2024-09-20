import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from torchvision.transforms import ColorJitter
from scipy.ndimage import gaussian_filter

class Augmentations:
    def __init__(self, rotate_probs, flip_probs, brightness_prob=0.5, brightness_factor=0.2,
                 exposure_prob=0.5, exposure_factor=0.2, saturation_prob=0.5, saturation_factor=0.2,
                 blur_prob=0.5, blur_sigma=1, visualize=False):
        # Rotation probabilities for 90, 180, and 270 degrees
        self.rotate_probs = rotate_probs
        # Flip probabilities (horizontal and vertical)
        self.flip_probs = flip_probs
        # Brightness, exposure, saturation augmentation
        self.color_jitter = ColorJitter(brightness=brightness_factor, contrast=exposure_factor, saturation=saturation_factor)
        self.brightness_prob = brightness_prob
        self.exposure_prob = exposure_prob
        self.saturation_prob = saturation_prob
        # Blur augmentation
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        # Visualization flag
        self.visualize = visualize

    def rotate(self, image, mask):
        """Randomly rotate the image and mask by 90, 180, or 270 degrees."""
        if random.random() < self.rotate_probs[0]:
            return np.rot90(image, 1), np.rot90(mask, 1), '90 degrees'
        elif random.random() < self.rotate_probs[1]:
            return np.rot90(image, 2), np.rot90(mask, 2), '180 degrees'
        elif random.random() < self.rotate_probs[2]:
            return np.rot90(image, 3), np.rot90(mask, 3), '270 degrees'
        return image, mask, 'No rotation'

    def flip(self, image, mask):
        """Randomly flip the image and mask horizontally or vertically."""
        flip_type = ''
        if random.random() < self.flip_probs[0]:
            image, mask, flip_type = np.fliplr(image), np.fliplr(mask), 'Horizontal flip'
        if random.random() < self.flip_probs[1]:
            image, mask, flip_type = np.flipud(image), np.flipud(mask), 'Vertical flip'
        return image, mask, flip_type

    def apply_color_jitter(self, image):
        """Apply brightness, exposure, and saturation augmentation."""
        if random.random() < self.brightness_prob:
            image = np.array(self.color_jitter(image))
            return image, 'Color jitter applied'
        return image, 'No color jitter'

    def apply_blur(self, image):
        """Apply Gaussian blur to the image."""
        if random.random() < self.blur_prob:
            return gaussian_filter(image, sigma=self.blur_sigma), 'Blur applied'
        return image, 'No blur'

    def augment(self, image, mask):
        """Apply the series of augmentations."""
        original_image, original_mask = image.copy(), mask.copy()

        # Rotation and visualize
        image, mask, rotate_info = self.rotate(image, mask)
        if self.visualize:
            self.visualize_augmentation_step(original_image, original_mask, image, mask, f'Rotation: {rotate_info}')
        
        # Flip and visualize
        original_image, original_mask = image.copy(), mask.copy()
        image, mask, flip_info = self.flip(image, mask)
        if self.visualize:
            self.visualize_augmentation_step(original_image, original_mask, image, mask, f'Flip: {flip_info}')
        
        # Color jitter and visualize
        original_image = image.copy()
        image, jitter_info = self.apply_color_jitter(image)
        if self.visualize:
            self.visualize_augmentation_step(original_image, original_mask, image, mask, f'Color Jitter: {jitter_info}')
        
        # Blur and visualize
        original_image = image.copy()
        image, blur_info = self.apply_blur(image)
        if self.visualize:
            self.visualize_augmentation_step(original_image, original_mask, image, mask, f'Blur: {blur_info}')

        return image, mask

    def visualize_augmentation_step(self, original_image, original_mask, augmented_image, augmented_mask, augmentation_name):
        """Visualize the step of augmentation."""
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title(f'Original Image ({augmentation_name})')
        ax[0, 1].imshow(original_mask, cmap='gray')
        ax[0, 1].set_title('Original Mask')

        ax[1, 0].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        ax[1, 0].set_title(f'Augmented Image ({augmentation_name})')
        ax[1, 1].imshow(augmented_mask, cmap='gray')
        ax[1, 1].set_title('Augmented Mask')

        plt.show()

# Example of how to use the Augmentations class
if __name__ == "__main__":
    # Define probabilities for rotations and flips
    rotate_probs = [0.25, 0.25, 0.25]  # Rotate by 90, 180, 270 degrees with equal probability
    flip_probs = [0.5, 0.5]  # 50% chance for horizontal and vertical flip

    # Create the augmentation object with probabilities and visualize flag
    augmentor = Augmentations(rotate_probs=rotate_probs, flip_probs=flip_probs, 
                              brightness_prob=0.5, exposure_prob=0.5, saturation_prob=0.5,
                              blur_prob=0.3, blur_sigma=1.5, visualize=True)

    # Load a sample image and mask for testing
    sample_image = cv2.imread('sample_image.jpg')  # Example RGB image
    sample_mask = np.load('sample_mask.npy')  # Corresponding mask

    # Apply augmentations
    augmented_image, augmented_mask = augmentor.augment(sample_image, sample_mask)
