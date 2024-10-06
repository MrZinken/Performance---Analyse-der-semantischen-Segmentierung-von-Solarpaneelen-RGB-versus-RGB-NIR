import torch
from torch.utils.data import DataLoader
import json
from dataset_rgb import RGBDataset
import matplotlib.pyplot as plt
import numpy as np

# Load validation annotations
def get_validation_loader(val_annotations_path, val_npy_dir, batch_size=4):
    with open(val_annotations_path, 'r') as f:
        val_annotations = json.load(f)
    
    val_dataset = RGBDataset(val_annotations, val_npy_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader


def validate(model, val_loader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for fused_image, target in val_loader:
            fused_image = fused_image.to(device)
            target = target.to(device)

            output = model(fused_image)
            pred = torch.argmax(output, dim=1)

            # Calculate IoU
            iou = calculate_iou(pred, target)
            ious.append(iou.cpu().numpy())  # Move IoU to CPU before converting to NumPy
                        # Visualize original image, prediction, and IoU visualization
            visualize(fused_image.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy(), pred, target)

    avg_iou = np.mean(ious)
    print(f'Average IoU: {avg_iou:.4f}')

def calculate_iou(pred, target):
    intersection = (pred == target).float().sum((1, 2))  # Intersection for label matching
    union = torch.logical_or(pred, target).float().sum((1, 2))  # Union for matching areas
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add epsilon to avoid division by zero
    return iou.mean()


def visualize(rgb_image_batch, pred, target, pred_tensor, target_tensor):
    # rgb_image_batch is in the shape [batch_size, 3, H, W] (batch of RGB images)
    batch_size = rgb_image_batch.shape[0]  # Get the batch size
    print(f"RGB image shape: {rgb_image_batch.shape}")  # For debugging

    # Iterate over each image in the batch
    for i in range(batch_size):
        current_image = rgb_image_batch[i]  # Extract the i-th image from the batch
        current_pred = pred[i]
        current_target = target[i]

        # Check if the image has at least 3 channels for RGB visualization
        if current_image.shape[0] == 3:
            # Do not overwrite `rgb_image_batch`; use a different variable
            current_rgb_image = np.transpose(current_image, (1, 2, 0))  # (H, W, C) for RGB
            # Rescale pixel values for proper visualization
            current_rgb_image = (current_rgb_image - current_rgb_image.min()) / (current_rgb_image.max() - current_rgb_image.min())
            print(f"RGB image shape after rescale: {current_rgb_image.shape}")
        else:
            print(f"Image does not have 3 channels, it has {current_image.shape[0]} channels.")
            continue  # Skip visualization if not RGB

        # Visualize the image, prediction, and target side by side
        plt.figure(figsize=(12, 4))

        # Show RGB image
        plt.subplot(1, 3, 1)
        plt.imshow(current_rgb_image)
        plt.title('Original RGB Image')

        # Show predicted mask
        plt.subplot(1, 3, 2)
        plt.imshow(current_pred)
        plt.title('Predicted Mask')

        # Show ground truth mask
        plt.subplot(1, 3, 3)
        plt.imshow(current_target)
        plt.title('Ground Truth Mask')

        plt.show()
