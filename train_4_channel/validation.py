import torch
from torch.utils.data import DataLoader
import json
from dataset import RGBNIRDataset
import matplotlib.pyplot as plt
import numpy as np

# Load validation annotations
def get_validation_loader(val_annotations_path, val_npy_dir, batch_size=4):
    with open(val_annotations_path, 'r') as f:
        val_annotations = json.load(f)
    
    val_dataset = RGBNIRDataset(val_annotations, val_npy_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader

# Validation function
def validate(model, val_loader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for fused_image, target in val_loader:
            fused_image = fused_image.to(device)
            target = target.to(device)

            output = model(fused_image)
            pred = torch.argmax(output, dim=1)

            # Calculate IoU and visualize
            iou = calculate_iou(pred, target)
            ious.append(iou.cpu().numpy())  # Ensure IoU is moved to the CPU before converting to NumPy

            # Visualize original image, prediction, and IoU visualization
            visualize(fused_image.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy(), pred, target)
    
    avg_iou = np.mean(ious)
    print(f'Average IoU: {avg_iou:.4f}')


# Modified IoU calculation with visualization
def calculate_iou(pred, target):
    intersection = (pred == target).float().sum((1, 2))  # Intersection for label matching
    union = torch.logical_or(pred, target).float().sum((1, 2))  # Union for matching areas
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add epsilon to avoid division by zero

    # Visualize the intersection and union
    visualize_iou(pred, target)

    return iou.mean()

# Visualize the image, prediction, target, and IoU-related areas
def visualize(fused_image, pred, target, pred_tensor, target_tensor):
    batch_size = fused_image.shape[0]

    for i in range(batch_size):
        current_image = fused_image[i]  # Extract the i-th image from the batch
        current_pred = pred[i]
        current_target = target[i]

        # Check if the image has at least 3 channels for RGB visualization
        if current_image.shape[0] >= 3:
            rgb_image = current_image[:3, :, :]  # Extract first 3 channels (RGB)
            rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Transpose to (H, W, C)

            # Rescale the pixel values to the [0, 1] range for proper visualization
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        else:
            print(f"Fused image does not have 3 channels for RGB. It has {current_image.shape[0]} channels.")
            continue  # Skip visualization if we can't extract RGB

        # Visualize the image, prediction, and target side by side
        plt.figure(figsize=(16, 4))
        
        # Show RGB image
        plt.subplot(1, 4, 1)
        plt.imshow(rgb_image)
        plt.title('Original RGB Image')
        
        # Show predicted mask
        plt.subplot(1, 4, 2)
        plt.imshow(current_pred)
        plt.title('Predicted Mask')
        
        # Show ground truth mask
        plt.subplot(1, 4, 3)
        plt.imshow(current_target)
        plt.title('Ground Truth Mask')

        plt.show()

# Visualization for intersection and union
def visualize_iou(pred_tensor, target_tensor):
    intersection = (pred_tensor == target_tensor).float()  # Pixels that match between pred and target
    union = torch.logical_or(pred_tensor, target_tensor).float()  # All relevant pixels in pred or target

    # Convert to numpy for visualization
    intersection = intersection.cpu().numpy()
    union = union.cpu().numpy()

    plt.figure(figsize=(10, 4))

    # Visualize the intersection
    plt.subplot(1, 2, 1)
    plt.imshow(intersection[0], cmap='gray')
    plt.title("Intersection")

    # Visualize the union
    plt.subplot(1, 2, 2)
    plt.imshow(union[0], cmap='gray')
    plt.title("Union")

    plt.show()
