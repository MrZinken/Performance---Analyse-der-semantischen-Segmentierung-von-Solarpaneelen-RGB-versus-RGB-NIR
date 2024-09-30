import torch
from torch.utils.data import DataLoader
import json
from dataset import RGBNIRDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

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

            # Resize the target to the size of the output (224x224)
            target_resized = F.interpolate(target.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()

            output = model(fused_image)
            pred = torch.argmax(output, dim=1)

            # Calculate IoU
            iou = calculate_iou(pred, target_resized)
            ious.append(iou.cpu())  # Move IoU to CPU before appending

            # Visualize original image and prediction
            visualize(fused_image.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy())
    
    avg_iou = np.mean([iou.numpy() for iou in ious])  # Convert to NumPy before computing mean
    print(f'Average IoU: {avg_iou:.4f}')

def calculate_iou(pred, target):
    intersection = (pred & target).float().sum((1, 2))  # Sum of the intersection
    union = (pred | target).float().sum((1, 2))  # Sum of the union
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add epsilon to avoid division by zero
    return iou.mean()

def visualize(fused_image, pred, target):
    # fused_image is in the shape [batch_size, 4, H, W] (batch of images with 4 channels: RGB + NIR)
    batch_size = fused_image.shape[0]  # Get the batch size
    print(f"Fused image shape: {fused_image.shape}")  # For debugging

    # Iterate over each image in the batch
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
            print(f"RGB image shape: {rgb_image.shape}")
        else:
            print(f"Fused image does not have 3 channels for RGB. It has {current_image.shape[0]} channels.")
            continue  # Skip visualization if we can't extract RGB

        # Visualize the image, prediction, and target side by side
        plt.figure(figsize=(12, 4))
        
        # Show RGB image
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image)
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
