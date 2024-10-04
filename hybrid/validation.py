import torch
from torch.utils.data import DataLoader
import json
from dataset import RGBNIRDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Load validation annotations
def get_validation_loader(val_annotations_path, val_npy_dir, batch_size=4):
    with open(val_annotations_path, 'r') as f:
        val_annotations = json.load(f)
    
    val_dataset = RGBNIRDataset(val_annotations, val_npy_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader

# Validation function with IoU, Precision, Recall, and F1 Score
def validate(model, val_loader, device):
    model.eval()
    ious, precisions, recalls, f1s = [], [], [], []
    
    with torch.no_grad():
        for fused_image, target in val_loader:
            fused_image = fused_image.to(device)
            target = target.to(device)

            output = model(fused_image)
            pred = torch.argmax(output, dim=1)

            # Resize target to match the output size (224x224)
            target_resized = F.interpolate(target.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()

            # Calculate IoU, Precision, Recall, F1 Score
            iou = calculate_iou(pred, target_resized)
            precision, recall, f1_score = calculate_metrics(pred, target_resized)

            ious.append(iou.cpu().numpy())
            precisions.append(precision.cpu().numpy())
            recalls.append(recall.cpu().numpy())
            f1s.append(f1_score.cpu().numpy())

            # Visualize original image, prediction, and IoU visualization
            visualize(fused_image.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy(), pred, target_resized)
    
    # Compute the average of each metric
    avg_iou = np.mean(ious)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    print(f'Average IoU: {avg_iou:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')


# Calculate precision, recall, and F1 score
def calculate_metrics(pred, target):
    # True positives (pred = 1, target = 1)
    tp = ((pred == 1) & (target == 1)).float().sum((1, 2))
    
    # False positives (pred = 1, target = 0)
    fp = ((pred == 1) & (target == 0)).float().sum((1, 2))
    
    # False negatives (pred = 0, target = 1)
    fn = ((pred == 0) & (target == 1)).float().sum((1, 2))
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp + 1e-6)
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn + 1e-6)
    
    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision.mean(), recall.mean(), f1_score.mean()

# IoU calculation (Intersection over Union)
def calculate_iou(pred, target):
    intersection = (pred == target).float().sum((1, 2))  # Intersection for label matching
    union = torch.logical_or(pred, target).float().sum((1, 2))  # Union for matching areas
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add epsilon to avoid division by zero

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
            if rgb_image.max() > rgb_image.min():
                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            else:
                # If all pixel values are the same, just make it a flat gray image
                rgb_image = np.zeros_like(rgb_image) + 0.5  # Flat gray image to avoid visualization issues
        else:
            print(f"Fused image does not have 3 channels for RGB. It has {current_image.shape[0]} channels.")
            continue  # Skip visualization if we can't extract RGB

        # Visualize the image, prediction, and target side by side
        plt.figure(figsize=(16, 4))
        
        # Show RGB image
        plt.subplot(1, 4, 1)
        plt.imshow(rgb_image)
        plt.title('Original RGB Image')
        
        # Show predicted mask in color
        plt.subplot(1, 4, 2)
        plt.imshow(current_pred, cmap='coolwarm')  # Use a color map for better visualization
        plt.title('Predicted Mask')
        
        # Show ground truth mask in color
        plt.subplot(1, 4, 3)
        plt.imshow(current_target, cmap='coolwarm')  # Use a color map for better visualization
        plt.title('Ground Truth Mask')

        plt.show()