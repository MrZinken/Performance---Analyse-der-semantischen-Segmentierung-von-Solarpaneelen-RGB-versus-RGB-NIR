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

# Validation function with IoU, Precision, Recall, and F1 Score
def validate(model, val_loader, device):
    model.eval()
    ious, precisions, recalls, f1s = [], [], [], []

    with torch.no_grad():
        for rgb_image, target in val_loader:
            rgb_image = rgb_image.to(device)
            target = target.to(device)

            output = model(rgb_image)
            pred = torch.argmax(output, dim=1)

            # Calculate IoU, Precision, Recall, F1 Score
            iou = calculate_iou(pred, target)
            precision, recall, f1_score = calculate_metrics(pred, target)

            ious.append(iou.cpu().numpy())
            precisions.append(precision.cpu().numpy())
            recalls.append(recall.cpu().numpy())
            f1s.append(f1_score.cpu().numpy())

            # Visualize original image, prediction, and IoU visualization
            visualize(rgb_image.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy())

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
    tp = ((pred == 1) & (target == 1)).float().sum((1, 2))
    fp = ((pred == 1) & (target == 0)).float().sum((1, 2))
    fn = ((pred == 0) & (target == 1)).float().sum((1, 2))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision.mean(), recall.mean(), f1_score.mean()

# IoU calculation (Intersection over Union) for binary masks
def calculate_iou(pred, target):
    pred = pred == 1
    target = target == 1
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean()

# Visualize the image, prediction, and target
def visualize(rgb_image_batch, pred, target):
    batch_size = rgb_image_batch.shape[0]

    for i in range(batch_size):
        current_image = rgb_image_batch[i]
        current_pred = pred[i]
        current_target = target[i]

        if current_image.shape[0] == 3:
            current_rgb_image = np.transpose(current_image, (1, 2, 0))
            current_rgb_image = (current_rgb_image - current_rgb_image.min()) / (current_rgb_image.max() - current_rgb_image.min())
        else:
            print(f"Image does not have 3 channels, it has {current_image.shape[0]} channels.")
            continue

        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(current_rgb_image)
        plt.title('Original RGB Image')

        plt.subplot(1, 3, 2)
        plt.imshow(current_pred, cmap='coolwarm')
        plt.title('Predicted Mask')

        plt.subplot(1, 3, 3)
        plt.imshow(current_target, cmap='coolwarm')
        plt.title('Ground Truth Mask')

        plt.show()
