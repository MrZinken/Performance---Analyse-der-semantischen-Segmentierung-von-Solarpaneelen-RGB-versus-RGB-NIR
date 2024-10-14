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
    ious, precisions, recalls, f1s = [], [], [], []

    with torch.no_grad():
        for fused_image, target in val_loader:
            fused_image = fused_image.to(device)
            target = target.to(device)

            # Resize the target to the size of the output (224x224)
            target_resized = F.interpolate(target.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()

            output = model(fused_image)
            pred = torch.argmax(output, dim=1)

            # Calculate metrics
            iou = calculate_iou(pred, target_resized)
            precision, recall, f1_score = calculate_metrics(pred, target_resized)

            # Append metrics to lists
            ious.append(iou.cpu().numpy())
            precisions.append(precision.cpu().numpy())
            recalls.append(recall.cpu().numpy())
            f1s.append(f1_score.cpu().numpy())

            # Visualize the results
            visualize(fused_image.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy())

    # Compute the average of each metric
    avg_iou = np.mean(ious)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    print(f'Average IoU: {avg_iou:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')

# Calculate Precision, Recall, and F1 Score
def calculate_metrics(pred, target):
    tp = ((pred == 1) & (target == 1)).float().sum((1, 2))
    fp = ((pred == 1) & (target == 0)).float().sum((1, 2))
    fn = ((pred == 0) & (target == 1)).float().sum((1, 2))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = torch.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall + 1e-6),
        torch.tensor(0.0)
    )
    
    return precision.mean(), recall.mean(), f1_score.mean()

# Improved IoU calculation (Intersection over Union) for binary masks
def calculate_iou(pred, target):
    pred = pred == 1
    target = target == 1
    
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = torch.where(union > 0, (intersection + 1e-6) / (union + 1e-6), torch.tensor(0.0))
    
    return iou.mean()

# Visualize the image, prediction, and target
def visualize(fused_image, pred, target):
    batch_size = fused_image.shape[0]

    for i in range(batch_size):
        current_image = fused_image[i]
        current_pred = pred[i]
        current_target = target[i]

        if current_image.shape[0] >= 3:
            rgb_image = current_image[:3, :, :]
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        else:
            continue

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image)
        plt.title('Original RGB Image')

        plt.subplot(1, 3, 2)
        plt.imshow(current_pred, cmap='coolwarm')
        plt.title('Predicted Mask')

        plt.subplot(1, 3, 3)
        plt.imshow(current_target, cmap='coolwarm')
        plt.title('Ground Truth Mask')

        plt.show()
