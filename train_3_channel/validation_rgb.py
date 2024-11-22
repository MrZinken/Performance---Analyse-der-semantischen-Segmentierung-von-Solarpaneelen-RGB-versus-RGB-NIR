import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
from dataset_rgb import RGBDataset

"""
Testing with metrics of weights
"""


# Load validation data
def get_validation_loader(val_annotations_path, val_npy_dir, batch_size=4):
    with open(val_annotations_path, "r") as f:
        val_annotations = json.load(f)
    val_dataset = RGBDataset(val_annotations, val_npy_dir, transform=None)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Validation function to compute key metrics
def validate(model, val_loader, device, visualize_results=True):
    model.eval()
    ious, precisions, recalls, f1s, total_loss = [], [], [], [], 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for fused_image, target in val_loader:
            fused_image, target = fused_image.to(device), target.to(device)
            output = model(fused_image)
            pred = torch.argmax(output, dim=1)

            # Append mean IoU for the batch
            iou = calculate_iou(pred, target).mean().cpu().numpy()
            ious.append(iou)

            # Calculate and append mean precision, recall, F1 score for the batch
            precision, recall, f1_score = calculate_metrics(pred, target)
            precisions.append(precision.mean().cpu().numpy())
            recalls.append(recall.mean().cpu().numpy())
            f1s.append(f1_score.mean().cpu().numpy())

            # Compute loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Visualization
            if visualize_results:
                visualize(
                    fused_image.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy()
                )


    avg_metrics = {
        "IoU": np.mean(ious),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1 Score": np.mean(f1s),
        "Validation Loss": total_loss / len(val_loader),
    }

    for metric, value in avg_metrics.items():
        print(f"Average {metric}: {value:.4f}")
    return avg_metrics


# Helper to calculate precision, recall, and F1
def calculate_metrics(pred, target):
    tp = ((pred == 1) & (target == 1)).float().sum((1, 2))
    fp = ((pred == 1) & (target == 0)).float().sum((1, 2))
    fn = ((pred == 0) & (target == 1)).float().sum((1, 2))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision.mean(), recall.mean(), f1_score.mean()


# IoU computation for binary masks
def calculate_iou(pred, target):
    pred, target = pred == 1, target == 1
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    return (intersection + 1e-6) / (union + 1e-6).mean()


# Visualization function for RGB images, predictions, and targets
def visualize(rgb_image_batch, pred, target):
    batch_size = rgb_image_batch.shape[0]
    for i in range(batch_size):
        image, predicted, truth = rgb_image_batch[i], pred[i], target[i]

        if image.shape[0] == 3:
            rgb_img = np.transpose(image, (1, 2, 0))
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        else:
            print(f"Expected 3 channels, but got {image.shape[0]}")
            continue

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_img)
        plt.title("Original RGB Image")

        plt.subplot(1, 3, 2)
        plt.imshow(predicted, cmap="coolwarm")
        plt.title("Predicted Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(truth, cmap="coolwarm")
        plt.title("Ground Truth Mask")

        plt.show()
