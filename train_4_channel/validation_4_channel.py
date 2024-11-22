import torch
from torch.utils.data import DataLoader
import json
from dataset_4_channel import RGBNIRDataset
import matplotlib.pyplot as plt
import numpy as np

"""
Testing with metrics of weights
"""

# Function to load validation data
def prepare_validation_loader(annotation_path, npy_dir, batch_size=4):
    with open(annotation_path, "r") as file:
        annotations = json.load(file)

    dataset = RGBNIRDataset(annotations, npy_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


# Function to evaluate model on validation set
def evaluate(model, loader, device, visualize_results=True):
    model.eval()
    iou_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
    cumulative_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Model prediction
            predictions = model(images)
            predicted_mask = torch.argmax(predictions, dim=1)

            # Calculate metrics
            iou = compute_iou(predicted_mask, labels)
            precision, recall, f1 = compute_metrics(predicted_mask, labels)

            iou_scores.append(iou.cpu().numpy())
            precision_scores.append(precision.cpu().numpy())
            recall_scores.append(recall.cpu().numpy())
            f1_scores.append(f1.cpu().numpy())

            # Calculate validation loss
            loss = loss_fn(predictions, labels)
            cumulative_loss += loss.item()

            # Optional result visualization
            if visualize_results:
                show_results(
                    images.cpu().numpy(),
                    predicted_mask.cpu().numpy(),
                    labels.cpu().numpy(),
                )

    # Compute and display average metrics
    avg_iou = np.mean(iou_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_loss = cumulative_loss / len(loader)

    print(f"IoU: {avg_iou:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")

    return {
        "IoU": avg_iou,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F1 Score": avg_f1,
        "Validation Loss": avg_loss,
    }


# Helper function to compute precision, recall, and F1 score
def compute_metrics(pred, target):
    # Calculate true positives, false positives, and false negatives
    tp = ((pred == 1) & (target == 1)).float().sum((1, 2))
    fp = ((pred == 1) & (target == 0)).float().sum((1, 2))
    fn = ((pred == 0) & (target == 1)).float().sum((1, 2))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision.mean(), recall.mean(), f1.mean()


# Helper function to compute IoU for binary masks
def compute_iou(pred, target):
    pred, target = pred == 1, target == 1
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    return ((intersection + 1e-6) / (union + 1e-6)).mean()


# Visualization function for predictions and ground truth
def show_results(images, predicted_masks, ground_truths):
    batch_size = images.shape[0]

    for idx in range(batch_size):
        image = images[idx]
        pred_mask = predicted_masks[idx]
        true_mask = ground_truths[idx]

        if image.shape[0] >= 3:
            rgb_image = image[:3, :, :]
            rgb_image = np.transpose(rgb_image, (1, 2, 0))

            if rgb_image.max() > rgb_image.min():
                rgb_image = (rgb_image - rgb_image.min()) / (
                    rgb_image.max() - rgb_image.min()
                )
            else:
                rgb_image = np.ones_like(rgb_image) * 0.5
        else:
            print(f"Image lacks RGB channels; has {image.shape[0]} channels.")
            continue

        # Display the images and masks side by side
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(rgb_image)
        plt.title("Input Image (RGB)")

        plt.subplot(1, 4, 2)
        plt.imshow(pred_mask, cmap="coolwarm")
        plt.title("Predicted Mask")

        plt.subplot(1, 4, 3)
        plt.imshow(true_mask, cmap="coolwarm")
        plt.title("Ground Truth Mask")

        plt.show()