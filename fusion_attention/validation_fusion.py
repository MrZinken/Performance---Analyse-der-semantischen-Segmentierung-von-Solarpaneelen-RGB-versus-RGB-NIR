import torch
from torch.utils.data import DataLoader
import json
from dataset_loader_fusion import RGBNIRDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Load validation annotations
def get_validation_loader(val_annotations_path, val_npy_dir, batch_size=4):
    with open(val_annotations_path, 'r') as f:
        val_annotations = json.load(f)
    
    val_dataset = RGBNIRDataset(val_annotations, val_npy_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader

def validate(model, data_loader, device, visualize_results=False):
    model.eval()
    ious, precisions, recalls, f1s = [], [], [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for fused_image, target in tqdm(data_loader):
            fused_image = fused_image.to(device)
            target = target.to(device)

            # Forward pass
            output = model(fused_image)
            pred = torch.argmax(output, dim=1)

            # Calculate IoU, Precision, Recall, F1 Score
            iou = calculate_iou(pred, target)
            precision, recall, f1_score = calculate_metrics(pred, target)

            # Accumulate metrics
            ious.append(iou.cpu().numpy())
            precisions.append(precision.cpu().numpy())
            recalls.append(recall.cpu().numpy())
            f1s.append(f1_score.cpu().numpy())

            # Calculate and accumulate validation loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Optionally visualize results
            if visualize_results:
                visualize(fused_image.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy())

    # Compute the average metrics
    avg_iou = np.mean(ious)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    avg_loss = total_loss / len(data_loader)

    # Print metrics
    print(f'Average IoU: {avg_iou:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')
    print(f'Average Validation Loss: {avg_loss:.4f}')

    # Return metrics as a dictionary
    return {
        'IoU': avg_iou,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1 Score': avg_f1,
        'Validation Loss': avg_loss
    }




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

# Improved IoU calculation (Intersection over Union) for binary masks
def calculate_iou(pred, target):
    # Convert the prediction and target to boolean tensors (only consider class 1 for IoU)
    pred = pred == 1
    target = target == 1
    
    # Calculate intersection and union
    intersection = (pred & target).float().sum((1, 2))  # True positives
    union = (pred | target).float().sum((1, 2))  # Union of positives

    # Calculate IoU with added epsilon to avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean()



# Visualize function without unused arguments
def visualize(fused_image, pred, target):
    batch_size = fused_image.shape[0]

    for i in range(batch_size):
        current_image = fused_image[i]
        current_pred = pred[i]
        current_target = target[i]

        # Extract RGB for visualization
        if current_image.shape[0] >= 3:
            rgb_image = current_image[:3, :, :]
            rgb_image = np.transpose(rgb_image, (1, 2, 0))

            if rgb_image.max() > rgb_image.min():
                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            else:
                rgb_image = np.zeros_like(rgb_image) + 0.5
        else:
            print(f"Fused image does not have 3 channels for RGB. It has {current_image.shape[0]} channels.")
            continue

        # Display the images
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(rgb_image)
        plt.title('Original RGB Image')
        plt.subplot(1, 4, 2)
        plt.imshow(current_pred, cmap='coolwarm')
        plt.title('Predicted Mask')
        plt.subplot(1, 4, 3)
        plt.imshow(current_target, cmap='coolwarm')
        plt.title('Ground Truth Mask')
        plt.show()


