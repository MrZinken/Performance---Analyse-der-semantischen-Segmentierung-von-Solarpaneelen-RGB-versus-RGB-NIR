import torch
import json
import time
from torch.utils.data import DataLoader
from model_rgb import RGBSegmentationModel
from dataset_rgb import RGBDataset
from validation_rgb import validate, get_validation_loader

"""
tests single model weights
"""
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pre-trained model
def load_trained_model(model_path, num_classes=2):
    model = RGBSegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Load validation data
val_annotations_path = "/home/kai/Documents/dataset/test/_annotations.coco.json"
val_npy_dir = "/home/kai/Documents/dataset/test"

# Read validation annotations
with open(val_annotations_path, "r") as f:
    val_annotations = json.load(f)

# Set up validation dataset and dataloader for RGB images
batch_size = 4
val_dataset = RGBDataset(val_annotations, val_npy_dir)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load model weights
model_path = "runs/3_channel/2024-10-15_16-06-32/best_model_weights.pth"
model = load_trained_model(model_path)


# Enhanced validation to track timing
def validate_with_timing(model, val_loader, device):
    model.eval()
    total_inference_time = 0.0

    for i, (images, targets) in enumerate(val_loader):
        images, targets = images.to(device), targets.to(device)

        # Time inference for each batch
        start_time = time.time()
        with torch.no_grad():
            outputs = model(images)
        end_time = time.time()

        # Track time taken for each batch
        batch_time = end_time - start_time
        total_inference_time += batch_time
        print(f"Batch {i+1}/{len(val_loader)} inference time: {batch_time:.4f}s")

    # Compute average inference time per batch
    avg_time_per_batch = total_inference_time / len(val_loader)
    print(f"Average inference time per batch: {avg_time_per_batch:.4f}s")


# Run validation with timing
print("Starting validation with timing on RGB images...")
#validate_with_timing(model, val_loader, device)
validate(model, val_loader, device, visualize_results=True)
