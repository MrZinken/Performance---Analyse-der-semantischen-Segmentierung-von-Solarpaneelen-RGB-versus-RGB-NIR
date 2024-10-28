import torch
import json
import time
from torch.utils.data import DataLoader
from model_rgb import RGBSegmentationModel  # Import your RGB model
from dataset_rgb import RGBDataset  # Import your RGB dataset class
from validation_rgb import validate, get_validation_loader  # Import RGB validation functions

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_trained_model(model_path, num_classes=2):
    model = RGBSegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model weights
    model.to(device)  # Move model to GPU if available
    model.eval()  # Set the model to evaluation mode
    return model

# Load validation annotations
val_annotations_path = '/home/kai/Documents/dataset/test/_annotations.coco.json'  # Replace with your path
val_npy_dir = '/home/kai/Documents/dataset/test'  # Replace with your path

# Load the validation annotations
with open(val_annotations_path, 'r') as f:
    val_annotations = json.load(f)

# Initialize the validation dataset and dataloader for RGB images
batch_size = 4
val_dataset = RGBDataset(val_annotations, val_npy_dir, transform=None)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Path to your trained RGB model weights
trained_model_path = 'runs/3_channel/2024-10-15_15-55-43/best_model_weights.pth'  # Replace with your path

# Load the trained model
model = load_trained_model(trained_model_path)

# Modified validate function to measure inference time
def validate_with_timing(model, val_loader, device, visualize_results=True):
    model.eval()
    total_time = 0.0  # To accumulate the total inference time

    # Iterate over the validation batches
    for i, (images, targets) in enumerate(val_loader):
        images, targets = images.to(device), targets.to(device)

        # Start timing for this batch
        start_time = time.time()

        # Perform inference
        with torch.no_grad():
            outputs = model(images)

        # End timing for this batch
        end_time = time.time()

        # Calculate and accumulate the time taken for this batch
        batch_time = end_time - start_time
        total_time += batch_time
        print(f"Inference time for batch {i+1}/{len(val_loader)}: {batch_time:.4f} seconds")

    # Calculate and print average inference time per batch
    avg_time_per_batch = total_time / len(val_loader)
    print(f"Average inference time per batch: {avg_time_per_batch:.4f} seconds")

    # Optionally, include any other validation metrics or visualization logic as required

# Run the validation with timing
print("Running validation with timing on RGB images...")
#validate_with_timing(model, val_loader, device, visualize_results=True)
validate(model, val_loader, device, visualize_results=True)