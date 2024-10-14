import torch
import json
import os
from torch.utils.data import DataLoader
from model import MultimodalSegmentationModel  # Import your model
from dataset import RGBNIRDataset  # Import your dataset class
from validation import validate, get_validation_loader  # Import validation functions

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_trained_model(model_path, num_classes=2):
    model = MultimodalSegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model weights
    model.to(device)  # Move model to GPU if available
    model.eval()  # Set the model to evaluation mode
    return model

# Load validation annotations
val_annotations_path = '/home/kai/Documents/dataset_np/valid/_annotations.coco.json'  # Replace with your path
val_npy_dir = '/home/kai/Documents/dataset_np/valid'  # Replace with your path

# Load the validation annotations
with open(val_annotations_path, 'r') as f:
    val_annotations = json.load(f)

# Initialize the validation dataset and dataloader
batch_size = 4
val_dataset = RGBNIRDataset(val_annotations, val_npy_dir, transform=None)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Path to your trained model weights
trained_model_path = 'runs/2024-09-19_21-48-10/multimodal_model_weights.pth'  # Replace with your path

# Load the trained model
model = load_trained_model(trained_model_path)

# Run the validation script
print("Running validation...")
validate(model, val_loader, device)
