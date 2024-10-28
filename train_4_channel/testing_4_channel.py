import torch
import json
import time
from torch.utils.data import DataLoader
from model_4_channel import MultimodalSegmentationModel  # Import your model
from dataset_4_channel import RGBNIRDataset  # Import your dataset class
from validation_4_channel import validate  # Import validation function

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
val_annotations_path = '/home/kai/Documents/dataset/test/_annotations.coco.json'  # Replace with your path
val_npy_dir = '/home/kai/Documents/dataset/test'  # Replace with your path

# Load the validation annotations
with open(val_annotations_path, 'r') as f:
    val_annotations = json.load(f)

# Initialize the validation dataset and dataloader
batch_size = 4
val_dataset = RGBNIRDataset(val_annotations, val_npy_dir, transform=None)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Path to your trained model weights
trained_model_path = 'runs/4_channel/2024-10-15_14-24-00/best_multimodal_model_weights_red.pth'  # Replace with your path

# Load the trained model
model = load_trained_model(trained_model_path)

# Run the validation script with timing
print("Running validation...")
start_time = time.time()  # Start timing

# Run the validate function
validate(model, val_loader, device, visualize_results=True)

end_time = time.time()  # End timing

# Calculate and print the total and average inference time
total_inference_time = end_time - start_time
average_inference_time_per_image = total_inference_time / len(val_dataset)

print(f"Total inference time: {total_inference_time:.4f} seconds")
print(f"Average inference time per image: {average_inference_time_per_image:.4f} seconds")
