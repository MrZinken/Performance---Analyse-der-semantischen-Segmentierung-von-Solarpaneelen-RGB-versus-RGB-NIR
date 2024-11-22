import torch
import json
import time
from torch.utils.data import DataLoader
from model_4_channel import MultimodalSegmentationModel
from dataset_4_channel import RGBNIRDataset
from validation_4_channel import evaluate

"""
tests single model weights
"""
# Configure device for model operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load a trained model with specified weights
def load_trained_model(model_path, num_classes=2):
    model = MultimodalSegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


# Path configurations for validation data
val_annotations_path = "/home/kai/Documents/dataset/test/_annotations.coco.json"
val_npy_dir = "/home/kai/Documents/dataset/test"

# Load validation annotations
with open(val_annotations_path, "r") as file:
    val_annotations = json.load(file)

# Initialize validation dataset and data loader
batch_size = 4
val_dataset = RGBNIRDataset(val_annotations, val_npy_dir, transform=None)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Specify path to trained model weights
trained_model_path = (
    "runs/4_channel/2024-10-15_14-51-32/best_multimodal_model_weights_red.pth"
)

# Load trained model for evaluation
model = load_trained_model(trained_model_path)

# Run validation with timing
print("Starting validation...")
start_time = time.time()  # Start timer

# Execute validation
evaluate(model, val_loader, device, visualize_results=True)

# Record end time and calculate timing metrics
end_time = time.time()
total_inference_time = end_time - start_time
avg_inference_time_per_image = total_inference_time / len(val_dataset)

# Display inference timing
print(f"Total inference time: {total_inference_time:.4f} seconds")
print(f"Average inference time per image: {avg_inference_time_per_image:.4f} seconds")
