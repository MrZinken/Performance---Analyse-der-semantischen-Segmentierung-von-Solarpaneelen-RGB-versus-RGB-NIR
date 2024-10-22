import torch
import json
import time
import os
from torch.utils.data import DataLoader
from model_rgb import RGBSegmentationModel  # Import your RGB model
from dataset_rgb import RGBDataset  # Import your RGB dataset class
from validation_rgb import validate  # Import RGB validation function

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_trained_model(model_path, num_classes=2):
    model = RGBSegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model weights
    model.to(device)  # Move model to GPU if available
    model.eval()  # Set the model to evaluation mode
    return model

# Function to validate and log results
def validate_and_log(model, val_loader, device, metrics_log):
    # Start timing the validation process
    start_time = time.time()
    
    # Run the validation function (assumed to return metrics as a dictionary)
    metrics = validate(model, val_loader, device, visualize_results=False)
    
    # End timing the validation process
    end_time = time.time()
    total_inference_time = end_time - start_time
    avg_inference_time_per_image = total_inference_time / len(val_loader.dataset)
    
    # Store metrics in the log dictionary for later logging, converting to native Python floats
    for metric, value in metrics.items():
        metrics_log[metric].append(float(value))
    metrics_log["Total Inference Time"].append(float(total_inference_time))
    metrics_log["Average Inference Time per Image"].append(float(avg_inference_time_per_image))

# Function to log the final aggregated metrics across all models
def log_final_metrics(metrics_log, log_filepath):
    with open(log_filepath, 'w') as log_file:
        log_file.write("Aggregated Metrics across all models:\n")
        for metric, values in metrics_log.items():
            values = [float(v) for v in values]  # Ensure no np.float32 types are saved
            log_file.write(f"{metric}: {values}\n")
    print(f"All metrics logged to {log_filepath}")

# Function to find all model files within subfolders of a given directory
def find_model_files_in_subfolders(folder_path):
    model_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pth'):  # Assuming model files are saved with .pth extension
                model_files.append(os.path.join(root, file))
    return model_files

# Main function to process all models in the specified directory
def process_all_models_in_directory(base_dir, val_annotations_path, val_npy_dir):
    # Load validation annotations and initialize the dataset and dataloader
    with open(val_annotations_path, 'r') as f:
        val_annotations = json.load(f)
    
    batch_size = 4
    val_dataset = RGBDataset(val_annotations, val_npy_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Find all model files in subfolders
    model_files = find_model_files_in_subfolders(base_dir)
    print(f"Found {len(model_files)} model(s) to validate.")

    # Initialize a dictionary to store metric arrays
    metrics_log = {
        "IoU": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Validation Loss": [],
        "Total Inference Time": [],
        "Average Inference Time per Image": []
    }

    # Validate each model and accumulate metrics
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.pth', '')
        print(f"Validating model: {model_name}")
        
        # Load the model
        model = load_trained_model(model_path)
        
        # Run validation and accumulate results
        validate_and_log(model, val_loader, device, metrics_log)
        print(f"Validation complete for model: {model_name}")

    # Log all metrics to the log file in the base directory
    final_log_filepath = os.path.join(base_dir, 'all_models_metrics_log.txt')
    log_final_metrics(metrics_log, final_log_filepath)

# Run the script with your paths
if __name__ == "__main__":
    base_dir = 'runs/3_channel_75_img'  # Replace with the path to your models directory
    val_annotations_path = '/home/kai/Documents/dataset_75/test/_annotations.coco.json'  # Replace with your path
    val_npy_dir = '/home/kai/Documents/dataset_75/test'  # Replace with your path

    process_all_models_in_directory(base_dir, val_annotations_path, val_npy_dir)
