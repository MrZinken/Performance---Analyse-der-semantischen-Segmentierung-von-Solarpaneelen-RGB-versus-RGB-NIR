import torch
import json
import time
import os
from torch.utils.data import DataLoader
from model_rgb import RGBSegmentationModel
from dataset_rgb import RGBDataset
from validation_rgb import validate

"""
Tests all weights in different subfolders
"""
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model
def load_trained_model(model_path, num_classes=2):
    model = RGBSegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Run validation and log metrics
def validate_and_log(model, val_loader, device, metrics_log):
    start_time = time.time()
    metrics = validate(model, val_loader, device, visualize_results=False)
    end_time = time.time()

    total_inference_time = end_time - start_time
    avg_inference_time_per_image = total_inference_time / len(val_loader.dataset)

    # Append metrics to log
    for metric, value in metrics.items():
        metrics_log[metric].append(float(value))
    metrics_log["Total Inference Time"].append(float(total_inference_time))
    metrics_log["Average Inference Time per Image"].append(
        float(avg_inference_time_per_image)
    )


# Log aggregated metrics for all models
def log_final_metrics(metrics_log, log_filepath):
    with open(log_filepath, "w") as log_file:
        log_file.write("Metrics Summary across models:\n")
        for metric, values in metrics_log.items():
            values = [float(v) for v in values]
            log_file.write(f"{metric}: {values}\n")
    print(f"Metrics logged at {log_filepath}")


# Locate all model files in a directory and its subfolders
def find_model_files_in_subfolders(folder_path):
    model_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pth"):
                model_files.append(os.path.join(root, file))
    return model_files


# Process and validate each model in a directory
def process_all_models_in_directory(base_dir, val_annotations_path, val_npy_dir):
    with open(val_annotations_path, "r") as f:
        val_annotations = json.load(f)

    val_dataset = RGBDataset(val_annotations, val_npy_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model_files = find_model_files_in_subfolders(base_dir)
    print(f"Found {len(model_files)} model(s) for validation.")

    # Initialize metric storage
    metrics_log = {
        "IoU": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Validation Loss": [],
        "Total Inference Time": [],
        "Average Inference Time per Image": [],
    }

    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".pth", "")
        print(f"Validating model: {model_name}")

        model = load_trained_model(model_path)
        validate_and_log(model, val_loader, device, metrics_log)
        print(f"Completed validation for model: {model_name}")

    # Save aggregated metrics
    log_filepath = os.path.join(base_dir, "all_models_metrics_log.txt")
    log_final_metrics(metrics_log, log_filepath)


# Execute the script
if __name__ == "__main__":
    base_dir = "runs/3_channel"
    val_annotations_path = "/home/kai/Documents/dataset_300/test/_annotations.coco.json"
    val_npy_dir = "/home/kai/Documents/dataset_300/test"

    process_all_models_in_directory(base_dir, val_annotations_path, val_npy_dir)
