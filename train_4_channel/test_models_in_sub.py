import os
import json
import time
import torch
from torch.utils.data import DataLoader
from model_4_channel import MultimodalSegmentationModel
from dataset_4_channel import RGBNIRDataset
from validation_4_channel import validate

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model loading
def load_model(path, num_classes=2):
    model = MultimodalSegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Validation and metrics logging
def evaluate_and_log(model, loader, device, metrics):
    start = time.time()
    results = validate(model, loader, device, visualize_results=False)
    end = time.time()

    total_time = end - start
    avg_time_per_image = total_time / len(loader.dataset)

    for key, value in results.items():
        metrics[key].append(float(value))
    metrics["Total Inference Time"].append(total_time)
    metrics["Average Inference Time per Image"].append(avg_time_per_image)


# Writing results to file
def log_results(metrics, file_path):
    with open(file_path, "w") as file:
        file.write("Aggregate Model Metrics:\n")
        for key, values in metrics.items():
            file.write(f"{key}: {[float(v) for v in values]}\n")


# Find models within folders
def gather_model_files(directory):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith(".pth")
    ]


# Main validation process
def validate_models_in_directory(base_dir, annotations_path, npy_dir):
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    val_loader = DataLoader(
        RGBNIRDataset(annotations, npy_dir), batch_size=4, shuffle=False
    )

    models = gather_model_files(base_dir)
    print(f"Found {len(models)} model(s) for validation.")

    metrics = {
        metric: []
        for metric in [
            "IoU",
            "Precision",
            "Recall",
            "F1 Score",
            "Validation Loss",
            "Total Inference Time",
            "Average Inference Time per Image",
        ]
    }

    for model_path in models:
        print(f"Evaluating {os.path.basename(model_path).replace('.pth', '')}")

        model = load_model(model_path)
        evaluate_and_log(model, val_loader, device, metrics)

        print(f"Completed evaluation for {os.path.basename(model_path)}")

    log_results(metrics, os.path.join(base_dir, "model_metrics_summary.txt"))


# Execution entry point
if __name__ == "__main__":
    base_dir = "runs/4_channel_75_img/"  # Set model directory path
    annotations_path = "/home/kai/Documents/dataset_75/test/_annotations.coco.json"  # Set annotations path
    npy_dir = "/home/kai/Documents/dataset_75/test"  # Set data directory

    validate_models_in_directory(base_dir, annotations_path, npy_dir)
