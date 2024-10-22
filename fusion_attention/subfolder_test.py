import torch
import json
import os
import time
from torch.utils.data import DataLoader
from cross_attention_model import MultimodalSegmentationModel
from dataset_loader_fusion import RGBNIRDataset
from validation_fusion import validate

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model_with_mapping(model_path, num_classes=2):
    model = MultimodalSegmentationModel(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    
    # Remap keys to align with current model structure
    new_state_dict = {}
    for key, value in state_dict.items():
        if "cross_attention" in key:
            new_key = key.replace("cross_attention", "multi_head_attention")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load state dict, but ignore layers that don't match the current model structure
    model_state_dict = model.state_dict()
    new_state_dict_filtered = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

    # Print out keys that are mismatched and won't be loaded
    missing_keys = set(model_state_dict.keys()) - set(new_state_dict_filtered.keys())
    if missing_keys:
        print(f"Warning: The following keys are missing or mismatched in the loaded state_dict and will be randomly initialized: {missing_keys}")

    model_state_dict.update(new_state_dict_filtered)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model



# Function to validate and log results
def validate_and_log(model, test_loader, device, metrics_log):
    start_time = time.time()
    
    # Run the validation function and collect metrics
    metrics = validate(model, test_loader, device, visualize_results=False)
    
    end_time = time.time()
    total_inference_time = end_time - start_time
    avg_inference_time_per_image = total_inference_time / len(test_loader.dataset)
    
    # Log metrics
    for metric, value in metrics.items():
        metrics_log[metric].append(float(value))
    metrics_log["Total Inference Time"].append(float(total_inference_time))
    metrics_log["Average Inference Time per Image"].append(float(avg_inference_time_per_image))

# Function to log aggregated metrics
def log_final_metrics(metrics_log, log_filepath):
    with open(log_filepath, 'w') as log_file:
        log_file.write("Aggregated Metrics across all models:\n")
        for metric, values in metrics_log.items():
            log_file.write(f"{metric}: {values}\n")
    print(f"All metrics logged to {log_filepath}")

# Function to find all model files in subfolders
def find_model_files_in_subfolders(folder_path):
    model_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    return model_files

# Function to process all models
def process_all_models_in_directory(base_dir, test_annotations_path, test_npy_dir):
    # Load testing annotations and initialize DataLoader
    with open(test_annotations_path, 'r') as f:
        test_annotations = json.load(f)
    
    batch_size = 1
    test_dataset = RGBNIRDataset(test_annotations, test_npy_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Find all model files in subfolders
    model_files = find_model_files_in_subfolders(base_dir)
    print(f"Found {len(model_files)} model(s) to validate.")

    # Initialize metrics log dictionary
    metrics_log = {
        "IoU": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Validation Loss": [],
        "Total Inference Time": [],
        "Average Inference Time per Image": []
    }

    # Validate each model
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.pth', '')
        print(f"Validating model: {model_name}")
        
        # Load model with remapped keys
        model = load_trained_model_with_mapping(model_path)
        
        # Validate and log results
        validate_and_log(model, test_loader, device, metrics_log)
        print(f"Validation complete for model: {model_name}")

    # Log all metrics
    final_log_filepath = os.path.join(base_dir, 'all_models_metrics_log.txt')
    log_final_metrics(metrics_log, final_log_filepath)

# Main function
if __name__ == "__main__":
    base_dir = 'runs/cross_fusion_75_img/'  # Replace with your base directory path
    test_annotations_path = '/home/kai/Documents/dataset_75/test/_annotations.coco.json'
    test_npy_dir = '/home/kai/Documents/dataset_75/test'

    process_all_models_in_directory(base_dir, test_annotations_path, test_npy_dir)
