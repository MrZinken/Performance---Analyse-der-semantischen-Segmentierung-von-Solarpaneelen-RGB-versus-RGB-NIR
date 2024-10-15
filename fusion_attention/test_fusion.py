import torch
from torch.utils.data import DataLoader
import json
import os
from dataset_loader_fusion import RGBNIRDataset
#from model_fusion import MultimodalSegmentationModel
from cross_attention_model import MultimodalSegmentationModel
from validation_fusion import validate

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best model weights and move to the correct device
def load_best_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)  # Move model to the device after loading weights
    model.eval()
    print(f"Loaded model weights from {weights_path}")
    return model

# Load testing dataset
def get_testing_loader(test_annotations_path, test_npy_dir, batch_size=1):
    with open(test_annotations_path, 'r') as f:
        test_annotations = json.load(f)
    
    test_dataset = RGBNIRDataset(test_annotations, test_npy_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

# Main testing script
if __name__ == "__main__":
    # Paths to the model and testing data
    best_model_weights_path = 'runs/cross_fusion/2024-10-15_14-57-39/best_model_weights.pth'
    test_annotations_path = '/home/kai/Documents/dataset/test/_annotations.coco.json'
    test_npy_dir = '/home/kai/Documents/dataset/test'

    # Load model
    model = MultimodalSegmentationModel(num_classes=2)
    model = load_best_model(model, best_model_weights_path)

    # Load testing data
    test_loader = get_testing_loader(test_annotations_path, test_npy_dir)

    # Evaluate the model on the testing dataset using the validate function
    validate(model, test_loader, device, visualize_results=True)
