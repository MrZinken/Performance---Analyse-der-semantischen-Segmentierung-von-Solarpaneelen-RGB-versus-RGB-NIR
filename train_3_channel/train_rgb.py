import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
import psutil
from dataset_rgb import RGBDataset  # Import the RGB dataset
from model_rgb import RGBSegmentationModel  # Import the RGB segmentation model
from validation_rgb import validate, get_validation_loader  # Validation for RGB
from datetime import datetime

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to log metrics and save to a file
def log_metrics(epoch, batch_size, learning_rate, loss, val_loss, epoch_time, gpu_mem, total_time, filepath):
    with open(filepath, 'a') as f:
        f.write(f"Epoch {epoch}:\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Time Taken for Epoch: {epoch_time:.2f} seconds\n")
        f.write(f"GPU Memory Usage: {gpu_mem / (1024 ** 2):.2f} MB\n")
        f.write(f"Total Time So Far: {total_time:.2f} seconds\n\n")

# Function to log dataset info
def log_dataset_info(dataset, filepath):
    with open(filepath, 'a') as f:
        f.write(f"Original dataset size: {len(dataset)}\n")
        f.write(f"No augmentations applied.\n")

# Create a folder for the current run based on date and time
def create_run_directory():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(os.getcwd(), 'runs', '3_channel_75_img', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Function to train the model and log results
def train_model(num_epochs, train_loader, val_loader, learning_rate, batch_size):
    # Initialize the model
    model = RGBSegmentationModel(num_classes=2).to(device)

    # Set up loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a run directory for saving logs and model weights
    run_dir = create_run_directory()
    log_file = os.path.join(run_dir, 'training_log_rgb.txt')
    best_model_weights_path = os.path.join(run_dir, 'best_model_weights.pth')

    # Log dataset info
    log_dataset_info(train_loader.dataset, log_file)

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Timing and memory tracking variables
    total_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for rgb_image, target in train_loader:
            rgb_image = rgb_image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(rgb_image)

            # Compute loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss for this epoch
        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        total_time = time.time() - total_start_time
        gpu_mem = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else psutil.virtual_memory().used

        # Run validation and calculate validation loss
        val_metrics = validate(model, val_loader, device, visualize_results=False)
        val_loss = val_metrics['Validation Loss']  # Extract validation loss from metrics

        # Print metrics for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s, GPU Memory: {gpu_mem / (1024 ** 2):.2f} MB')

        # Log metrics to the file
        log_metrics(epoch+1, batch_size, learning_rate, avg_loss, val_loss, epoch_time, gpu_mem, total_time, log_file)

        # Early stopping logic and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_weights_path)
            print(f"Saving best model with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Total training time
    total_training_time = time.time() - total_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Log final training time to the file
    with open(log_file, 'a') as f:
        f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
        f.write(f"Best Model Weights saved to {best_model_weights_path}\n")

# Main function to execute multiple trainings
def run_multiple_trainings(num_trainings, num_epochs):
    for i in range(num_trainings):
        print(f"\nStarting training run {i + 1}/{num_trainings}")
        
        # Load training annotations
        with open('/home/kai/Documents/dataset_75/train/_annotations.coco.json', 'r') as f:
            train_annotations = json.load(f)
        
        # Load the dataset
        train_npy_dir = '/home/kai/Documents/dataset_75/train'
        val_npy_dir = '/home/kai/Documents/dataset_75/valid'
        batch_size = 3
        train_dataset = RGBDataset(train_annotations, train_npy_dir, transform=None)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = get_validation_loader('/home/kai/Documents/dataset_75/valid/_annotations.coco.json', val_npy_dir, batch_size=batch_size)
        
        # Train the model
        train_model(num_epochs, train_loader, val_loader, learning_rate=1e-5, batch_size=batch_size)

# Set the number of trainings
num_trainings = 10  # You can set this to the desired number of training runs
num_epochs = 100   # Number of epochs per training run

# Run the trainings
run_multiple_trainings(num_trainings, num_epochs)
