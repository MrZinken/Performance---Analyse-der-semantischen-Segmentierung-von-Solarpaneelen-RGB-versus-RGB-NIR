import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
import psutil
from dataset import RGBNIRDataset
from train_4_channel.model_4_channel import MultimodalSegmentationModel
from train_4_channel.validation_4_channel import validate, get_validation_loader
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

# Function to log dataset info and weight initialization type
def log_dataset_info(dataset, filepath, init_type):
    with open(filepath, 'a') as f:
        f.write(f"Original dataset size: {len(dataset)}\n")
        f.write(f"No augmentations applied.\n")
        f.write(f"Model initialized with NIR channel using method: {init_type}\n")

# Create a folder for the current run based on date and time
def create_run_directory():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(os.getcwd(), 'runs', '4_channel', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Load training annotations
with open('/home/kai/Documents/dataset/train/_annotations.coco.json', 'r') as f:
    train_annotations = json.load(f)
train_npy_dir = '/home/kai/Documents/dataset/train'

# Load validation annotations
val_annotations_path = '/home/kai/Documents/dataset/valid/_annotations.coco.json'
val_npy_dir = '/home/kai/Documents/dataset/valid'

# Load the dataset
batch_size = 3
train_dataset = RGBNIRDataset(train_annotations, train_npy_dir, transform=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Get validation loader
val_loader = get_validation_loader(val_annotations_path, val_npy_dir, batch_size=batch_size)

# Choose the weight initialization method for the NIR channel
init_type = "red"  # Change this to "pretrained", "red" or "random" as needed

# Create the model with the chosen weight initialization
model = MultimodalSegmentationModel(num_classes=2, nir_init_method=init_type).to(device)

# Set up loss function and optimizer
learning_rate = 1e-5
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a run directory for saving logs and model weights
run_dir = create_run_directory()
log_file = os.path.join(run_dir, 'training_log_multimodal.txt')

# Modify the model weights path to include the initialization method in the name
best_model_weights_path = os.path.join(run_dir, f'best_multimodal_model_weights_{init_type}.pth')

# Log dataset info and initialization type
log_dataset_info(train_dataset, log_file, init_type)

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Timing and memory tracking variables
total_start_time = time.time()

# Training loop
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for fused_image, target in train_loader:
        fused_image = fused_image.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(fused_image)

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
    val_loss = validate(model, val_loader, device, visualize_results=False)

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

# Log final model and training time to the file
with open(log_file, 'a') as f:
    f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
    f.write(f"Best Model Weights saved to {best_model_weights_path}\n")
    f.write(f"Model initialized with NIR channel using method: {init_type}\n")
