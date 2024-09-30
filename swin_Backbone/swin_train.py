import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
import psutil
from dataset import RGBNIRDataset
from swin_model import MultimodalSegmentationModel  # Import Swin model here
from validation import validate, get_validation_loader
from datetime import datetime
from torchvision import transforms
import torch.nn.functional as F

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to log metrics and save to a file
def log_metrics(epoch, batch_size, learning_rate, loss, epoch_time, gpu_mem, total_time, filepath):
    with open(filepath, 'a') as f:
        f.write(f"Epoch {epoch}:\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Loss: {loss:.4f}\n")
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
    run_dir = os.path.join(os.getcwd(), 'runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir



# Define transformations (resize and normalize if needed, but skip ToTensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for Swin Transformer
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])  # Example normalization for 4 channels
])


# Load training annotations
with open('/home/kai/Desktop/np_dataset/train/_annotations.coco.json', 'r') as f:
    train_annotations = json.load(f)

# Load validation annotations
val_annotations_path = '/home/kai/Desktop/np_dataset/valid/_annotations.coco.json'

# Set the paths to the dataset
train_npy_dir = '/home/kai/Desktop/np_dataset/train'
val_npy_dir = '/home/kai/Desktop/np_dataset/valid'

# Load the dataset with the resize transformation
batch_size = 1
train_dataset = RGBNIRDataset(train_annotations, train_npy_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load the validation dataset with the necessary transformations
val_dataset = RGBNIRDataset(val_annotations_path, val_npy_dir, transform=transform)

# Now pass the validation dataset to the DataLoader
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model using Swin Transformer
model = MultimodalSegmentationModel(num_classes=2).to(device)

# Set up loss function and optimizer
learning_rate = 1e-4
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a run directory for saving logs and model weights
run_dir = create_run_directory()
log_file = os.path.join(run_dir, 'training_log_multimodal.txt')
model_weights_path = os.path.join(run_dir, 'multimodal_model_weights.pth')

# Log dataset info
log_dataset_info(train_dataset, log_file)

# Timing and memory tracking variables
total_start_time = time.time()

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()


    for fused_image, target in train_loader:
        fused_image = fused_image.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(fused_image)

        # Resize the target to match the output size
        target_resized = F.interpolate(target.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()

        # Compute loss
        loss = criterion(output, target_resized)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()

    # Calculate time and memory usage
    epoch_time = time.time() - start_time
    total_time = time.time() - total_start_time
    gpu_mem = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else psutil.virtual_memory().used

    # Print metrics for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s, GPU Memory: {gpu_mem / (1024 ** 2):.2f} MB')

    # Log metrics to the file
    log_metrics(epoch+1, batch_size, learning_rate, running_loss/len(train_loader), epoch_time, gpu_mem, total_time, log_file)

    # Run validation after each epoch
    validate(model, val_loader, device)

# Total training time
total_training_time = time.time() - total_start_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Save the final model weights
torch.save(model.state_dict(), model_weights_path)

# Append final training time to the log file
with open(log_file, 'a') as f:
    f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
    f.write(f"Model Weights saved to {model_weights_path}\n")
