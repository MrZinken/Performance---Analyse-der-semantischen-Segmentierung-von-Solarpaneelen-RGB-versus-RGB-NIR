import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
from dataset import RGBNIRDataset
from model_fusion import MultimodalSegmentationModel
from fusion_attention.validation_fusion import validate, get_validation_loader
from datetime import datetime
import torch.nn.functional as F

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to create a directory for saving logs and model weights
def create_run_directory():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(os.getcwd(), 'runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Load training annotations
with open('/home/kai/Documents/dataset/train/_annotations.coco.json', 'r') as f:
    train_annotations = json.load(f)

# Load validation annotations
val_annotations_path = '/home/kai/Documents/dataset/valid/_annotations.coco.json'

# Set the paths to the dataset
train_npy_dir = '/home/kai/Documents/dataset/train'
val_npy_dir = '/home/kai/Documents/dataset/valid'

# Create DataLoader for training and validation
batch_size = 3
num_epochs = 40
train_dataset = RGBNIRDataset(train_annotations, train_npy_dir, transform=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = get_validation_loader(val_annotations_path, val_npy_dir, batch_size=batch_size)

# Initialize the model
model = MultimodalSegmentationModel(num_classes=2).to(device)
model_name = model.__class__.__name__

# Count the number of images in the training dataset
num_train_images = len(train_dataset)

# Set up optimizer and loss function
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Create directory to save logs and model weights
run_dir = create_run_directory()
log_file = os.path.join(run_dir, 'training_log.txt')
best_model_weights_path = os.path.join(run_dir, 'best_model_weights.pth')

# Early stopping parameters
patience = 5  # Stop after 'patience' epochs with no improvement
best_val_loss = float('inf')
epochs_no_improve = 0

# Initialize total training time
total_start_time = time.time()

# Open log file for writing
with open(log_file, 'w') as f:
    f.write("Training Parameters:\n")
    f.write(f"Model Name: {model_name}\n")
    f.write(f"Number of Training Images: {num_train_images}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Total Epochs: {num_epochs}\n")
    f.write(f"Patience: {patience}\n")
    f.write("\nEpoch Details:\n")
    f.write("Epoch, Loss, Validation Loss, Time (s), GPU Memory (MB)\n")

# Training loop

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for fused_image, target in train_loader:
        fused_image, target = fused_image.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(fused_image)

        # Resize the target mask if needed
        target_resized = F.interpolate(target.unsqueeze(1).float(), size=output.shape[2:], mode='nearest').squeeze(1).long()

        # Compute the loss
        loss = criterion(output, target_resized)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    epoch_time = time.time() - start_time

    # Calculate GPU memory usage in MB
    gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0

    # Run validation and calculate validation loss
    val_loss = validate(model, val_loader, device, visualize_results=False)

    # Log epoch details
    with open(log_file, 'a') as f:
        f.write(f"{epoch+1}, {avg_loss:.4f}, {val_loss:.4f}, {epoch_time:.2f}, {gpu_memory:.2f}\n")

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s, GPU Memory: {gpu_memory:.2f} MB')

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save best model weights
        torch.save(model.state_dict(), best_model_weights_path)
        print(f'Saving best model with validation loss: {best_val_loss:.4f}')
    else:
        epochs_no_improve += 1
        print(f'Validation loss did not improve. Patience: {epochs_no_improve}/{patience}')

    # Check if early stopping should be applied
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

# Calculate total training time
total_training_time = time.time() - total_start_time

# Log total training time
with open(log_file, 'a') as f:
    f.write(f"\nTotal Training Time: {total_training_time:.2f}s\n")

# Save final model weights (last epoch's model)
final_model_weights_path = os.path.join(run_dir, 'final_model_weights.pth')
torch.save(model.state_dict(), final_model_weights_path)
print("Training complete. Best model and final model weights are saved.")
