import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
from dataset_loader_fusion import RGBNIRDataset
from cross_attention_model import MultimodalSegmentationModel
from validation_fusion import validate, get_validation_loader
from datetime import datetime
import torch.nn.functional as F

"""
Training of model with auto stopping based on validation loss with 
Training can be repeated for independent runs 
"""
# Define device for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create directory for each training run
def setup_run_directory(run_index):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_path = os.path.join(os.getcwd(), "runs", "cross_fusion_75_img", timestamp)
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


# Load training and validation annotations
with open("/home/kai/Documents/dataset_75/train/_annotations.coco.json", "r") as f:
    train_annotations = json.load(f)

val_annotations_path = "/home/kai/Documents/dataset_75/valid/_annotations.coco.json"
train_npy_dir = "/home/kai/Documents/dataset_75/train"
val_npy_dir = "/home/kai/Documents/dataset_75/valid"

# Define training parameters
batch_size = 3
num_epochs = 100
model_repeats = 10  # Number of models to be trained independently

# Initialize data loaders for training and validation
train_dataset = RGBNIRDataset(train_annotations, train_npy_dir, transform=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = get_validation_loader(
    val_annotations_path, val_npy_dir, batch_size=batch_size
)

# Train multiple models, each with fresh initialization
for run_index in range(1, model_repeats + 1):
    # Initialize a fresh model for each training run
    model = MultimodalSegmentationModel(num_classes=2).to(device)

    # Configure optimizer and loss criterion
    learning_rate = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Setup logging and checkpoint paths
    run_directory = setup_run_directory(run_index)
    log_path = os.path.join(run_directory, "training_log.txt")
    best_model_path = os.path.join(run_directory, "best_model_weights.pth")

    # Configure early stopping
    patience_threshold = 5
    lowest_val_loss = float("inf")
    epochs_without_improvement = 0

    # Track total training time
    overall_start_time = time.time()

    # Log setup details
    with open(log_path, "w") as f:
        f.write(
            f"Training Configuration:\nModel Name: {model.__class__.__name__}\nRun: {run_index}\n"
        )
        f.write(
            f"Batch Size: {batch_size}\nLearning Rate: {learning_rate}\nEpochs: {num_epochs}\nPatience: {patience_threshold}\n\n"
        )
        f.write("Epoch, Loss, Validation Loss, Time (s), GPU Memory (MB)\n")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()

        for fused_image, target in train_loader:
            fused_image, target = fused_image.to(device), target.to(device)

            # Zero gradients, run forward pass, compute loss, backprop, and update weights
            optimizer.zero_grad()
            output = model(fused_image)

            # Adjust target size if needed
            target_resized = (
                F.interpolate(
                    target.unsqueeze(1).float(), size=output.shape[2:], mode="nearest"
                )
                .squeeze(1)
                .long()
            )

            loss = criterion(output, target_resized)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate metrics and timing for this epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        gpu_memory_usage = (
            torch.cuda.memory_allocated(device) / (1024**2)
            if torch.cuda.is_available()
            else 0
        )

        # Validate and compute validation loss
        val_metrics = validate(model, val_loader, device, visualize_results=False)
        val_loss = val_metrics["Validation Loss"]

        # Record epoch results in log file
        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1}, {avg_epoch_loss:.4f}, {val_loss:.4f}, {epoch_duration:.2f}, {gpu_memory_usage:.2f}\n"
            )

        print(
            f"Run {run_index} - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Time: {epoch_duration:.2f}s, GPU Memory: {gpu_memory_usage:.2f} MB"
        )

        # Apply early stopping
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Saving improved model for run {run_index} with validation loss: {lowest_val_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"Validation loss unchanged. Patience counter: {epochs_without_improvement}/{patience_threshold}"
            )

        if epochs_without_improvement >= patience_threshold:
            print("Early stopping triggered.")
            break

    # Log total training time for this run
    total_training_time = time.time() - overall_start_time
    with open(log_path, "a") as f:
        f.write(f"\nTotal Training Time: {total_training_time:.2f}s\n")

    # Save the final model of this run
    final_model_path = os.path.join(run_directory, "final_model_weights.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Run {run_index} completed. Best and final model weights saved.")
