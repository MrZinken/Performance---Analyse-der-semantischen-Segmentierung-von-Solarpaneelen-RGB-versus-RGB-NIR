import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
import psutil
from dataset_rgb import RGBDataset
from model_rgb import RGBSegmentationModel
from validation_rgb import validate, get_validation_loader
from datetime import datetime

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to log metrics
def log_metrics(
    epoch,
    batch_size,
    learning_rate,
    loss,
    val_loss,
    epoch_duration,
    gpu_memory,
    total_duration,
    filepath,
):
    with open(filepath, "a") as f:
        f.write(f"Epoch {epoch}:\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Epoch Duration: {epoch_duration:.2f} seconds\n")
        f.write(f"GPU Memory Used: {gpu_memory / (1024 ** 2):.2f} MB\n")
        f.write(f"Cumulative Duration: {total_duration:.2f} seconds\n\n")


# Log dataset details
def log_dataset_info(dataset, filepath):
    with open(filepath, "a") as f:
        f.write(f"Dataset size: {len(dataset)}\n")
        f.write("No data augmentations applied.\n")


# Create a unique folder for the current run
def create_run_directory():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(os.getcwd(), "runs", "3_channel_75_img", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# Model training function
def train_model(num_epochs, train_loader, val_loader, learning_rate, batch_size):
    model = RGBSegmentationModel(num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run_dir = create_run_directory()
    log_file = os.path.join(run_dir, "training_log_rgb.txt")
    best_model_weights_path = os.path.join(run_dir, "best_model_weights.pth")
    log_dataset_info(train_loader.dataset, log_file)

    # Early stopping setup
    patience = 5
    best_val_loss = float("inf")
    epochs_no_improve = 0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for rgb_image, target in train_loader:
            rgb_image, target = rgb_image.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(rgb_image)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_duration = time.time() - start_time
        total_duration = time.time() - total_start_time
        gpu_memory = (
            torch.cuda.memory_allocated(device)
            if torch.cuda.is_available()
            else psutil.virtual_memory().used
        )

        # Run validation
        val_metrics = validate(model, val_loader, device, visualize_results=False)
        val_loss = val_metrics["Validation Loss"]

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Duration: {epoch_duration:.2f}s, GPU Memory: {gpu_memory / (1024 ** 2):.2f} MB"
        )
        log_metrics(
            epoch + 1,
            batch_size,
            learning_rate,
            avg_loss,
            val_loss,
            epoch_duration,
            gpu_memory,
            total_duration,
            log_file,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_weights_path)
            print(f"Saved model with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(
                f"No improvement in validation loss. Patience: {epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            print("Early stopping applied.")
            break

    total_training_time = time.time() - total_start_time
    print(f"Total training duration: {total_training_time:.2f} seconds")

    with open(log_file, "a") as f:
        f.write(f"Total Training Duration: {total_training_time:.2f} seconds\n")
        f.write(f"Best Model saved at: {best_model_weights_path}\n")


# Main function to manage multiple training runs
def run_multiple_trainings(num_trainings, num_epochs):
    for i in range(num_trainings):
        print(f"\nInitiating training run {i + 1}/{num_trainings}")

        # Load dataset annotations
        with open(
            "/home/kai/Documents/dataset_75/train/_annotations.coco.json", "r"
        ) as f:
            train_annotations = json.load(f)

        train_npy_dir = "/home/kai/Documents/dataset_75/train"
        val_npy_dir = "/home/kai/Documents/dataset_75/valid"
        batch_size = 3
        train_dataset = RGBDataset(train_annotations, train_npy_dir, transform=None)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = get_validation_loader(
            "/home/kai/Documents/dataset_75/valid/_annotations.coco.json",
            val_npy_dir,
            batch_size=batch_size,
        )

        # Start model training
        train_model(
            num_epochs,
            train_loader,
            val_loader,
            learning_rate=1e-5,
            batch_size=batch_size,
        )


# Set the number of runs and epochs
num_trainings = 10
num_epochs = 100

# Execute the training runs
run_multiple_trainings(num_trainings, num_epochs)
