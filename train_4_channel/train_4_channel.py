import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
import psutil
from dataset_4_channel import RGBNIRDataset
from model_4_channel import MultimodalSegmentationModel
from validation_4_channel import evaluate, prepare_validation_loader
from datetime import datetime

"""
Training of model with auto stopping based on validation loss with 
Training can be repeated for independent runs 
"""

# Select device based on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Log training metrics to file
def save_metrics(
    epoch, batch_size, lr, train_loss, val_loss, duration, memory, total_time, filepath
):
    with open(filepath, "a") as log:
        log.write(f"Epoch {epoch} - Batch Size: {batch_size} - Learning Rate: {lr}\n")
        log.write(
            f"Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}\n"
        )
        log.write(
            f"Epoch Duration: {duration:.2f}s - GPU Memory: {memory / (1024 ** 2):.2f} MB\n"
        )
        log.write(f"Cumulative Time: {total_time:.2f}s\n\n")


# Log initial dataset info and weight initialization method
def record_dataset_info(dataset, log_path, init_method):
    with open(log_path, "a") as log:
        log.write(f"Dataset size: {len(dataset)} - No augmentations applied\n")
        log.write(f"Model initialized with NIR method: {init_method}\n")


# Create a directory for each training session
def setup_run_directory(run_count):
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(os.getcwd(), "runs", f"4_channel_75_img", run_time)
    os.makedirs(path, exist_ok=True)
    return path


# Main loop for repeated training sessions
n_repeats = 10
for count in range(1, n_repeats + 1):
    print(f"Starting run {count} of {n_repeats}")

    # Load train and validation annotation data
    with open("/home/kai/Documents/dataset_75/train/_annotations.coco.json", "r") as f:
        train_data = json.load(f)
    train_path = "/home/kai/Documents/dataset_75/train"

    val_annotations = "/home/kai/Documents/dataset_75/valid/_annotations.coco.json"
    val_path = "/home/kai/Documents/dataset_75/valid"

    # Initialize training data loader
    batch_size = 3
    train_ds = RGBNIRDataset(train_data, train_path)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = prepare_validation_loader(val_annotations, val_path, batch_size=batch_size)
 
    # Select NIR weight initialization type
    nir_init = "red"  # Options: "pretrained", "red", "random"

    # Initialize model
    model = MultimodalSegmentationModel(num_classes=2, nir_init=nir_init).to(
        device
    )

    # Set optimizer and loss function
    lr = 1e-5
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Set up logging
    run_dir = setup_run_directory(count)
    log_file = os.path.join(run_dir, "training_log.txt")
    model_path = os.path.join(run_dir, f"best_model_{nir_init}.pth")

    # Log dataset and initialization information
    record_dataset_info(train_ds, log_file, nir_init)

    # Early stopping configuration
    max_patience = 5
    best_val_loss = float("inf")
    no_improve_epochs = 0

    # Track overall time
    total_start = time.time()

    # Training phase
    epochs = 40
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.time()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            predictions = model(data)

            # Calculate loss and optimize
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        duration = time.time() - start
        total_elapsed = time.time() - total_start
        memory_used = (
            torch.cuda.memory_allocated(device)
            if torch.cuda.is_available()
            else psutil.virtual_memory().used
        )

        # Validate and log performance
        validation = evaluate(model, val_loader, device)
        val_loss = validation["Validation Loss"]

        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {duration:.2f}s - GPU Mem: {memory_used / (1024 ** 2):.2f} MB"
        )

        # Log epoch metrics
        save_metrics(
            epoch + 1,
            batch_size,
            lr,
            avg_loss,
            val_loss,
            duration,
            memory_used,
            total_elapsed,
            log_file,
        )

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_path)
            print(f"Model saved with validation loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(
                f"No improvement in validation loss. Patience: {no_improve_epochs}/{max_patience}"
            )

        if no_improve_epochs >= max_patience:
            print("Early stopping triggered.")
            break

    print(
        f"Total training time for run {count}: {time.time() - total_start:.2f} seconds"
    )

    # Final log entry for each run
    with open(log_file, "a") as log:
        log.write(f"Total Training Duration: {time.time() - total_start:.2f} seconds\n")
        log.write(f"Best model saved at: {model_path}\n")
        log.write(f"NIR weight initialization: {nir_init}\n")

print("All training runs completed.")