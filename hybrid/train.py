import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import RGBNIRDataset  # Ensure this is your dataset class
from model import HybridSegmentationModel  # Your hybrid model script
import time
import os
import json
from validation import validate  # Import the new validation function

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the dataset and annotation files
train_npy_dir = '/home/kai/Documents/dataset/train'
val_npy_dir = '/home/kai/Documents/dataset/valid'
train_annotations_path = '/home/kai/Documents/dataset/train/_annotations.coco.json'
val_annotations_path = '/home/kai/Documents/dataset/valid/_annotations.coco.json'

# Hyperparameters
num_epochs = 4
batch_size = 4
learning_rate = 1e-4
num_classes = 2

# Load the dataset
with open(train_annotations_path, 'r') as f:
    train_annotations = json.load(f)
with open(val_annotations_path, 'r') as f:
    val_annotations = json.load(f)

from torchvision import transforms

# Define transformations to resize and normalize images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])  # Normalize for 4 channels
])

# Pass the transform to the dataset
train_dataset = RGBNIRDataset(train_annotations, train_npy_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = RGBNIRDataset(val_annotations, val_npy_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = HybridSegmentationModel(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a directory to save model weights
run_dir = 'runs/hybrid_model'
os.makedirs(run_dir, exist_ok=True)

# Training loop
def train_model():
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (fused_image, target) in enumerate(train_loader):
            fused_image, target = fused_image.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(fused_image)

            # Resize target to match the output size (224x224)
            target_resized = F.interpolate(target.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()

            # Compute loss
            loss = criterion(output, target_resized)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s')

        # Validate the model after every epoch
        validate_model()

    total_training_time = time.time() - total_start_time
    print(f"Training completed in {total_training_time:.2f} seconds")

    # Save the final model weights
    model_weights_path = os.path.join(run_dir, 'hybrid_model_final.pth')
    torch.save(model.state_dict(), model_weights_path)
    print(f"Model weights saved at {model_weights_path}")


# Validation function integrated with IoU, Precision, Recall, and F1 Score
def validate_model():
    print("Running validation...")
    validate(model, val_loader, device)  # Call the provided validation function


if __name__ == '__main__':
    train_model()
