import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import os
from dataset import RGBNIRDataset
from model import MultimodalSegmentationModel
from validation import validate, get_validation_loader
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
train_dataset = RGBNIRDataset(train_annotations, train_npy_dir, transform=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = get_validation_loader(val_annotations_path, val_npy_dir, batch_size=batch_size)

# Initialize the model
model = MultimodalSegmentationModel(num_classes=2).to(device)

# Set up optimizer and loss function
learning_rate = 1e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Create directory to save logs and model weights
run_dir = create_run_directory()
log_file = os.path.join(run_dir, 'training_log.txt')
model_weights_path = os.path.join(run_dir, 'model_weights.pth')

# Training loop
num_epochs = 40
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

    # Log epoch metrics
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')

# Save the trained model
torch.save(model.state_dict(), model_weights_path)

# Run validation
validate(model, val_loader, device)
