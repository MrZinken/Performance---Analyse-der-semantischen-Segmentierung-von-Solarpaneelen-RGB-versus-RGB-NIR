import re
import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing your log files (including subfolders)
base_dir = 'runs/3_channel/'


# Dictionary to store all validation losses for each epoch
all_epoch_losses = {}

# Regex pattern to find 'Validation Loss' lines
loss_pattern = re.compile(r'Validation Loss: (\d+\.\d+)')

# Walk through all subdirectories and process .txt files
for root, dirs, files in os.walk(base_dir):
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(root, filename)
            print(f"Processing file: {file_path}")

            with open(file_path, 'r') as file:
                epoch_num = 1
                for line in file:
                    match = loss_pattern.search(line)
                    if match:
                        loss = float(match.group(1))
                        if epoch_num not in all_epoch_losses:
                            all_epoch_losses[epoch_num] = []
                        all_epoch_losses[epoch_num].append(loss)
                        epoch_num += 1

# Calculate average loss for each epoch
epochs = sorted(all_epoch_losses.keys())
average_losses = [np.mean(all_epoch_losses[epoch]) for epoch in epochs]

# Plot the average validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, average_losses, marker='o', linestyle='-', color='b', label='Average Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Validation Loss')
plt.title('Average Validation Loss per Epoch across all Files')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



