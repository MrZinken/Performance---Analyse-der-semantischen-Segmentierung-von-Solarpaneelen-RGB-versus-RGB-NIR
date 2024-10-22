import matplotlib.pyplot as plt

# Data points for each model
fusion_iou = [0.6113, 0.6763, 0.8243]
rgb_iou = [0.4962, 0.6470, 0.8176]
rgbir_iou = [0.5803, 0.7220, 0.8586]

fusion_f1 = [0.7164, 0.7793, 0.8929]
rgb_f1 = [0.5999, 0.7535, 0.8876]
rgbir_f1 = [0.6837, 0.8176, 0.9190]

fusion_training = [685.4490, 958.1370, 1327.9270]
rgb_training = [101.6250, 328.7120, 336.2440]
rgbir_training = [91.6510, 361.5730, 314.4400]

fusion_inference = [0.0318, 0.0325, 0.0330]
rgb_inference = [0.0202, 0.0206, 0.0210]
rgbir_inference = [0.0216, 0.0216, 0.0224]

# X axis: number of images
x_axis = [75, 150, 300]

# Create subplots for IoU, F1, Training time, and Inference time
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot IoU for the three models
axs[0, 0].plot(x_axis, fusion_iou, marker='o', label='Fusion')
axs[0, 0].plot(x_axis, rgb_iou, marker='o', label='RGB')
axs[0, 0].plot(x_axis, rgbir_iou, marker='o', label='RGB + IR')
axs[0, 0].set_xlabel('Number of Images')
axs[0, 0].set_ylabel('IoU')
axs[0, 0].set_title('IoU Comparison')
axs[0, 0].legend()

# Plot F1 scores for the three models
axs[0, 1].plot(x_axis, fusion_f1, marker='o', label='Fusion')
axs[0, 1].plot(x_axis, rgb_f1, marker='o', label='RGB')
axs[0, 1].plot(x_axis, rgbir_f1, marker='o', label='RGB + IR')
axs[0, 1].set_xlabel('Number of Images')
axs[0, 1].set_ylabel('F1 Score')
axs[0, 1].set_title('F1 Score Comparison')
axs[0, 1].legend()

# Plot Training time for the three models
axs[1, 0].plot(x_axis, fusion_training, marker='o', label='Fusion')
axs[1, 0].plot(x_axis, rgb_training, marker='o', label='RGB')
axs[1, 0].plot(x_axis, rgbir_training, marker='o', label='RGB + IR')
axs[1, 0].set_xlabel('Number of Images')
axs[1, 0].set_ylabel('Training Time (s)')
axs[1, 0].set_title('Training Time Comparison')
axs[1, 0].legend()

# Plot Inference time for the three models
axs[1, 1].plot(x_axis, fusion_inference, marker='o', label='Fusion')
axs[1, 1].plot(x_axis, rgb_inference, marker='o', label='RGB')
axs[1, 1].plot(x_axis, rgbir_inference, marker='o', label='RGB + IR')
axs[1, 1].set_xlabel('Number of Images')
axs[1, 1].set_ylabel('Inference Time (s)')
axs[1, 1].set_title('Inference Time Comparison')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
