import torch

model_path = "runs/3_channel/2024-10-15_16-06-32/best_model_weights.pth"
checkpoint = torch.load(model_path, map_location='cpu')
print("Saved model keys:", checkpoint.keys())