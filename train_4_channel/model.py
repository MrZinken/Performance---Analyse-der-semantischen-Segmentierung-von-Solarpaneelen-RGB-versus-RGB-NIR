import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# ResNet50 Backbone accepting 4-channel input (RGB + NIR)
class ResNet50Backbone(nn.Module):
    def __init__(self, input_channels=4, nir_init_method="red"):
        """
        Parameters:
        - input_channels: Number of input channels (4 in this case for RGB + NIR)
        - nir_init_method: "red" (use red channel weights), "pretrained" (use mean of RGB), "random" (random initialization)
        """
        super(ResNet50Backbone, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the first conv layer to accept 4 channels instead of 3
        pretrained_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the NIR (4th channel) weights based on the method specified
        with torch.no_grad():
            # Initialize RGB channels with pretrained weights
            new_conv1.weight[:, :3, :, :] = pretrained_conv1.weight  
            
            if nir_init_method == "red":
                # Use red channel weights for NIR
                new_conv1.weight[:, 3, :, :] = pretrained_conv1.weight[:, 0, :, :]
            elif nir_init_method == "pretrained":
                # Initialize NIR channel with the mean of RGB weights
                new_conv1.weight[:, 3:, :, :] = torch.mean(pretrained_conv1.weight, dim=1, keepdim=True)
            elif nir_init_method == "random":
                # Initialize NIR channel randomly
                nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :])
            else:
                raise ValueError("Invalid NIR initialization method. Choose from 'red', 'pretrained', or 'random'.")

        self.backbone.conv1 = new_conv1

        # Keep the rest of the ResNet layers
        self.backbone = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,  # Output from here will be used for segmentation
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

# Segmentation Head to produce 2-channel output (binary segmentation)
class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)  # Output 2 channels for binary segmentation

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)  # Final output [batch_size, 2, height, width]
        return x

# Complete multimodal segmentation model
class MultimodalSegmentationModel(nn.Module):
    def __init__(self, num_classes=2, nir_init_method="red"):
        """
        Parameters:
        - num_classes: Number of segmentation classes
        - nir_init_method: How to initialize the NIR channel ('red', 'pretrained', 'random')
        """
        super(MultimodalSegmentationModel, self).__init__()
        self.backbone = ResNet50Backbone(input_channels=4, nir_init_method=nir_init_method)  # Fused input with 4 channels (RGB + NIR)
        self.segmentation_head = SegmentationHead(input_channels=2048, num_classes=num_classes)

    def forward(self, fused_image):
        features = self.backbone(fused_image)
        output = self.segmentation_head(features)
        output = F.interpolate(output, size=(fused_image.shape[2], fused_image.shape[3]), mode='bilinear', align_corners=False)  # Upsample to original size
        return output
