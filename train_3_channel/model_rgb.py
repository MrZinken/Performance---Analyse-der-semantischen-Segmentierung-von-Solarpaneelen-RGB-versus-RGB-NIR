import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# ResNet50 Backbone accepting 3-channel input (RGB only)
class ResNet50Backbone(nn.Module):
    def __init__(self, input_channels=3):
        super(ResNet50Backbone, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Modify input to accept 3 channels (RGB only)
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
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

# Complete RGB segmentation model
class RGBSegmentationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(RGBSegmentationModel, self).__init__()
        self.backbone = ResNet50Backbone(input_channels=3)  # RGB input
        self.segmentation_head = SegmentationHead(input_channels=2048, num_classes=num_classes)

    def forward(self, rgb_image):
        features = self.backbone(rgb_image)
        output = self.segmentation_head(features)
        output = F.interpolate(output, size=(rgb_image.shape[2], rgb_image.shape[3]), mode='bilinear', align_corners=False)  # Upsample to original size
        return output
