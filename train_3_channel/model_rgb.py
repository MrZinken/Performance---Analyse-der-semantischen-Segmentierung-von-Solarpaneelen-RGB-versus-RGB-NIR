import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

"""
RGB-Modell
"""


# Backbone of the model based on ResNet-50, adapted for 3-channel RGB input
class ResNet50Backbone(nn.Module):
    def __init__(self, input_channels=3):
        super(ResNet50Backbone, self).__init__()

        # Load pretrained ResNet-50 with ImageNet weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Adjust the first convolutional layer to accommodate 3 input channels (RGB)
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Keep ResNet layers up to the final feature layer
        self.backbone_layers = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,  # Extracted features
        )

    def forward(self, x):
        return self.backbone_layers(x)


# Segmentation Head to produce binary segmentation output
class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)  # Final output with specified number of classes


# Full segmentation model combining ResNet-50 backbone and segmentation head
class RGBSegmentationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(RGBSegmentationModel, self).__init__()
        self.backbone = ResNet50Backbone(input_channels=3)
        self.segmentation_head = SegmentationHead(
            input_channels=2048, num_classes=num_classes
        )

    def forward(self, rgb_image):
        features = self.backbone(rgb_image)
        output = self.segmentation_head(features)

        # Upsample to match input dimensions
        return F.interpolate(
            output,
            size=(rgb_image.shape[2], rgb_image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
