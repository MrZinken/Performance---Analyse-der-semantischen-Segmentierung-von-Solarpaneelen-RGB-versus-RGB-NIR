import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# ResNet50 Backbone for 4-channel input (RGB + NIR)
class ResNet50Backbone(nn.Module):
    def __init__(self, input_channels=4, nir_init="red"):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Adjust first convolution layer to take 4 channels
        conv1_pretrained = self.backbone.conv1
        conv1_new = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            conv1_new.weight[:, :3, :, :] = (
                conv1_pretrained.weight
            )  # Initialize with RGB weights

            if nir_init == "red":
                conv1_new.weight[:, 3, :, :] = conv1_pretrained.weight[
                    :, 0, :, :
                ]  # Use red weights for NIR
            elif nir_init == "pretrained":
                conv1_new.weight[:, 3:, :, :] = torch.mean(
                    conv1_pretrained.weight, dim=1, keepdim=True
                )
            elif nir_init == "random":
                nn.init.kaiming_normal_(conv1_new.weight[:, 3:, :, :])
            else:
                raise ValueError("nir_init should be 'red', 'pretrained', or 'random'.")

        self.backbone.conv1 = conv1_new
        self.backbone = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )

    def forward(self, x):
        return self.backbone(x)


# Segmentation head
class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)


# Combined segmentation model
class MultimodalSegmentationModel(nn.Module):
    def __init__(self, num_classes=2, nir_init="red"):
        super().__init__()
        self.backbone = ResNet50Backbone(input_channels=4, nir_init=nir_init)
        self.segmentation_head = SegmentationHead(
            input_channels=2048, num_classes=num_classes
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.segmentation_head(features)
        return F.interpolate(
            output, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
        )
