import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet50Backbone(nn.Module):
    def __init__(self, input_channels: int = 3, pretrained_weights: str = 'red'):
        super(ResNet50Backbone, self).__init__()
        # Load the ResNet-50 model
        if pretrained_weights == 'imagenet':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Modify first convolution layer to match input channels (RGB or NIR)
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Apply custom weights if specified
        if pretrained_weights == 'red':
            # Assuming 'red' means using pretrained weights on a red channel; modify as needed.
            self.apply_red_weights()
        elif pretrained_weights == 'nir_random':
            # Initialize with random weights for NIR
            self.apply_random_weights()
        
        # Define the feature extractor layers
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )

    def apply_red_weights(self):
        # Custom method to initialize weights based on the red channel, modify as needed
        with torch.no_grad():
            # Copy the weights from the red channel (assuming red is the first channel in RGB pretrained weights)
            self.backbone.conv1.weight[:, 0:1, :, :] = self.backbone.conv1.weight[:, 0:1, :, :]
            # Zero out other channels for the red-only pretrained NIR input
            self.backbone.conv1.weight[:, 1:, :, :] = 0

    def apply_random_weights(self):
        # Reinitialize weights with random values
        self.backbone.apply(self._init_weights)
        
    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class FusionAttention(nn.Module):
    def __init__(self, channels: int = 2048):
        super(FusionAttention, self).__init__()
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_features: torch.Tensor, nir_features: torch.Tensor) -> torch.Tensor:
        if rgb_features.shape != nir_features.shape:
            raise ValueError("RGB and NIR features must have the same shape.")
        
        fused = torch.cat((rgb_features, nir_features), dim=1)
        attention_weights = self.attention(fused)
        rgb_attention, nir_attention = torch.chunk(attention_weights, chunks=2, dim=1)
        output = rgb_attention * rgb_features + nir_attention * nir_features
        return output

class SegmentationHead(nn.Module):
    def __init__(self, input_channels: int = 2048, num_classes: int = 2):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

class MultimodalSegmentationModel(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained_rgb: bool = True, nir_weights: str = 'imagenet'):
        super(MultimodalSegmentationModel, self).__init__()
        # Initialize the RGB backbone with ImageNet pretrained weights
        self.rgb_backbone = ResNet50Backbone(input_channels=3, pretrained_weights='imagenet' if pretrained_rgb else None)
        # Initialize the NIR backbone with specified pretrained weights
        self.nir_backbone = ResNet50Backbone(input_channels=1, pretrained_weights=nir_weights)
        
        self.fusion_attention = FusionAttention(channels=2048)
        self.segmentation_head = SegmentationHead(input_channels=2048, num_classes=num_classes)

    def forward(self, fused_image: torch.Tensor) -> torch.Tensor:
        rgb_image = fused_image[:, :3, :, :]  # RGB channels
        nir_image = fused_image[:, 3:, :, :]  # NIR channel
        
        rgb_features = self.rgb_backbone(rgb_image)
        nir_features = self.nir_backbone(nir_image)
        
        fused_features = self.fusion_attention(rgb_features, nir_features)
        output = self.segmentation_head(fused_features)
        
        output = F.interpolate(output, size=(fused_image.shape[2], fused_image.shape[3]), mode='bilinear', align_corners=False)
        
        return output
