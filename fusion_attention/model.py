import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet50Backbone(nn.Module):
    def __init__(self, input_channels: int = 3, pretrained: bool = True):
        super(ResNet50Backbone, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
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
    def __init__(self, num_classes: int = 2, pretrained_rgb: bool = True, pretrained_nir: bool = True):
        super(MultimodalSegmentationModel, self).__init__()
        self.rgb_backbone = ResNet50Backbone(input_channels=3, pretrained=pretrained_rgb)
        self.nir_backbone = ResNet50Backbone(input_channels=1, pretrained=pretrained_nir)
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

