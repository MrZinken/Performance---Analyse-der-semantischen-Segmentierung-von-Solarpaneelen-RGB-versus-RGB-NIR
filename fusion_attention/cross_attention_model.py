import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

"""
Fusion Modell with dual Resnet50 Backbone
"""


class ResNet50Backbone(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        pretrained_weights: str = "imagenet",
        nir_init_method: str = "red",
    ):
        super(ResNet50Backbone, self).__init__()

        # Load pretrained model
        if pretrained_weights == "imagenet":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.resnet50(weights=None)

        # Modify the first convolutional layer to accommodate 4 input channels (RGB + NIR)
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.bn1 = nn.BatchNorm2d(64)

        # Initialize NIR channel based on the specified method
        if input_channels == 4 and nir_init_method == "red":
            self.initialize_nir_with_red()

        # Include the full ResNet with batch normalization layers
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

    def initialize_nir_with_red(self):
        with torch.no_grad():
            self.backbone.conv1.weight.data[:, 3:4, :, :] = (
                self.backbone.conv1.weight.data[:, 0:1, :, :]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


# Multi-Head Self-Attention for richer feature interaction in CrossAttention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.head_dim = channels // num_heads

    def forward(
        self, rgb_features: torch.Tensor, nir_features: torch.Tensor
    ) -> torch.Tensor:
        batch_size, channels, height, width = rgb_features.size()

        # Reshape and split for multi-head attention
        query = self.query_conv(rgb_features).view(
            batch_size, self.num_heads, self.head_dim, -1
        )
        key = self.key_conv(nir_features).view(
            batch_size, self.num_heads, self.head_dim, -1
        )
        value = self.value_conv(nir_features).view(
            batch_size, self.num_heads, self.head_dim, -1
        )

        # Compute attention weights
        attention_scores = torch.matmul(query.permute(0, 1, 3, 2), key) / (
            self.head_dim**0.5
        )
        attention_weights = self.softmax(attention_scores)

        out = torch.matmul(attention_weights, value.permute(0, 1, 3, 2))
        out = (
            out.permute(0, 1, 3, 2)
            .contiguous()
            .view(batch_size, channels, height, width)
        )

        # Add residual connection and return
        out = self.gamma * out + rgb_features
        return out


class FusionAttention(nn.Module):
    def __init__(self, channels: int = 2048, fusion_type: str = "multihead_attention"):
        super(FusionAttention, self).__init__()
        self.channels = channels
        self.fusion_type = fusion_type
        self.multi_head_attention = MultiHeadSelfAttention(channels)

        if fusion_type == "multihead_attention":
            self.attention = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 2, kernel_size=1),
                nn.Softmax(dim=1),
            )
        elif fusion_type == "late_fusion":
            self.late_fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(
        self, rgb_features: torch.Tensor, nir_features: torch.Tensor
    ) -> torch.Tensor:
        if self.fusion_type == "multihead_attention":
            return self.multi_head_attention(rgb_features, nir_features)
        elif self.fusion_type == "late_fusion":
            fused = torch.cat((rgb_features, nir_features), dim=1)
            return self.late_fusion_conv(fused)
        else:
            raise ValueError(
                "Invalid fusion type. Choose 'multihead_attention' or 'late_fusion'."
            )


class SegmentationHead(nn.Module):
    def __init__(self, input_channels: int = 2048, num_classes: int = 2):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)


class MultimodalSegmentationModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        nir_init_method: str = "red",
        fusion_type: str = "multihead_attention",
    ):
        super(MultimodalSegmentationModel, self).__init__()
        # Initialize RGB and NIR backbones
        self.rgb_backbone = ResNet50Backbone(
            input_channels=3, pretrained_weights="imagenet"
        )
        self.nir_backbone = ResNet50Backbone(
            input_channels=1, pretrained_weights=None, nir_init_method=nir_init_method
        )

        # Fusion with Multi-Head Self-Attention
        self.fusion_attention = FusionAttention(channels=2048, fusion_type=fusion_type)
        self.segmentation_head = SegmentationHead(
            input_channels=2048, num_classes=num_classes
        )

    def forward(self, fused_image: torch.Tensor) -> torch.Tensor:
        rgb_image = fused_image[:, :3, :, :]  # RGB channels
        nir_image = fused_image[:, 3:, :, :]  # NIR channel

        rgb_features = self.rgb_backbone(rgb_image)
        nir_features = self.nir_backbone(nir_image)

        fused_features = self.fusion_attention(rgb_features, nir_features)
        output = self.segmentation_head(fused_features)
        return F.interpolate(
            output,
            size=(fused_image.shape[2], fused_image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
