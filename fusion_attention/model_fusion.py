import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet50Backbone(nn.Module):
    def __init__(self, input_channels=3, pretrained_weights="red"):
        super(ResNet50Backbone, self).__init__()

        # Initialize ResNet-50 model with or without ImageNet weights
        if pretrained_weights == "imagenet":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.resnet50(weights=None)

        # Adjust the input layer to fit custom channel count (e.g., RGB or NIR)
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Configure custom weight initialization if required
        if pretrained_weights == "red":
            self.apply_red_weights()
        elif pretrained_weights == "nir_random":
            self.apply_random_weights()

        # Select main feature extraction layers
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
        # Custom weight adjustment focusing on the red channel
        with torch.no_grad():
            # Copy the pretrained weights from the red channel and zero out others
            self.backbone.conv1.weight[:, 1:, :, :] = 0

    def apply_random_weights(self):
        # Initialize weights randomly for specific layers
        self.backbone.apply(self._init_weights)

    def _init_weights(self, layer):
        # Custom initialization method for convolutional and linear layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.feature_extractor(x)


class FusionAttention(nn.Module):
    def __init__(self, channels=2048):
        super(FusionAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, rgb_features, nir_features):
        # Ensure the feature shapes are compatible
        if rgb_features.shape != nir_features.shape:
            raise ValueError("Shapes of RGB and NIR features must match for fusion.")

        # Merge features for attention calculation
        combined = torch.cat((rgb_features, nir_features), dim=1)
        attention_weights = self.attention_layer(combined)
        rgb_attention, nir_attention = torch.chunk(attention_weights, 2, dim=1)

        # Apply attention scaling to both feature maps
        return rgb_attention * rgb_features + nir_attention * nir_features


class SegmentationHead(nn.Module):
    def __init__(self, input_channels=2048, num_classes=2):
        super(SegmentationHead, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_block(x)
        return self.output_layer(x)


class MultimodalSegmentationModel(nn.Module):
    def __init__(self, num_classes=2, pretrained_rgb=True, nir_weights="imagenet"):
        super(MultimodalSegmentationModel, self).__init__()

        # Set up RGB and NIR backbones
        self.rgb_backbone = ResNet50Backbone(
            input_channels=3, pretrained_weights="imagenet" if pretrained_rgb else None
        )
        self.nir_backbone = ResNet50Backbone(
            input_channels=1, pretrained_weights=nir_weights
        )

        # Integrate attention mechanism and segmentation head
        self.fusion_attention = FusionAttention(channels=2048)
        self.segmentation_head = SegmentationHead(
            input_channels=2048, num_classes=num_classes
        )

    def forward(self, fused_image):
        # Split input into RGB and NIR parts
        rgb_image, nir_image = fused_image[:, :3, :, :], fused_image[:, 3:, :, :]

        # Extract features from each branch
        rgb_features = self.rgb_backbone(rgb_image)
        nir_features = self.nir_backbone(nir_image)

        # Fuse features with attention mechanism
        fused_features = self.fusion_attention(rgb_features, nir_features)

        # Final segmentation prediction
        output = self.segmentation_head(fused_features)

        # Resize output to match input dimensions
        return F.interpolate(
            output,
            size=(fused_image.shape[2], fused_image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
