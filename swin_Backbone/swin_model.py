import torch
import torch.nn as nn
import timm  # PyTorch Image Models library
import torch.nn.functional as F

class SwinTransformerBackbone(nn.Module):
    def __init__(self, input_channels=4, model_name='swin_base_patch4_window7_224', img_size=224, pretrained=True):
        super(SwinTransformerBackbone, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=input_channels, img_size=img_size, features_only=True)
        self.feature_channels = self.backbone.feature_info.channels()[-1]  # Get output channels from the last feature map

    def forward(self, x):
        features = self.backbone(x)
        # Print the shape of each feature map
        for idx, feature in enumerate(features):
            print(f"Feature {idx} shape: {feature.shape}")
        return features[-1]  # Ensure this is the feature map you want to use



# Segmentation Head to produce 2-channel output (binary segmentation)
class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_classes=2):  # Adjust to match the output of the Swin Transformer
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)  # First conv layer
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)  # Output 2 channels for binary segmentation

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)  # Final output [batch_size, num_classes, height, width]
        return x


# Complete multimodal segmentation model using Swin Transformer backbone
class MultimodalSegmentationModel(nn.Module):
    def __init__(self, num_classes=2, backbone_name='swin_base_patch4_window7_224', pretrained=True):
        super(MultimodalSegmentationModel, self).__init__()
        self.backbone = SwinTransformerBackbone(input_channels=4, model_name=backbone_name, pretrained=pretrained)
        self.segmentation_head = SegmentationHead(input_channels=self.backbone.feature_channels, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        selected_feature_map = features[-1]  # Use the correct feature map with 1024 channels

        # Check if the feature map has the correct number of dimensions
        if selected_feature_map.dim() == 3:  # If the feature map has 3 dimensions
            selected_feature_map = selected_feature_map.unsqueeze(0)  # Add batch dimension

        # Now permute the dimensions to [batch_size, channels, height, width]
        selected_feature_map = selected_feature_map.permute(0, 3, 1, 2)  # [batch_size, height, width, channels] -> [batch_size, channels, height, width]

        output = self.segmentation_head(selected_feature_map)
        output = F.interpolate(output, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return output


