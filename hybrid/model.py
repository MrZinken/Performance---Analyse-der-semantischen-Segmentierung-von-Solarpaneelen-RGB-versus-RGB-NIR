import torch
import torch.nn as nn
import timm  # PyTorch Image Models library
import torch.nn.functional as F

# Define the CNN backbone
class CNNBackbone(nn.Module):
    def __init__(self, input_channels=4):
        super(CNNBackbone, self).__init__()
        # A simple CNN architecture
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output: [batch_size, 64, h, w]
        x = self.pool(x)  # Downsample by 2
        x = F.relu(self.conv2(x))  # Output: [batch_size, 128, h/2, w/2]
        x = self.pool(x)  # Downsample by 2
        x = F.relu(self.conv3(x))  # Output: [batch_size, 256, h/4, w/4]
        return x

# Swin Transformer Backbone
class SwinTransformerBackbone(nn.Module):
    def __init__(self, input_channels=4, model_name='swin_base_patch4_window7_224', img_size=224, pretrained=True):
        super(SwinTransformerBackbone, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=input_channels, img_size=img_size, features_only=True)
        self.feature_channels = self.backbone.feature_info.channels()[-1]  # Get output channels from the last feature map

    def forward(self, x):
        features = self.backbone(x)
        return features[-1]  # Return only the last feature map


# Segmentation Head to produce output with the desired number of classes
class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)  # First conv layer
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)  # Output 2 channels for segmentation

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)  # Final output [batch_size, num_classes, height, width]
        return x




class HybridSegmentationModel(nn.Module):
    def __init__(self, num_classes=2, cnn_input_channels=4, swin_input_channels=4, backbone_name='swin_base_patch4_window7_224', pretrained=True):
        super(HybridSegmentationModel, self).__init__()
        # Define the CNN and Swin Transformer backbones
        self.cnn_backbone = CNNBackbone(input_channels=cnn_input_channels)
        self.swin_backbone = SwinTransformerBackbone(input_channels=swin_input_channels, model_name=backbone_name, pretrained=pretrained)

        # Define the segmentation head which takes in both CNN and Swin features
        total_feature_channels = 256 + self.swin_backbone.feature_channels  # Combine CNN (256) + Swin Transformer output channels
        self.segmentation_head = SegmentationHead(input_channels=total_feature_channels, num_classes=num_classes)

    def forward(self, x):
        # Pass the input through both the CNN and Swin Transformer backbones
        cnn_features = self.cnn_backbone(x)  # Output: [batch_size, 256, h/4, w/4]
        swin_features = self.swin_backbone(x)  # Output: [batch_size, swin_feature_channels, different spatial size]

        # Ensure swin_features are in the correct format
        if swin_features.dim() == 3:
            swin_features = swin_features.unsqueeze(0)  # Add batch dimension
        swin_features = swin_features.permute(0, 3, 1, 2)  # [batch_size, height, width, channels] -> [batch_size, channels, height, width]

        # Resize swin_features to match the spatial size of cnn_features
        swin_features_resized = F.interpolate(swin_features, size=(cnn_features.shape[2], cnn_features.shape[3]), mode='bilinear', align_corners=False)

        # Concatenate CNN and Swin Transformer features along the channel dimension
        combined_features = torch.cat([cnn_features, swin_features_resized], dim=1)  # Concatenate along the channel axis

        # Pass the combined features through the segmentation head
        output = self.segmentation_head(combined_features)

        # Upsample the output to match the input size
        output = F.interpolate(output, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return output
