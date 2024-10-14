import torch
import torch.nn as nn
import timm  # PyTorch Image Models library
import torch.nn.functional as F

# Define the CNN Backbone
class CNNBackbone(nn.Module):
    def __init__(self, input_channels=4):
        super(CNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        return x


class SwinTransformerBackbone(nn.Module):
    def __init__(self, input_channels=4, model_name='swin_base_patch4_window7_224', pretrained=True, img_size=1000):
        super(SwinTransformerBackbone, self).__init__()
        # Modify the img_size to match the input size
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            in_chans=input_channels, 
            features_only=True,
            img_size=img_size  # Use the larger input size
        )
        self.feature_channels = self.backbone.feature_info.channels()[-1]

    def forward(self, x):
        features = self.backbone(x)
        return features[-1]  # Use last feature map

# Segmentation Head
class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Hybrid Multimodal Segmentation Model
class HybridSegmentationModel(nn.Module):
    def __init__(self, num_classes=2, cnn_input_channels=4, swin_input_channels=4, backbone_name='swin_base_patch4_window7_224', pretrained=True):
        super(HybridSegmentationModel, self).__init__()
        self.cnn_backbone = CNNBackbone(input_channels=cnn_input_channels)
        self.swin_backbone = SwinTransformerBackbone(input_channels=swin_input_channels, model_name=backbone_name, pretrained=pretrained, img_size=1000)


        # Compute total feature channels for the segmentation head
        cnn_feature_channels = 256
        swin_feature_channels = self.swin_backbone.feature_channels
        total_feature_channels = cnn_feature_channels + swin_feature_channels

        # Initialize segmentation head with the total feature channels
        self.segmentation_head = SegmentationHead(input_channels=288, num_classes=num_classes)


    def forward(self, x):
        cnn_features = self.cnn_backbone(x)
        swin_features = self.swin_backbone(x)

        # Print shapes for debugging
        #print("CNN feature shape:", cnn_features.shape)  # Expecting [batch_size, channels, h/4, w/4]
        #print("Swin feature shape:", swin_features.shape)  # Expecting [batch_size, channels, h/4, w/4]

        # Adaptive pooling to match dimensions
        cnn_features = F.adaptive_avg_pool2d(cnn_features, swin_features.shape[2:])
        
        # Concatenate features along channel dimension
        combined_features = torch.cat([cnn_features, swin_features], dim=1)
        #print("Combined feature shape:", combined_features.shape)  # Expecting [batch_size, total_channels, h/4, w/4]
        
        # Pass through the segmentation head
        output = self.segmentation_head(combined_features)
        
        # Upsample the output to match input image size
        output = F.interpolate(output, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return output

