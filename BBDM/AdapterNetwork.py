import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import CLIPTextModel, CLIPTokenizer

class PretrainedUNetAdapter(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        
        # Create a U-Net with a pre-trained ImageNet encoder
        # segmentation_models_pytorch provides various encoders pre-trained on ImageNet
        self.unet = smp.Unet(
            encoder_name=encoder_name,        # Use ResNet50 backbone (pre-trained on ImageNet)
            encoder_weights=encoder_weights,  # Use ImageNet weights
            in_channels=input_channels,
            classes=64,                       # Intermediate feature dimension
            decoder_attention_type="scse"     # Spatial and channel squeeze & excitation for attention
        )
        
        # Additional layers to process features
        self.feature_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.feature_norm = nn.BatchNorm2d(64)
        self.feature_act = nn.ReLU(inplace=True)
        
        # Cross-attention module for conditioning
        self.cross_attention = CrossAttention(
            query_dim=64,       # Feature dimension from U-Net
            context_dim=64,     # Dimension of conditioning features
            heads=8,            # Number of attention heads
            dim_head=8,         # Dimension per head
            dropout=0.1         # Dropout rate
        )
        
        # Final output layers
        self.final_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.final_norm1 = nn.BatchNorm2d(32)
        self.final_act1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, output_channels, kernel_size=1)
        
    def forward(self, x, x_cond):
        # x is of shape (N, B, C, H, W) where N is the number of predictions
        N, B, C, H, W = x.shape
        
        # Process conditioning input with the same U-Net
        cond_features = self.unet(x_cond)
        cond_features = self.feature_conv(cond_features)
        cond_features = self.feature_norm(cond_features)
        cond_features = self.feature_act(cond_features)
        
        # Initialize output list
        output_list = []
        
        # Process each prediction
        for i in range(N):
            x_sample = x[i]  # (B, C, H, W) for the i-th prediction
            
            # Get features from U-Net
            sample_features = self.unet(x_sample)
            sample_features = self.feature_conv(sample_features)
            sample_features = self.feature_norm(sample_features)
            sample_features = self.feature_act(sample_features)
            
            # Apply cross-attention
            attended_features = self.cross_attention(sample_features, cond_features)
            
            # Final output layers
            output = self.final_conv1(attended_features)
            output = self.final_norm1(output)
            output = self.final_act1(output)
            output = self.final_conv2(output)
            
            output_list.append(output)
        
        # Combine outputs
        output_tensor = x + torch.stack(output_list, dim=0)  # Shape will be (N, B, C, H, W)
        weighted_output = torch.mean(output_tensor, dim=0)   # Average along the N dimension
        
        return weighted_output

class CrossAttention(nn.Module):
    """
    Cross-attention module for conditioning features
    """
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        # Projections for query, key, value
        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(context_dim, inner_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(context_dim, inner_dim, kernel_size=1, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # x: [B, C, H, W] - features to be attended to
        # context: [B, C, H, W] - conditioning features
        context = context if context is not None else x
        
        # Get dimensions
        b, c, h, w = x.shape
        
        # Get query, key, value projections
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for attention computation
        q = q.reshape(b, self.heads, -1, h * w).transpose(-2, -1)  # [b, heads, h*w, dim_head]
        k = k.reshape(b, self.heads, -1, h * w)                   # [b, heads, dim_head, h*w]
        v = v.reshape(b, self.heads, -1, h * w).transpose(-2, -1)  # [b, heads, h*w, dim_head]
        
        # Compute attention scores
        attn = torch.matmul(q, k) * self.scale  # [b, heads, h*w, h*w]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [b, heads, h*w, dim_head]
        
        # Reshape and combine heads
        out = out.transpose(-2, -1).reshape(b, -1, h, w)  # [b, heads*dim_head, h, w]
        
        # Final output projection
        return self.to_out(out)

# class AdapterNetwork(nn.Module):
#     def __init__(self, input_channels=3, output_channels=3, encoder_name="resnet50", encoder_weights="imagenet"):
#         super().__init__()
        
#         # Create the U-Net adapter with pre-trained ImageNet weights
#         self.unet_adapter = PretrainedUNetAdapter(
#             input_channels=input_channels,
#             output_channels=output_channels,
#             encoder_name=encoder_name,
#             encoder_weights=encoder_weights
#         )
        
#     def forward(self, x, x_cond):
#         # x is of shape (N, B, C, H, W) where N is the number of predictions
#         return self.unet_adapter(x, x_cond)

# Alternative implementation with more advanced pre-trained models
class DeepLabV3PlusAdapter(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        
        # Use DeepLabV3+ which has more advanced architecture than U-Net
        # Still using ImageNet pre-trained encoder
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,        # Pre-trained ResNet50 backbone
            encoder_weights=encoder_weights,  # ImageNet weights
            in_channels=input_channels,
            classes=64                        # Intermediate feature dimension
        )
        
        # The rest of the model follows the same structure as PretrainedUNetAdapter
        self.feature_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.feature_norm = nn.BatchNorm2d(64)
        self.feature_act = nn.ReLU(inplace=True)
        
        self.cross_attention = CrossAttention(
            query_dim=64,
            context_dim=64,
            heads=8,
            dim_head=8,
            dropout=0.1
        )
        
        self.final_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.final_norm1 = nn.BatchNorm2d(32)
        self.final_act1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, output_channels, kernel_size=1)
        
    def forward(self, x, x_cond):
        # x is of shape (N, B, C, H, W) where N is the number of predictions
        N, B, C, H, W = x.shape
        
        # Process conditioning input
        cond_features = self.model(x_cond)
        cond_features = self.feature_conv(cond_features)
        cond_features = self.feature_norm(cond_features)
        cond_features = self.feature_act(cond_features)
        
        output_list = []
        
        for i in range(N):
            x_sample = x[i]
            
            sample_features = self.model(x_sample)
            sample_features = self.feature_conv(sample_features)
            sample_features = self.feature_norm(sample_features)
            sample_features = self.feature_act(sample_features)
            
            attended_features = self.cross_attention(sample_features, cond_features)
            
            output = self.final_conv1(attended_features)
            output = self.final_norm1(output)
            output = self.final_act1(output)
            output = self.final_conv2(output)
            
            output_list.append(output)
        
        output_tensor = x + torch.stack(output_list, dim=0)
        weighted_output = torch.mean(output_tensor, dim=0)
        
        return weighted_output

# Usage example
"""
# Install required packages
# !pip install segmentation-models-pytorch

# Available pre-trained encoders (partial list):
#   - resnet18, resnet34, resnet50, resnet101, resnet152 (ImageNet)
#   - efficientnet-b0 through efficientnet-b7 (ImageNet)
#   - vgg16, vgg19 (ImageNet)
#   - densenet121, densenet161, densenet169, densenet201 (ImageNet)

# Create the model
model = AdapterNetwork(
    input_channels=3,
    output_channels=3,
    encoder_name="resnet50",    # Choose backbone architecture
    encoder_weights="imagenet"  # Use ImageNet pre-trained weights
)

# Alternatively, use the DeepLabV3+ adapter
# model = DeepLabV3PlusAdapter(
#     input_channels=3,
#     output_channels=3,
#     encoder_name="resnet50",
#     encoder_weights="imagenet"
# )

# Example forward pass
N, B, C, H, W = 5, 8, 3, 256, 256
x = torch.randn(N, B, C, H, W)
x_cond = torch.randn(B, C, H, W)
output = model(x, x_cond)
print(output.shape)  # Should be [B, C, H, W]
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class AdapterNetwork(nn.Module):
    def __init__(self, input_channels, output_channels=3):
        super().__init__()

        # Initialize EfficientNet backbone (without BatchNorm)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')  # Using EfficientNet-B0
        # self.efficientnet._conv_stem = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.downsize1 = nn.Conv2d(1280, 640, kernel_size=1, padding=0)
        self.downsize2 = nn.Conv2d(1280, 640, kernel_size=1, padding=0)


        # Fully connected layer to bring the output of EfficientNet back to spatial dimensions
        self.fc = nn.Linear(1280*8*8, 256*8*8)  # Output from EfficientNet is 1280-dimensional feature vector
        self.unflatten = nn.Unflatten(1, (256, 8, 8))  # Convert flattened vector to (B, 64, 8, 8)

        # Upsample layer to increase spatial dimensions from 8x8 to 256x256
        self.upsample = nn.ConvTranspose2d(256, 128, kernel_size=6, stride=4, padding=1, output_padding=0)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=6, stride=4, padding=1, output_padding=0)
        self.upsample3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, output_padding=0)

        self.refine1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.refine2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.refine3 = nn.Conv2d(32, output_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x, x_cond):
        # x is of shape (N, B, C, H, W) where N is the number of predictions (5 in this case)
        N, B, C, H, W = x.shape

        # Initialize an empty list to hold the processed outputs for each prediction
        output_list = []

        # Iterate through each prediction in the batch (dimension N)
        for i in range(N):
            x_sample = x[i]  # (B, C, H, W) for the i-th prediction

            # Process the sample with EfficientNet
            x1 = self.efficientnet.extract_features(x_sample)
            x1 = self.downsize1(x1)

            x2 = self.efficientnet.extract_features(x_cond)
            x2 = self.downsize2(x2)

            x_combined = torch.cat((x1, x2), dim=1)
    
   

            # Reshape the output of EfficientNet to match the spatial dimensions (1280, 8, 8)
            x_combined = self.fc(x_combined.view(B, -1))  # Flattened to (B, 256*8*8)
            x_combined = self.unflatten(x_combined)  # Reshape to (B, 256, 8, 8)

            # print('after fc: ', x_combined.shape)

            # Upsample the EfficientNet output to (B, 128, 32, 32)
            x_combined = self.relu(self.upsample(x_combined))
            

            # Upsample the EfficientNet output to (B, 64, 128, 128)
            x_combined = self.relu(self.upsample2(x_combined))
            
            # Upsample the EfficientNet output to (B, 64, 256, 256)
            x_combined = self.relu(self.upsample3(x_combined))
            # print('after upsample: ', x_combined.shape)

            # Refining the output
            x_refined = self.relu(self.refine1(x_combined))
            x_refined = self.relu(self.refine2(x_refined))
            x_refined = self.refine3(x_refined)

            # Append the refined output to the list
            output_list.append(x_refined)

        # Combine the N outputs (stack the predictions together)
        output_tensor = x + torch.stack(output_list, dim=0)  # Shape will be (N, B, C, H, W)

        # Combine the predictions (averaging them across the N dimension)
        weighted_output = torch.mean(output_tensor, dim=0)  # Sum along the N dimension, shape will be (B, C, H, W)

        return weighted_output