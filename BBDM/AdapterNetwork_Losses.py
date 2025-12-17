# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import lpips  # pip install lpips

# class MultiScaleGradientLoss(nn.Module):
#     """
#     Multi-scale gradient loss for preserving edges and details in microscopy images.
#     Computes gradient differences at multiple scales to ensure feature preservation
#     across different levels of detail.
#     """
#     def __init__(self, scales=(1, 2, 4, 8), weights=None):
#         """
#         Args:
#             scales (tuple): Scales at which to compute gradients
#             weights (list, optional): Weight for each scale. If None, all scales weighted equally.
#         """
#         super(MultiScaleGradientLoss, self).__init__()
#         self.scales = scales
#         if weights is None:
#             self.weights = [1.0/len(scales)] * len(scales)
#         else:
#             assert len(weights) == len(scales), "Weights must match number of scales"
#             self.weights = weights
        
#         # Create Sobel filters for x and y directions without registering as buffers
#         # We'll move these to the appropriate device in the forward pass
#         self.sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
#         self.sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    
#     def compute_gradient_magnitude(self, x):
#         """Compute gradient magnitude using Sobel filters"""
#         batch_size, channels, height, width = x.shape
        
#         # Move Sobel filters to the same device as input
#         sobel_x = self.sobel_x.to(x.device)
#         sobel_y = self.sobel_y.to(x.device)
        
#         # Compute gradients for each channel separately
#         grad_x = torch.zeros_like(x)
#         grad_y = torch.zeros_like(x)
        
#         for c in range(channels):
#             # Extract channel and add batch dimension for conv2d
#             channel = x[:, c:c+1, :, :]
#             # Apply Sobel filters
#             grad_x[:, c:c+1, :, :] = F.conv2d(F.pad(channel, [1, 1, 1, 1], mode='reflect'), sobel_x)
#             grad_y[:, c:c+1, :, :] = F.conv2d(F.pad(channel, [1, 1, 1, 1], mode='reflect'), sobel_y)
            
#         # Compute gradient magnitude
#         grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
#         return grad_magnitude
    
#     def forward(self, pred, target):
#         """
#         Compute multi-scale gradient loss between predicted and target images
        
#         Args:
#             pred (torch.Tensor): Predicted images [B, C, H, W]
#             target (torch.Tensor): Target images [B, C, H, W]
            
#         Returns:
#             torch.Tensor: Scalar loss value
#         """
#         total_loss = 0
        
#         for scale_idx, scale in enumerate(self.scales):
#             # Downsample images if scale > 1
#             if scale > 1:
#                 curr_pred = F.interpolate(pred, scale_factor=1/scale, mode='bilinear', align_corners=False)
#                 curr_target = F.interpolate(target, scale_factor=1/scale, mode='bilinear', align_corners=False)
#             else:
#                 curr_pred = pred
#                 curr_target = target
            
#             # Compute gradient magnitudes
#             pred_grad = self.compute_gradient_magnitude(curr_pred)
#             target_grad = self.compute_gradient_magnitude(curr_target)
            
#             # Compute L1 loss between gradients
#             scale_loss = F.l1_loss(pred_grad, target_grad)
            
#             # Weight the loss by scale importance
#             total_loss += self.weights[scale_idx] * scale_loss
            
#         return total_loss


# # Example of how to use this loss in training:
# def create_combined_microscopy_loss(lpips_net='vgg'):
#     """
#     Creates a combined loss function with MSE, SSIM, multi-scale gradient loss,
#     and LPIPS for brightfield microscopy image enhancement
    
#     Args:
#         lpips_net (str): Network to use for LPIPS ('alex', 'vgg', or 'squeeze')
#     """
#     # Import SSIM if available
#     ssim_available = False
#     try:
#         from pytorch_msssim import SSIM
#         ssim_available = True
#     except ImportError:
#         print("pytorch_msssim not found. Using MSE, gradient, and LPIPS loss only.")
    
#     # Create component losses
#     mse_loss = nn.MSELoss()
#     gradient_loss = MultiScaleGradientLoss(scales=(1, 2, 4), weights=[0.5, 0.3, 0.2])
    
#     # Create LPIPS loss - using the official implementation
#     lpips_loss = lpips.LPIPS(net=lpips_net, spatial=False)  # Set spatial=True for per-pixel loss map
    
#     if ssim_available:
#         ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
        
#         def combined_loss(pred, target):
#             # Make sure both inputs are on the same device
#             if pred.device != target.device:
#                 target = target.to(pred.device)
                
#             # Ensure LPIPS is on the same device
#             if next(lpips_loss.parameters()).device != pred.device:
#                 lpips_loss.to(pred.device)
            
#             # Normalize images to [-1, 1] for LPIPS if needed
#             if pred.min() >= 0 and pred.max() <= 1:
#                 lpips_pred = pred * 2 - 1
#                 lpips_target = target * 2 - 1
#             else:
#                 lpips_pred = pred
#                 lpips_target = target
            
#             # Handle single channel images for LPIPS
#             if lpips_pred.shape[1] == 1:
#                 lpips_pred = lpips_pred.repeat(1, 3, 1, 1)
#                 lpips_target = lpips_target.repeat(1, 3, 1, 1)
            
#             # Compute individual losses
#             mse = mse_loss(pred, target)
#             ssim_value = 1.0 - ssim_module(pred, target)  # Convert to loss (1-SSIM)
#             grad_loss = gradient_loss(pred, target)
#             percept_loss = lpips_loss(lpips_pred, lpips_target).mean()
            
#             # Combine with weights - higher weight on LPIPS for brightfield microscopy
#             return 0.25 * mse + 0.2 * ssim_value + 0.2 * grad_loss + 0.35 * percept_loss
#     else:
#         def combined_loss(pred, target):
#             # Make sure both inputs are on the same device
#             if pred.device != target.device:
#                 target = target.to(pred.device)
                
#             # Ensure LPIPS is on the same device
#             if next(lpips_loss.parameters()).device != pred.device:
#                 lpips_loss.to(pred.device)
            
#             # Normalize images to [-1, 1] for LPIPS if needed
#             if pred.min() >= 0 and pred.max() <= 1:
#                 lpips_pred = pred * 2 - 1
#                 lpips_target = target * 2 - 1
#             else:
#                 lpips_pred = pred
#                 lpips_target = target
                
#             # Handle single channel images for LPIPS
#             if lpips_pred.shape[1] == 1:
#                 lpips_pred = lpips_pred.repeat(1, 3, 1, 1)
#                 lpips_target = lpips_target.repeat(1, 3, 1, 1)
            
#             # Compute individual losses
#             mse = mse_loss(pred, target)
#             grad_loss = gradient_loss(pred, target)
#             percept_loss = lpips_loss(lpips_pred, lpips_target).mean()
            
#             # Combine with weights - higher weight on LPIPS for brightfield microscopy
#             return 0.35 * mse + 0.2 * grad_loss + 0.45 * percept_loss
    
#     return combined_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleGradientLoss(nn.Module):
    """
    Multi-scale gradient loss for preserving edges and details in microscopy images.
    Computes gradient differences at multiple scales to ensure feature preservation
    across different levels of detail.
    """
    def __init__(self, scales=(1, 2, 4, 8), weights=None):
        """
        Args:
            scales (tuple): Scales at which to compute gradients
            weights (list, optional): Weight for each scale. If None, all scales weighted equally.
        """
        super(MultiScaleGradientLoss, self).__init__()
        self.scales = scales
        if weights is None:
            self.weights = [1.0/len(scales)] * len(scales)
        else:
            assert len(weights) == len(scales), "Weights must match number of scales"
            self.weights = weights
        
        # Create Sobel filters for x and y directions without registering as buffers
        # We'll move these to the appropriate device in the forward pass
        self.sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        self.sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    
    def compute_gradient_magnitude(self, x):
        """Compute gradient magnitude using Sobel filters"""
        batch_size, channels, height, width = x.shape
        
        # Move Sobel filters to the same device as input
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        
        # Compute gradients for each channel separately
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(x)
        
        for c in range(channels):
            # Extract channel and add batch dimension for conv2d
            channel = x[:, c:c+1, :, :]
            # Apply Sobel filters
            grad_x[:, c:c+1, :, :] = F.conv2d(F.pad(channel, [1, 1, 1, 1], mode='reflect'), sobel_x)
            grad_y[:, c:c+1, :, :] = F.conv2d(F.pad(channel, [1, 1, 1, 1], mode='reflect'), sobel_y)
            
        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        return grad_magnitude
    
    def forward(self, pred, target):
        """
        Compute multi-scale gradient loss between predicted and target images
        
        Args:
            pred (torch.Tensor): Predicted images [B, C, H, W]
            target (torch.Tensor): Target images [B, C, H, W]
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        total_loss = 0
        
        for scale_idx, scale in enumerate(self.scales):
            # Downsample images if scale > 1
            if scale > 1:
                curr_pred = F.interpolate(pred, scale_factor=1/scale, mode='bilinear', align_corners=False)
                curr_target = F.interpolate(target, scale_factor=1/scale, mode='bilinear', align_corners=False)
            else:
                curr_pred = pred
                curr_target = target
            
            # Compute gradient magnitudes
            pred_grad = self.compute_gradient_magnitude(curr_pred)
            target_grad = self.compute_gradient_magnitude(curr_target)
            
            # Compute L1 loss between gradients
            scale_loss = F.l1_loss(pred_grad, target_grad)
            
            # Weight the loss by scale importance
            total_loss += self.weights[scale_idx] * scale_loss
            
        return total_loss


# Example of how to use this loss in training:
def create_combined_microscopy_loss():
    """
    Creates a combined loss function with MSE, SSIM and multi-scale gradient loss
    for microscopy image enhancement
    """
    # Import SSIM if available
    ssim_available = False
    try:
        from pytorch_msssim import SSIM
        ssim_available = True
    except ImportError:
        print("pytorch_msssim not found. Using MSE and gradient loss only.")
    
    # Create component losses
    mse_loss = nn.MSELoss()
    gradient_loss = MultiScaleGradientLoss(scales=(1, 2, 4), weights=[0.5, 0.3, 0.2])
    
    if ssim_available:
        ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
        
        def combined_loss(pred, target):
            # Make sure both inputs are on the same device
            if pred.device != target.device:
                target = target.to(pred.device)
            
            # Compute individual losses
            mse = mse_loss(pred, target)
            ssim_value = 1.0 - ssim_module(pred, target)  # Convert to loss (1-SSIM)
            grad_loss = gradient_loss(pred, target)
            
            # Combine with weights
            return 0.4 * mse + 0.3 * ssim_value + 0.3 * grad_loss
    else:
        def combined_loss(pred, target):
            # Make sure both inputs are on the same device
            if pred.device != target.device:
                target = target.to(pred.device)
                
            # Compute individual losses
            mse = mse_loss(pred, target)
            grad_loss = gradient_loss(pred, target)
            
            # Combine with weights
            return 0.7 * mse + 0.3 * grad_loss
    
    return combined_loss
