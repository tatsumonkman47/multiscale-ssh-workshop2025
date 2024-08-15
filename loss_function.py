import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        
    def nan_mse_loss(self, output, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(float('nan'), device=input.device)
        out = (output[mask] - target[mask]) ** 2
        return out.mean()
        
    def compute_gradient(self, img):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
        
        return grad_x, grad_y

    def forward(self, output, target):
        # MSE loss ignoring NaN values
        mse_loss = self.nan_mse_loss(output, target)
        
        # Gradient loss
        output_grad_x, output_grad_y = self.compute_gradient(output)
        target_grad_x, target_grad_y = self.compute_gradient(target)
        
        # Mask for gradient loss (propagate NaN mask to gradients)
        grad_mask = ~torch.isnan(target)
        grad_mask = grad_mask & F.pad(grad_mask[:, :, 1:, :], (0, 0, 1, 0))  # shift mask for x gradient
        grad_mask = grad_mask & F.pad(grad_mask[:, :, :, 1:], (1, 0, 0, 0))  # shift mask for y gradient
        
        grad_loss_x = self.nan_mse_loss(output_grad_x[grad_mask], target_grad_x[grad_mask])
        grad_loss_y = self.nan_mse_loss(output_grad_y[grad_mask], target_grad_y[grad_mask])
        grad_loss = grad_loss_x + grad_loss_y
        
        # Combine losses
        combined_loss = mse_loss + self.alpha * grad_loss
        
        # Handle case where all losses are NaN
        if torch.isnan(combined_loss):
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        return combined_loss
        
   