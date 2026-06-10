"""
    Weighted relative L2 loss function
"""

import torch
import torch.nn as nn


DEFAULT_HUMIDITY_WEIGHT = 10.0


class RelativeL2Loss(nn.Module):
    """
    Weighted relative L2 loss
    
    Args:
        return_per_step: whether to also return per-time-step loss values
        channel_weights: channel weights [C]; None uses default humidity weighting
        humidity_weight: humidity channel weight multiplier (only when channel_weights=None)
        clamp_min: minimum denominator to avoid division by zero
        clamp_max: maximum relative error to prevent extreme values dominating
        climatological_std: climatological std [C]; used as denominator if provided (more stable)
    
    Inputs:
        output: List[Tensor [B, C, H, W]]
        label:  List[Tensor [B, C, H, W]]
    """
    
    def __init__(self, return_per_step: bool = False, channel_weights: torch.Tensor = None,
                 humidity_weight: float = DEFAULT_HUMIDITY_WEIGHT,
                 clamp_min: float = 1e-10, clamp_max: float = 5.0,
                 climatological_std: torch.Tensor = None):
        super().__init__()
        self.return_per_step = return_per_step
        self.eps = 1e-16
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        if climatological_std is not None:
            self.register_buffer('climatological_std', climatological_std.float())
        else:
            self.climatological_std = None
        
        if channel_weights is not None:
            cw = channel_weights.float()
            cw = cw / (cw.mean() + 1e-9)
            self.register_buffer('channel_weights', cw)
        elif humidity_weight != 1.0:
            cw = _create_channel_weights(20, humidity_weight)
            cw = cw / (cw.mean() + 1e-9)
            self.register_buffer('channel_weights', cw)
        else:
            self.channel_weights = None

    def forward(self, output, label, weight_dict=None):
        total_loss = 0.0
        step_losses = []
        
        for i, (pred, gt) in enumerate(zip(output, label)):
            pred = pred.float()
            gt = gt.float()
            B, C, H, W = pred.shape

            # RMSE of difference
            mse_diff = torch.mean((pred - gt)**2, dim=(2, 3))
            rmse_diff = torch.sqrt(mse_diff + self.eps)
            
            # Denominator
            if self.climatological_std is not None:
                denominator = self.climatological_std.to(pred.device).view(1, C)
                denominator = torch.clamp(denominator, min=self.clamp_min)
            else:
                mse_gt = torch.mean(gt**2, dim=(2, 3))
                rmse_gt = torch.sqrt(mse_gt + self.eps)
                denominator = torch.clamp(rmse_gt, min=self.clamp_min)
            
            # Relative error with upper bound
            rel_rmse = rmse_diff / denominator
            rel_rmse = torch.clamp(rel_rmse, max=self.clamp_max)
            
            # Channel weighting
            if self.channel_weights is not None:
                w = self.channel_weights.to(pred.device).view(1, C)
                rel_rmse = rel_rmse * w
            
            loss_mean = rel_rmse.mean()
            if weight_dict is not None:
                loss_mean = loss_mean * weight_dict[i]
            
            step_losses.append(loss_mean.item())
            total_loss += loss_mean

        if self.return_per_step:
            return total_loss, step_losses
        return total_loss


def _create_channel_weights(num_channels: int = 20, humidity_weight: float = 10.0) -> torch.Tensor:
    """Create channel weights with humidity channels (8-11) weighted."""
    weights = torch.ones(num_channels)
    weights[8:12] = humidity_weight
    return weights
