"""
    FiLM-Enhanced UNet / UNet 3+ network modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.proj = nn.Linear(emb_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias[:channels] = 1.0
            self.proj.bias[channels:] = 0.0

    def forward(self, x, emb):
        params = self.proj(emb)
        scale, shift = params.chunk(2, dim=1)
        return x * scale[:, :, None, None] + shift[:, :, None, None]


class FiLMConvBlock(nn.Module):
    """Two-layer convolution + FiLM"""
    
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.film1 = FiLM(out_ch, emb_dim)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film2 = FiLM(out_ch, emb_dim)

    def forward(self, x, emb):
        x = F.relu(self.film1(self.norm1(self.conv1(x)), emb), inplace=True)
        x = F.relu(self.film2(self.norm2(self.conv2(x)), emb), inplace=True)
        return x


class FiLMConvBlock3Plus(nn.Module):
    """Single-layer convolution + FiLM (for UNet3+ aggregation)"""
    
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film = FiLM(out_ch, emb_dim)

    def forward(self, x, emb):
        return F.relu(self.film(self.norm(self.conv(x)), emb), inplace=True)


class DownsampleConv(nn.Module):
    """
    More physics-friendly downsampling: strided conv + normalization + ReLU
    - vs MaxPool: does not take extremes directly, less biased toward outliers
    - vs plain stride conv: adds normalization/activation for stability
    """

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)), inplace=True)


class DownsampleStack(nn.Module):
    """Repeated stride=2 DownsampleConv; supports factor=2/4/8/..."""

    def __init__(self, channels: int, factor: int):
        super().__init__()
        if factor < 1 or (factor & (factor - 1)) != 0:
            raise ValueError(f"factor must be a power of 2, got factor={factor}")
        steps = int(math.log2(factor)) if factor > 1 else 0
        self.net = nn.Sequential(*[DownsampleConv(channels, stride=2) for _ in range(steps)])

    def forward(self, x):
        return self.net(x)


class FiLM_UNet(nn.Module):
    """Standard FiLM U-Net"""
    
    def __init__(self, in_ch, out_ch, emb_dim=128):
        super().__init__()

        f = [96, 192, 384, 768, 1536] 
        # f = [64, 128, 256, 512, 1024]
        
        # Separate downsampling layers (strided conv)
        self.enc1 = FiLMConvBlock(in_ch, f[0], emb_dim)
        self.down1 = DownsampleConv(f[0])
        
        self.enc2 = FiLMConvBlock(f[0], f[1], emb_dim)
        self.down2 = DownsampleConv(f[1])
        
        self.enc3 = FiLMConvBlock(f[1], f[2], emb_dim)
        self.down3 = DownsampleConv(f[2])
        
        self.enc4 = FiLMConvBlock(f[2], f[3], emb_dim)
        self.down4 = DownsampleConv(f[3])
        
        self.enc5 = FiLMConvBlock(f[3], f[4], emb_dim)

        self.up5 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2)
        self.dec5 = FiLMConvBlock(f[4], f[3], emb_dim)
        self.up4 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec4 = FiLMConvBlock(f[3], f[2], emb_dim)
        self.up3 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec3 = FiLMConvBlock(f[2], f[1], emb_dim)
        self.up2 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec2 = FiLMConvBlock(f[1], f[0], emb_dim)
        
        self.final = nn.Conv2d(f[0], out_ch, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x, emb):
        e1 = self.enc1(x, emb)
        e2 = self.enc2(self.down1(e1), emb)
        e3 = self.enc3(self.down2(e2), emb)
        e4 = self.enc4(self.down3(e3), emb)
        e5 = self.enc5(self.down4(e4), emb)

        d5 = self._up_cat(self.up5(e5), e4)
        d5 = self.dec5(d5, emb)
        d4 = self._up_cat(self.up4(d5), e3)
        d4 = self.dec4(d4, emb)
        d3 = self._up_cat(self.up3(d4), e2)
        d3 = self.dec3(d3, emb)
        d2 = self._up_cat(self.up2(d3), e1)
        d2 = self.dec2(d2, emb)

        return self.final(d2)
    
    def _up_cat(self, up, skip):
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([skip, up], dim=1)


class FiLM_UNet3Plus(nn.Module):
    """FiLM UNet 3+ (gradient checkpointing)"""
    
    def __init__(self, in_ch, out_ch, emb_dim=128, deep_supervision=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.deep_supervision = deep_supervision
        
        f = [96, 192, 384, 768, 1536]
        # f = [64, 128, 256, 512, 1024]
        cat_ch = f[0]
        up_ch = 5 * cat_ch
        
        # Encoder (strided conv instead of MaxPool)
        self.e1 = FiLMConvBlock(in_ch, f[0], emb_dim)
        self.pool2 = DownsampleConv(f[0])  # Downsample 1->2
        self.e2 = FiLMConvBlock(f[0], f[1], emb_dim)
        self.pool3 = DownsampleConv(f[1])  # Downsample 2->3
        self.e3 = FiLMConvBlock(f[1], f[2], emb_dim)
        self.pool4 = DownsampleConv(f[2])  # Downsample 3->4
        self.e4 = FiLMConvBlock(f[2], f[3], emb_dim)
        self.pool5 = DownsampleConv(f[3])  # Downsample 4->5
        self.e5 = FiLMConvBlock(f[3], f[4], emb_dim)

        # UNet3+ multi-scale downsampling (replaces max_pool2d)
        self.ds_e1_2 = DownsampleStack(f[0], 2)
        self.ds_e1_4 = DownsampleStack(f[0], 4)
        self.ds_e1_8 = DownsampleStack(f[0], 8)
        self.ds_e2_2 = DownsampleStack(f[1], 2)
        self.ds_e2_4 = DownsampleStack(f[1], 4)
        self.ds_e3_2 = DownsampleStack(f[2], 2)
        
        # Decoder 4
        self.d4_e1 = FiLMConvBlock3Plus(f[0], cat_ch, emb_dim)
        self.d4_e2 = FiLMConvBlock3Plus(f[1], cat_ch, emb_dim)
        self.d4_e3 = FiLMConvBlock3Plus(f[2], cat_ch, emb_dim)
        self.d4_e4 = FiLMConvBlock3Plus(f[3], cat_ch, emb_dim)
        self.d4_e5 = FiLMConvBlock3Plus(f[4], cat_ch, emb_dim)
        self.d4_conv = FiLMConvBlock3Plus(up_ch, up_ch, emb_dim)
        
        # Decoder 3
        self.d3_e1 = FiLMConvBlock3Plus(f[0], cat_ch, emb_dim)
        self.d3_e2 = FiLMConvBlock3Plus(f[1], cat_ch, emb_dim)
        self.d3_e3 = FiLMConvBlock3Plus(f[2], cat_ch, emb_dim)
        self.d3_d4 = FiLMConvBlock3Plus(up_ch, cat_ch, emb_dim)
        self.d3_e5 = FiLMConvBlock3Plus(f[4], cat_ch, emb_dim)
        self.d3_conv = FiLMConvBlock3Plus(up_ch, up_ch, emb_dim)
        
        # Decoder 2
        self.d2_e1 = FiLMConvBlock3Plus(f[0], cat_ch, emb_dim)
        self.d2_e2 = FiLMConvBlock3Plus(f[1], cat_ch, emb_dim)
        self.d2_d3 = FiLMConvBlock3Plus(up_ch, cat_ch, emb_dim)
        self.d2_d4 = FiLMConvBlock3Plus(up_ch, cat_ch, emb_dim)
        self.d2_e5 = FiLMConvBlock3Plus(f[4], cat_ch, emb_dim)
        self.d2_conv = FiLMConvBlock3Plus(up_ch, up_ch, emb_dim)
        
        # Decoder 1
        self.d1_e1 = FiLMConvBlock3Plus(f[0], cat_ch, emb_dim)
        self.d1_d2 = FiLMConvBlock3Plus(up_ch, cat_ch, emb_dim)
        self.d1_d3 = FiLMConvBlock3Plus(up_ch, cat_ch, emb_dim)
        self.d1_d4 = FiLMConvBlock3Plus(up_ch, cat_ch, emb_dim)
        self.d1_e5 = FiLMConvBlock3Plus(f[4], cat_ch, emb_dim)
        self.d1_conv = FiLMConvBlock3Plus(up_ch, up_ch, emb_dim)
        
        # Output
        self.final = nn.Conv2d(up_ch, out_ch, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        
        if deep_supervision:
            self.sup4 = nn.Conv2d(up_ch, out_ch, 1)
            self.sup3 = nn.Conv2d(up_ch, out_ch, 1)
            self.sup2 = nn.Conv2d(up_ch, out_ch, 1)
            self.sup5 = nn.Conv2d(f[4], out_ch, 1)
            for s in [self.sup4, self.sup3, self.sup2, self.sup5]:
                nn.init.zeros_(s.weight)
                nn.init.zeros_(s.bias)

    def _enc(self, x, emb):
        e1 = self.e1(x, emb)
        e2 = self.e2(self.pool2(e1), emb)
        e3 = self.e3(self.pool3(e2), emb)
        e4 = self.e4(self.pool4(e3), emb)
        e5 = self.e5(self.pool5(e4), emb)
        return e1, e2, e3, e4, e5

    def _dec4(self, e1, e2, e3, e4, e5, emb):
        h, w = e4.shape[2:]
        return self.d4_conv(torch.cat([
            self.d4_e1(self.ds_e1_8(e1), emb),
            self.d4_e2(self.ds_e2_4(e2), emb),
            self.d4_e3(self.ds_e3_2(e3), emb),
            self.d4_e4(e4, emb),
            self.d4_e5(F.interpolate(e5, (h, w), mode='bilinear', align_corners=False), emb)
        ], 1), emb)

    def _dec3(self, e1, e2, e3, d4, e5, emb):
        h, w = e3.shape[2:]
        return self.d3_conv(torch.cat([
            self.d3_e1(self.ds_e1_4(e1), emb),
            self.d3_e2(self.ds_e2_2(e2), emb),
            self.d3_e3(e3, emb),
            self.d3_d4(F.interpolate(d4, (h, w), mode='bilinear', align_corners=False), emb),
            self.d3_e5(F.interpolate(e5, (h, w), mode='bilinear', align_corners=False), emb)
        ], 1), emb)

    def _dec2(self, e1, e2, d3, d4, e5, emb):
        h, w = e2.shape[2:]
        return self.d2_conv(torch.cat([
            self.d2_e1(self.ds_e1_2(e1), emb),
            self.d2_e2(e2, emb),
            self.d2_d3(F.interpolate(d3, (h, w), mode='bilinear', align_corners=False), emb),
            self.d2_d4(F.interpolate(d4, (h, w), mode='bilinear', align_corners=False), emb),
            self.d2_e5(F.interpolate(e5, (h, w), mode='bilinear', align_corners=False), emb)
        ], 1), emb)

    def _dec1(self, e1, d2, d3, d4, e5, emb):
        h, w = e1.shape[2:]
        return self.d1_conv(torch.cat([
            self.d1_e1(e1, emb),
            self.d1_d2(F.interpolate(d2, (h, w), mode='bilinear', align_corners=False), emb),
            self.d1_d3(F.interpolate(d3, (h, w), mode='bilinear', align_corners=False), emb),
            self.d1_d4(F.interpolate(d4, (h, w), mode='bilinear', align_corners=False), emb),
            self.d1_e5(F.interpolate(e5, (h, w), mode='bilinear', align_corners=False), emb)
        ], 1), emb)

    def forward(self, x, emb):
        if self.use_checkpoint and self.training:
            e1, e2, e3, e4, e5 = checkpoint(self._enc, x, emb, use_reentrant=False)
            d4 = checkpoint(self._dec4, e1, e2, e3, e4, e5, emb, use_reentrant=False)
            d3 = checkpoint(self._dec3, e1, e2, e3, d4, e5, emb, use_reentrant=False)
            d2 = checkpoint(self._dec2, e1, e2, d3, d4, e5, emb, use_reentrant=False)
            d1 = checkpoint(self._dec1, e1, d2, d3, d4, e5, emb, use_reentrant=False)
        else:
            e1, e2, e3, e4, e5 = self._enc(x, emb)
            d4 = self._dec4(e1, e2, e3, e4, e5, emb)
            d3 = self._dec3(e1, e2, e3, d4, e5, emb)
            d2 = self._dec2(e1, e2, d3, d4, e5, emb)
            d1 = self._dec1(e1, d2, d3, d4, e5, emb)
        
        out = self.final(d1)
        
        if self.deep_supervision and self.training:
            h, w = e1.shape[2:]
            return [
                out,
                F.interpolate(self.sup2(d2), (h, w), mode='bilinear', align_corners=False),
                F.interpolate(self.sup3(d3), (h, w), mode='bilinear', align_corners=False),
                F.interpolate(self.sup4(d4), (h, w), mode='bilinear', align_corners=False),
                F.interpolate(self.sup5(e5), (h, w), mode='bilinear', align_corners=False)
            ]
        return out





def estimate_memory(batch_size, input_size=(408, 440)):
    """Estimate UNet3+ memory usage (MB)"""

    f = [96, 192, 384, 768, 1536]
    # f = [64, 128, 256, 512, 1024]
    H = ((input_size[0] + 15) // 16) * 16
    W = ((input_size[1] + 15) // 16) * 16
    
    # Encoder
    enc = [(H >> i, W >> i, f[i]) for i in range(5)]
    total = sum(batch_size * c * h * w for h, w, c in enc)
    
    # Decoder
    dec_ch = 5 * f[0]
    for i in range(4):
        scale = 8 >> i
        total += batch_size * dec_ch * (H // scale) * (W // scale)
    
    # Gradients
    total *= 2
    
    return total * 4 / (1024 ** 2)


if __name__ == "__main__":
    for bs in [1, 2, 4, 8]:
        mem = estimate_memory(bs)
        print(f"Batch {bs}: ~{mem:.0f} MB ({mem/1024:.2f} GB)")

