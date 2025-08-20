from typing import List, Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallModel(nn.Module):
    """PyTorch port of build_model_small from train.py.

    Input: B x 6 x H x W
    Output: B x 2 x H/8 x W/8 (matches TF sequence: pool 2x2 -> 3 pools total)
    """

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(6, 16)
        self.pool1 = nn.AvgPool2d(2)

        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.AvgPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.AvgPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 5, padding=2),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Conv2d(256, 2, 5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)
        x = self.enc4(x)
        x = self.out_conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GenericAE(nn.Module):
    """PyTorch port of build_model_ae from train_generic.py.

    Returns a tuple of multi-scale outputs: (full, x2, x4, x8, x16)
    Each output has 2 channels (flow x, y).
    """

    def __init__(self):
        super().__init__()
        padding = 2
        # Encoder
        self.enc1 = ConvBlock(8, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 128, 5, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)

        # Decoder with lateral connections and intermediate outputs
        self.dec4 = UpBlock(128, 128)
        self.dec4_out = nn.Conv2d(128, 2, 5, padding=2)

        self.dec3 = UpBlock(128 + 128, 128)
        self.dec3_out = nn.Conv2d(128, 2, 5, padding=2)

        # Input to dec2 is cat([upsampled d3 (128), c3 (128)]) -> 256
        self.dec2 = UpBlock(256, 64)
        self.dec2_out = nn.Conv2d(64, 2, 5, padding=2)

        # After upsample dec2 (64) and concat with c2 (64) -> 128
        self.dec1 = UpBlock(128, 32)
        self.dec1_out = nn.Conv2d(32, 2, 5, padding=2)

        # After upsample dec1 (32) and concat with c1 (32) -> 64
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)
        p3 = self.pool3(c3)

        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        # Decoder stage 4
        d4 = self.dec4(p4)
        d4_out = self.dec4_out(d4)
        d4 = nn.functional.interpolate(d4, scale_factor=2, mode="nearest")
        d4 = torch.cat([d4, c4], dim=1)

        # Decoder stage 3
        d3 = self.dec3(d4)
        d3_out = self.dec3_out(d3)
        d3 = nn.functional.interpolate(d3, scale_factor=2, mode="nearest")
        d3 = torch.cat([d3, c3], dim=1)

        # Decoder stage 2
        d2 = self.dec2(d3)
        d2_out = self.dec2_out(d2)
        d2 = nn.functional.interpolate(d2, scale_factor=2, mode="nearest")
        d2 = torch.cat([d2, c2], dim=1)

        # Decoder stage 1
        d1 = self.dec1(d2)
        d1_out = self.dec1_out(d1)
        d1 = nn.functional.interpolate(d1, scale_factor=2, mode="nearest")
        d1 = torch.cat([d1, c1], dim=1)

        out_full = self.final(d1)
        return out_full, d1_out, d2_out, d3_out, d4_out

