'''
irelium
XQ
2026-05-20
reusable primitive building blocks
'''

import torch
import torch.nn as nn


class ConvNormBlock(nn.Module):
    '''
    Conv2d → BatchNorm2d → ReLU with learned spatial downsampling.

    Strided conv instead of MaxPool — preserves spatial information
    for VAE reconstruction; decoder can learn the inverse mapping.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        kernel_size:  Convolution kernel size.
        stride:       Spatial downsampling factor.
        padding:      Zero-padding on each side.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int
    ) -> None:
        super().__init__()
        self.normblock = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride, padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normblock(x)

# --- decoder block
class ConvTransposeBlock(nn.Module):
    '''
    ConvTranspose2d for spatial upsampling in VAE decoder.

    No BatchNorm or activation here — caller wraps with nn.Sequential
    + ReLU if needed. Last decoder layer must be activation-free to
    allow raw pixel values; keeping activation outside the block makes
    this explicit at the assembly level.

    Args:
        in_channels:    Input feature channels.
        out_channels:   Output feature channels.
        kernel_size:    Convolution kernel size.
        stride:         Spatial upsampling factor.
        padding:        Zero-padding on each side.
        output_padding: Extra rows/cols added to output shape (0 or 1).
                        Required when stride > 1 to resolve shape ambiguity.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int
        ) -> None:
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            output_padding=output_padding,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_t(x)
    
class Reshape(nn.Module):
    '''
    Reshape tensor to target shape inside nn.Sequential.

    Enables latent vector → spatial feature map transition
    in decoder without breaking the Sequential pipeline.

    Args:
        *shape: Target shape excluding batch dimension.

    Example:
        Reshape(256, 4, 4)  # [B, 4096] → [B, 256, 4, 4]
    '''

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), *self.shape)