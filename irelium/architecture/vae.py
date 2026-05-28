'''
irelium
XQ
2026-05-20
dbvae
'''

import torch
import torch.nn as nn
from irelium.neuron.block import ConvTransposeBlock

def cnnDecoder(
    latent_dim: int,
    base_filter: int,
    bottleneck_multiplier: int = 6,
    stride: int = 2,
    channel_schedule: list = [4, 2, 1, 0],
    kernel_schedule: list  = [3, 3, 5, 5],
    padding_schedule: list = [1, 1, 2, 2],
    out_channels: int = 3,
) -> nn.Sequential:
    
    '''
    VAE decoder: latent vector z → reconstructed image [B, 3, 64, 64].

    Args:
        latent_dim:             Latent vector dimension.
        base_filter:            Base filter count, must match encoder.
        bottleneck_multiplier:  Channel multiplier at latent→spatial bridge.
        channel_schedule:       Descending multipliers, 0 = RGB output.
        kernel_schedule:        Kernel size per block.
        padding_schedule:       Padding per block.
        out_channels:           Output image channels (default 3 = RGB).
        device:                 Target device.

    Returns:
        nn.Sequential decoder on `device`.

    Raises:
        AssertionError: If schedules are not the same length.
    '''
    assert len(channel_schedule) == len(kernel_schedule) == len(padding_schedule), \
        "channel, kernel, padding schedules must be same length"

    conv_decode_blocks = []
    in_ch = bottleneck_multiplier * base_filter
    
    # setting Block
    for multiplier, kernel, padding in zip(
        channel_schedule,
        kernel_schedule,
        padding_schedule):
        
        is_last_layer = (multiplier == 0)
        out_ch = out_channels if is_last_layer else multiplier * base_filter
        
        cnnBlock = ConvTransposeBlock(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_padding=1,
            )
        if is_last_layer:
            conv_decode_blocks.append(cnnBlock)
        else: 
            conv_decode_blocks.append(
                nn.Sequential(
                    cnnBlock,
                    nn.ReLU(inplace=True),
                ))
        in_ch = out_ch
    
    
    model = nn.Sequential(
        nn.Linear(latent_dim, 4 * 4 * bottleneck_multiplier * base_filter),
        nn.ReLU(inplace=True),
        _Reshape(bottleneck_multiplier * base_filter, 4, 4),
        *conv_decode_blocks,
    )
    
    return model

# --------------------------------------------------
# UTILITY BLOCKS
# --------------------------------------------------
class _Reshape(nn.Module):
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
        return x.view(-1, *self.shape)