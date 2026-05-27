'''
irelium
XQ
2026-05-20
cnn baseline
'''

import torch.nn as nn
#from ..neuron.block import ConvNormBlock
from neuron.block import ConvNormBlock

def cnnConvNorm(
    n_outputs: int,
    H: int,
    W: int,
    in_channels: int,
    base_filter: int,
    stride: int,
    channel_schedule: list,
) -> nn.Sequential:
    '''
    Baseline CNN: strided conv blocks → linear classification head.

    Args:
        n_outputs:        Number of output logits.
        H:                Input image height (must be divisible by stride^n_blocks).
        W:                Input image width  (must be divisible by stride^n_blocks).
        in_channels:      Input image channels.
        base_filter:      Base filter count, scaled per block by channel_schedule.
        stride:           Stride per block — each block divides H and W by stride.
        channel_schedule: Filter multipliers per block — controls depth and capacity.

    Returns:
        nn.Sequential model on `device`.

    Raises:
        ValueError: If H or W are not divisible by stride^len(channel_schedule).
    '''
    
    # stride compounds: stride=2, 4 blocks → 2^4=16 total spatial reduction
    stride_times = stride ** len(channel_schedule)
    
    if H % stride_times != 0 or W % stride_times != 0:
        raise ValueError(
            f"H and W must be divisible by {stride_times} "
            f"({stride}^{len(channel_schedule)} blocks), got H={H}, W={W}."
        )
        
    in_ch = in_channels
    conv_blocks = []

    for multiplier in channel_schedule:
        out_ch = multiplier * base_filter
        conv_blocks.append(
            ConvNormBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=5,
                stride=stride,
                padding=2,
            )
        )
        in_ch = out_ch 
    
    # last block outputs channel_schedule[-1] * base_filter channels
    flat_dim = (H // stride_times) * (W // stride_times) * channel_schedule[-1] * base_filter

    models = nn.Sequential(
        # convo layers
        *conv_blocks,
        # classification layers
        ## flatten [B, C, H, W] → [B, C*H*W]
        nn.Flatten(), 
        nn.Linear(flat_dim, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, n_outputs),
    )    
      
    return models
