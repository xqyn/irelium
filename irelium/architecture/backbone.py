'''
irelium
XQ
2026-05-20
CNN backbone with linear classification head.
    05-29: add _CFG
'''

import torch.nn as nn
from irelium.neuron.conv import ConvNormBlock
from irelium.utils import load_config

_CFG = load_config("backbone")

def ConvBackbone(
    n_outputs: int,
    H: int,
    W: int,
    in_channels: int,
    base_filter: int,
    channel_schedule: list,
) -> nn.Sequential:
    '''
    Baseline CNN: strided conv blocks → linear classification head.

    Args:
        n_outputs:        Number of output logits.
        H:                Input image height — must be divisible by stride^len(channel_schedule).
        W:                Input image width  — must be divisible by stride^len(channel_schedule).
        in_channels:      Input image channels.
        base_filter:      Base filter count, scaled per block by channel_schedule.
        channel_schedule: Filter multipliers per block — controls depth and width.

    Returns:
        nn.Sequential backbone.

    Raises:
        ValueError: If H or W are not divisible by stride^len(channel_schedule).
    '''
    stride       = _CFG.conv.stride
    kernel_size  = _CFG.conv.kernel_size
    padding      = _CFG.conv.padding
    hidden_dim   = _CFG.head.hidden_dim
    
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
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        in_ch = out_ch 
    
    flat_dim = (H // stride_times) * (W // stride_times) * in_ch

    backbone = nn.Sequential(
        *conv_blocks,
        nn.Flatten(), 
        nn.Linear(flat_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, n_outputs),
    )    
      
    return backbone
