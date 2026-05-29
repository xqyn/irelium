'''
irelium
XQ
2026-05-20
CNN decoder: latent vector → reconstructed image.
    05-29: add _CFG
'''

import torch.nn as nn
from irelium.neuron.conv import ConvTransposeBlock, Reshape
from irelium.utils import load_config

_CFG = load_config("decoder")

def Decoder(
    latent_dim: int,
    base_filter: int,
) -> nn.Sequential:
    
    '''
    Latent vector z → reconstructed image [B, out_channels, H, W].

    All architecture hyperparameters loaded from config/decoder.yaml.

    Args:
        latent_dim:  Latent vector dimension — must match encoder.
        base_filter: Base filter count — must match encoder.

    Returns:
        nn.Sequential decoder.

    Raises:
        AssertionError: If channel, kernel, padding schedules differ in length.
    '''
    bottleneck_multiplier = _CFG.bottleneck_multiplier
    bottleneck_spatial    = _CFG.bottleneck_spatial
    stride                = _CFG.stride
    out_channels          = _CFG.out_channels
    channel_schedule      = _CFG.channel_schedule
    kernel_schedule       = _CFG.kernel_schedule
    padding_schedule      = _CFG.padding_schedule
    
    assert len(channel_schedule) == len(kernel_schedule) == len(padding_schedule), \
        "channel, kernel, padding schedules must be same length"

    conv_blocks = []
    in_ch       = bottleneck_multiplier * base_filter
    
    for multiplier, kernel, padding in zip(
        channel_schedule,
        kernel_schedule,
        padding_schedule):
        
        is_last = (multiplier == 0)
        out_ch = out_channels if is_last else multiplier * base_filter
        
        conv_t = ConvTransposeBlock(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_padding=1,
            )
        
        # last layer — no activation, raw pixel output
        
        conv_blocks.append(
            conv_t if is_last else
            nn.Sequential(
                conv_t,
                nn.ReLU(inplace=True),
            ))
        
        in_ch = out_ch
    
    decoder = nn.Sequential(
        nn.Linear(latent_dim, bottleneck_spatial ** 2 * bottleneck_multiplier * base_filter),
        nn.ReLU(inplace=True),
        Reshape(bottleneck_multiplier * base_filter, bottleneck_spatial, bottleneck_spatial),
        *conv_blocks,
    )
    
    return decoder

