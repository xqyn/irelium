'''
irelium
XQ
2026-05-22
'''

import numpy as np
import torch
import torch.nn as nn

def as_tensor(
  x: np.ndarray | torch.Tensor,
  model:  nn.Module | None = None,
  device: torch.device | None = None,
):
    '''
    Convert x to float32 tensor, optionally move to device.

    Device resolution order:
        1. model.parameters().device  — if model provided
        2. device argument            — if provided
        3. CPU                        — default

    Args:
        x:      Input array or tensor.
        model:  nn.Module — infers device from parameters.
                Takes priority over device argument.
        device: Explicit target device. Used if model is None.

    Returns:
        torch.Tensor float32 on resolved device.

    Raises:
        TypeError:  If x is not np.ndarray or torch.Tensor.
        TypeError:  If model is not nn.Module.
        ValueError: If model has no parameters.
    '''
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x, dtype=torch.float32)
    elif torch.is_tensor(x):
        x = x.to(dtype=torch.float32)
    else:
        raise TypeError(f'Expected np.ndarray or torch.Tensor, got {type(x)}')

    if model is not None:
        if not isinstance(model, nn.Module):
            raise TypeError(f'Expected nn.Module, got {type(model)}')
        if not any(True for _ in model.parameters()):
            raise ValueError('Model has no parameters — cannot infer device')
        device = next(model.parameters()).device

    if device is not None:
        x = x.to(device)

    return x