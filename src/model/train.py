'''
irelium
XQ
2026-05-36
'''

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from utils import as_tensor


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    step_fn: Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    '''
    Train model for one step and return scalar loss.

    Args:
        x: Input tensor, any shape.
        y: Target tensor, any shape compatible with loss_fn.
        model: Any nn.Module — moved to device before call.
        loss_fn: Any callable (nn.Module subclass, closure, or plain function)
                 with signature (pred, target) -> scalar Tensor.
        optimizer: Any torch optimizer bound to model.parameters().

    Returns:
        Scalar loss value as Python float.

    Raises:
        TypeError: If x or y are not torch.Tensor.
        ValueError: If loss_fn output is not a scalar Tensor.
    '''
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError(f"x and y must be torch.Tensor, got {type(x)}, {type(y)}")

    # sending x, y into same accelerator
    x = as_tensor(x, model = model)
    y = as_tensor(y, model = model)    
    
    # compute the loss
    model.train()
    optimizer.zero_grad()
    loss = step_fn(x, y, model)
    loss.backward()     # backpropagation
    optimizer.step()
    
    return loss.item()
    
    
    
    