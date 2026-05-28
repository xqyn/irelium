'''
irelium
XQ
2026-05-26
'''

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from irelium.utils import as_tensor
from typing import Callable


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
        x:         Input tensor or numpy array.
        y:         Target tensor or numpy array.
        model:     Any nn.Module.
        optimizer: Any torch optimizer.
        step_fn:   Callable(x, y, model) -> scalar Tensor.

    Returns:
        Scalar loss as Python float.

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
    
    return loss
    

def evaluation(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    '''
    Evaluate model on a single batch and return a scalar metric.

    Args:
        x:         Input tensor  [B, ...].
        y:         Target tensor [B, ...].
        model:     Any nn.Module.
        metric_fn: Callable(y_pred, y_true) -> scalar Tensor.
        pred_fn:   Callable(logits) -> predictions.

    Returns:
        Scalar metric as Python float.
    '''
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError(f"x and y must be torch.Tensor, got {type(x)}, {type(y)}")

    # sending x, y into same accelerator
    x = as_tensor(x, model = model)
    y = as_tensor(y, model = model)
    
    # model evaluation
    model.eval()
    
    with torch.inference_mode():
        logits = model(x)
        y_pred = pred_fn(logits)
        metric = metric_fn(y_pred, y)
    return metric
        
        