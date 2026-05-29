'''
irelium
XQ
2026-05-26
Training and evaluation loop utilities.
'''

import torch
import torch.nn as nn
from typing import Callable
from irelium.utils import as_tensor

def train(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    step_fn: Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    '''
    Single training step — forward, loss, backward, update.

    Args:
        x:         Input tensor   [B, ...].
        y:         Target tensor  [B, ...].
        model:     Any nn.Module.
        step_fn:   Callable(x, y, model) -> scalar Tensor.
        optimizer: Any torch optimizer.

    Returns:
        Scalar loss as Python float.

    Raises:
        TypeError: If x or y are not torch.Tensor.
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
    loss.backward()
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
    Single evaluation step — forward, predict, metric.

    pred_fn handles model output unpacking — for DB_VAE pass
    pred_fn=lambda out: out[0] to extract y_logit from the tuple.

    Args:
        x:         Input tensor  [B, ...].
        y:         Target tensor [B, ...].
        model:     Any nn.Module.
        pred_fn:   Callable(model_output) -> predictions.
        metric_fn: Callable(y_pred, y_true) -> scalar.

    Returns:
        Scalar metric as Python float.

    Raises:
        TypeError: If x or y are not torch.Tensor.
    '''
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError(f"x and y must be torch.Tensor, got {type(x)}, {type(y)}")

    # sending x, y into same accelerator
    x = as_tensor(x, model = model)
    y = as_tensor(y, model = model)
    
    
    model.eval()
    
    with torch.inference_mode():
        logits = model(x)
        y_pred = pred_fn(logits)
        metric = metric_fn(y_pred, y)
    return metric
        
        