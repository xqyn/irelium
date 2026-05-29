'''
irelium
XQ
2026-05-22
Shared utilities: config loading and tensor conversion.
'''

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from box import Box
import yaml



# config/ sits at project root — two levels up from this file
_CONFIG_ROOT = Path(__file__).parent.parent / "config"

def load_config(name: str) -> Box:
    '''
    Load a YAML config by name from config/.

    Args:
        name: Config filename without extension (e.g. 'backbone').

    Returns:
        Box of config values — supports dot access.

    Raises:
        FileNotFoundError: If config file does not exist.
    '''
    path = _CONFIG_ROOT / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return Box(yaml.safe_load(f))
    

def as_tensor(
    x: np.ndarray | torch.Tensor,
    model: nn.Module | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    '''
    Convert x to float32 tensor, optionally move to device.

    Device resolution order:
        1. model parameter device — if model provided
        2. device argument        — if provided
        3. CPU                    — default

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
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

    if model is not None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model)}")
        if next(model.parameters(), None) is None:
            raise ValueError("Model has no parameters — cannot infer device")
        device = next(model.parameters()).device

    if device is not None:
        x = x.to(device)

    return x
