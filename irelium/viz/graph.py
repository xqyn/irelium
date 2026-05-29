'''
irelium
XQ
2026-05-20
Diagnostic visualization utilities.
'''

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable


def plot_grad_norms(model: nn.Module) -> None:
    '''Bar chart of per-layer gradient L2 norms after backward pass.'''
    names, norms = [], []
    for name, param in model.named_parameters():
        if param.grad is not None:
            names.append(name)
            norms.append(param.grad.norm().item())

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(norms)), norms)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("Gradient Norm")
    plt.title("Per-layer Gradient Norms")
    plt.tight_layout()
    plt.show()


def plot_weight_distributions(model: nn.Module) -> None:
    '''Histogram of weight distributions per trainable layer.'''
    params = [
        (n, p.detach().cpu().numpy().flatten())
        for n, p in model.named_parameters()
        if p.requires_grad
    ]

    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3))
    axes = np.atleast_1d(axes)

    for ax, (name, weights) in zip(axes, params):
        ax.hist(weights, bins=50)
        ax.set_title(name, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_loss_landscape(
    model: nn.Module,
    loss_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 20,
    scale: float = 0.01,
) -> None:
    '''
    2D contour plot of loss surface along two random directions in weight space.

    Perturbs model weights along random directions dx, dy and evaluates loss.
    Model weights are always restored after plotting.
    '''
    orig = [p.data.clone() for p in model.parameters()]
    dx   = [torch.randn_like(p) for p in model.parameters()]
    dy   = [torch.randn_like(p) for p in model.parameters()]

    alphas = torch.linspace(-1, 1, steps)
    betas  = torch.linspace(-1, 1, steps)
    Z      = np.zeros((steps, steps))

    try:
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                for p, o, u, v in zip(model.parameters(), orig, dx, dy):
                    p.data = o + scale * a * u + scale * b * v
                with torch.no_grad():
                    Z[i, j] = loss_fn(model(x), y).item()
    finally:
        # restore original weights — always runs even if exception occurs
        for p, o in zip(model.parameters(), orig):
            p.data = o

    plt.figure(figsize=(6, 5))
    plt.contourf(alphas, betas, Z, levels=50, cmap="viridis")
    plt.colorbar(label="loss")
    plt.title("Loss Landscape")
    plt.show()


def plot_activations(model: nn.Module, x: torch.Tensor) -> None:
    '''Histogram of activation distributions at each leaf module.'''
    activations = {}

    def hook(name: str):
        def fn(module, input, output):
            activations[name] = output.detach().cpu()
        return fn

    handles = [
        m.register_forward_hook(hook(name))
        for name, m in model.named_modules()
        if len(list(m.children())) == 0
    ]

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    fig, axes = plt.subplots(1, len(activations), figsize=(4 * len(activations), 3))
    axes = np.atleast_1d(axes)

    for ax, (name, act) in zip(axes, activations.items()):
        ax.hist(act.numpy().flatten(), bins=50)
        ax.set_title(name, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_training_curves(
    loss_history: list,
    acc_history: list | None = None,
) -> None:
    '''Loss (log scale) and optional accuracy curves over training steps.'''
    fig, axes = plt.subplots(1, 2 if acc_history else 1, figsize=(12, 4))
    axes = np.atleast_1d(axes)

    axes[0].semilogy(loss_history)
    axes[0].set_title("Loss (log scale)")
    axes[0].set_xlabel("Step")
    axes[0].grid(True)

    if acc_history:
        axes[1].plot(acc_history)
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Step")
        axes[1].grid(True)

    plt.tight_layout()
    plt.show()