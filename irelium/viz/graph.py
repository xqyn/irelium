def plot_grad_norms(model):
    names, norms = [], []
    for name, param in model.named_parameters():
        if param.grad is not None:
            names.append(name)
            norms.append(param.grad.norm().item())
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(norms)), norms)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("Gradient Norm")
    plt.title("Per-layer Gradient Norms")
    plt.tight_layout()
    plt.show()

def plot_weight_distributions(model):
    import matplotlib.pyplot as plt
    params = [(n, p.detach().cpu().numpy().flatten()) 
              for n, p in model.named_parameters() if p.requires_grad]
    
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3))
    for ax, (name, weights) in zip(axes, params):
        ax.hist(weights, bins=50)
        ax.set_title(name, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_loss_landscape(model, loss_fn, x, y, steps=20, scale=0.01):
    import matplotlib.pyplot as plt
    import copy

    orig = [p.data.clone() for p in model.parameters()]
    dx = [torch.randn_like(p) for p in model.parameters()]
    dy = [torch.randn_like(p) for p in model.parameters()]

    alphas = torch.linspace(-1, 1, steps)
    betas  = torch.linspace(-1, 1, steps)
    Z = np.zeros((steps, steps))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for p, o, u, v in zip(model.parameters(), orig, dx, dy):
                p.data = o + scale * a * u + scale * b * v
            with torch.no_grad():
                Z[i, j] = loss_fn(model(x), y).item()

    # restore
    for p, o in zip(model.parameters(), orig):
        p.data = o

    plt.figure(figsize=(6, 5))
    plt.contourf(alphas, betas, Z, levels=50, cmap="viridis")
    plt.colorbar(label="loss")
    plt.title("Loss Landscape")
    plt.show()

def plot_activations(model, x):
    import matplotlib.pyplot as plt
    activations = {}

    def hook(name):
        def fn(module, input, output):
            activations[name] = output.detach().cpu()
        return fn

    handles = [m.register_forward_hook(hook(name)) 
               for name, m in model.named_modules() if len(list(m.children())) == 0]
    
    with torch.no_grad():
        model(x)
    
    for h in handles:
        h.remove()

    fig, axes = plt.subplots(1, len(activations), figsize=(4 * len(activations), 3))
    for ax, (name, act) in zip(axes, activations.items()):
        ax.hist(act.numpy().flatten(), bins=50)
        ax.set_title(name, fontsize=8)
    plt.tight_layout()
    plt.show()
    

def plot_training_curves(loss_history, acc_history=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2 if acc_history else 1, figsize=(12, 4))
    axes = [axes] if acc_history is None else axes

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