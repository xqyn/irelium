'''
irelium
XQ
2026-05-20
loss compute
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DB-VAE -----------------------------------------------
# --- VAE loss
def _vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    kl_weight: float = 0.0005,
) -> torch.Tensor:
    '''
    VAE loss = reconstruction loss + KL divergence.
    
    Latent loss (d_kl) KL DIVERGENCE LOSS:  measures how far latent distribution is from N(0, 1)
    
              1   k-1
       L_KL = ─ * SUM  [ sigma_j + mu_j^2 - 1 - log(sigma_j) ]
              2   j=0

    RECONSTRUCTION LOSS:
    
      L_x(x, x_hat) = || x - x_hat ||_1
    
      L1 norm — sum of absolute pixel differences
      penalizes every pixel equally
        # TOTAL VAE LOSS:
    
    L_VAE = c * L_KL + L_x(x, x_hat)
              ↑
              kl_weight — controls balance between:
              high c → forces latent space closer to N(0,1)
              low  c → prioritizes reconstruction quality
    
    Args:
        x:          Original input tensor    [B, C, H, W].
        x_recon:    Reconstructed tensor     [B, C, H, W].
        mu:         Encoded mean             [B, latent_dim].
        log_sigma:  Encoded log std dev      [B, latent_dim].
        kl_weight:  Weight on latent loss — controls disentanglement.

    Returns:
        Scalar loss tensor.
    '''
    latent_loss = 0.5 * torch.sum(torch.exp(log_sigma) + mu ** 2 -1 - log_sigma, dim=1)
    recon_loss = torch.mean(torch.abs(x - x_recon), dim=(1, 2, 3))
    vae_loss = kl_weight * latent_loss + recon_loss
    return vae_loss

# --- classification loss
def _classification_loss(
    y_logit: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # classification loss — per sample [B], reduction='none' for weighting
    cls_loss = F.binary_cross_entropy_with_logits(
        y_logit, y, reduction='none'
    )
    return cls_loss


# --- total dbvae loss
def debiasing_loss(
    x: torch.Tensor,
    x_pred: torch.Tensor,
    y: torch.Tensor,
    y_logit: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Loss function for DB-VAE.

    Args:
        x:          True input                              [B, C, H, W].
        x_pred:     Reconstructed input                     [B, C, H, W].
        y:          True binary labels — face=1, nonface=0  [B, 1].
        y_logit:    Predicted logits                        [B, 1].
        mu:         Mean of latent distribution             [B, latent_dim].
        log_sigma:  Log std dev of latent distribution      [B, latent_dim].

    Returns:
        total_loss:          DB-VAE total loss scalar.
        classification_loss: Classification loss scalar.
    '''
    
    # vae loss — per sample [B]
    vae_loss = _vae_loss(
        x=x,
        x_recon=x_pred,
        mu=mu,
        log_sigma=log_sigma,
    )
    
    # classification loss — per sample [B], reduction='none' for weighting
    cls_loss = _classification_loss(
        y_logit=y_logit,
        y=y)
    
    # using train data labels to create a variable for indicator:
    indicate   = (y == 1.0).float().squeeze(-1)
    cls_loss   = cls_loss.squeeze(-1)
    
    # define DB-VAE total loss:
    total_loss = torch.mean(cls_loss + indicate * vae_loss)
    
    return total_loss, cls_loss