'''
irelium
XQ
2026-05-20
dbvae
'''

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from irelium.architecture.backbone import ConvBackbone
from irelium.architecture.decoder import Decoder
from irelium.utils import as_tensor, load_config

_CFG = load_config("dbvae")


class DB_VAE(nn.Module):
    '''
    Debiasing variational autoencoder (DB-VAE).

    Encoder outputs classification logit + latent distribution (mu, logsigma).
    Decoder reconstructs input from sampled latent vector z.

    Public API:
        forward(x)  → y_logit, z_mean, z_logsigma, recon
        encode(x)   → y_logit, z_mean, z_logsigma
        predict(x)  → y_logit

    Args:
        latent_dim:  Dimension of latent space z.
        base_filter: Base filter count for encoder and decoder.
    '''
    
    def __init__(
        self,
        latent_dim: int,
        base_filter: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvBackbone(
            n_outputs=2 * latent_dim + 1,
            H=_CFG.model.H,
            W=_CFG.model.W,
            in_channels=_CFG.model.in_channels,
            base_filter=base_filter,
            channel_schedule=_CFG.encoder.channel_schedule,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_filter=base_filter,
        )
        
    def encode(self, x: torch.Tensor) -> tuple:
        '''
        Encode input → classification logit + latent distribution.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            y_logit:    Classification logit    [B, 1].
            z_mean:     Latent mean             [B, latent_dim].
            z_logsigma: Latent log std dev       [B, latent_dim].
        '''
        encoder_output = self.encoder(x)  # [B, 2*latent_dim+1]
        
        # [0]              → classification logit
        # [1:latent_dim+1] → latent mean
        # [latent_dim+1:]  → latent log std dev
        y_logit    = encoder_output[:, 0].unsqueeze(-1)
        z_mean     = encoder_output[:, 1 : self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1 :]
        return y_logit, z_mean, z_logsigma
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Classification only — skips decoder for inference speed.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            y_logit: Classification logit [B, 1].
        '''
        y_logit, _, _ = self.encode(x)
        return y_logit
    
    def forward(self, x: torch.Tensor) -> tuple:
        '''
        Full forward pass: encode → reparameterize → decode.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            y_logit:    Classification logit    [B, 1].
            z_mean:     Latent mean             [B, latent_dim].
            z_logsigma: Latent log std dev       [B, latent_dim].
            recon:      Reconstructed image      [B, 3, H, W].
        '''
        y_logit, z_mean, z_logsigma = self.encode(x)
        z_reparam = self._reparameterize(z_mean, z_logsigma)
        recon = self._decode(z_reparam)
        return y_logit, z_mean, z_logsigma, recon
    
    def _reparameterize(
        self,
        z_mean: torch.Tensor,
        z_logsigma: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Reparameterization trick — sample z from latent distribution.
        Private: only called internally by forward().
        # define reparameterization computation
        # REPARAMETERIZATION TRICK:
        #
        #   z = mu + exp(0.5 * log_sigma) * epsilon
        #
        #   where epsilon ~ N(0, I)  (random noise)
        #         mu               (learned mean)
        #         log_sigma        (learned log std dev)


        z_mean:      Latent mean        [B, latent_dim].
        z_logsigma:  Latent log std dev  [B, latent_dim].

        Returns:
            z: Sampled latent vector [B, latent_dim].
        '''
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_logsigma) * epsilon
    
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Reconstructed image [B, 3, H, W].
        '''
        reconstruction = self.decoder(z)
        return reconstruction



    def _decode(
            self,
            z: torch.Tensor
        ) -> torch.Tensor:
        '''
        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Reconstructed image [B, 3, H, W].
        '''
        reconstruction = self.decoder(z)
        return reconstruction


# --- get latent mean distribution
# training data is biased:
# For eg:
# 80% light skin faces   → model sees these constantly → learns them well
# 20% dark skin faces    → model sees these rarely     → learns them poorly
# this function give rare faces higher probability of being sampled in each batch
# so model sees them more often → learns them equally well
# Latent space density:
# faces that cluster together → common    → low sampling weight
# faces that are isolated     → rare      → high sampling weight
# step 1 — feed ALL training images through encoder
# get z_mean for each image [N, latent_dim]

#   z_means = get_latent_mu(images, dbvae)
#   z_means[i] = where image i lives in latent space

# step 2 — estimate density at each point
# crowded region → high density  → common face  → low weight
# sparse region  → low density   → rare face    → high weight

# step 3 — use density as sampling probability
# rare faces sampled more often → model sees them more → bias reduced 

def get_latent_mu(
    images: np.ndarray | torch.Tensor,
    model: nn.Module,
    batch_size: int,
) -> np.ndarray:
    '''
    Extract latent means for all images in batches.

    Args:
        images:     Input images [N, H, W, C] channels-last.
        model:      Trained DB_VAE — must be in eval mode.
        batch_size: Number of images per forward pass.

    Returns:
        z_mean: Latent means [N, latent_dim].

    Raises:
        ValueError: If images are not channels-last with C in (1, 3).
    '''
   
    # transfer image to model's device
    images_t = as_tensor(images, model=model)

    if images_t.ndim != 4 or images_t.shape[-1] not in (1, 3):
        raise ValueError(
            f"Expected channels-last [N, H, W, C] with C in (1, 3), "
            f"got shape {images_t.shape}"
        )
    
    all_z_mean = []
    
    with torch.inference_mode():
        for start in range(0, len(images_t), batch_size):
            batch = images_t[start : start + batch_size].permute(0, 3, 1, 2)
            _, z_mean, _ = model.encode(batch)
            all_z_mean.append(z_mean.cpu())

    return torch.cat(all_z_mean, dim=0).numpy()
    

# --- get training sample based on sample probability
def get_train_sample_probability(
    images: np.ndarray | torch.Tensor,
    model: DB_VAE,
    bins: int = 10,
    batch_size: int = 64,
    smoothing_fac: float = 0.001,
) -> np.ndarray:
    '''
    Compute per-image sampling probability inversely proportional to
    latent space density — rare samples get higher weight.

    For each latent dimension: histogram density → inverse → max across dims.
    Ensures underrepresented faces are sampled more often during training.

    Args:
        images:        Input images [N, H, W, C].
        model:         Trained DB_VAE — must be in eval mode.
        bins:          Histogram bins per latent dimension.
        batch_size:    Batch size for latent extraction.
        smoothing_fac: Additive smoothing to avoid zero density.

    Returns:
        training_sample_p: Sampling probabilities [N], sums to 1.
    '''
    model.eval()
    images = as_tensor(images, model=model)
    mu = get_latent_mu(images, model=model, batch_size=batch_size)
    training_sample_p = np.zeros(mu.shape[0], dtype=np.float64)

    # consider the distribution for each latent variable
    for i in range(model.latent_dim):
        latent_distribution = mu[:, i]
        
        # histogram distribution over this latent dimension
        hist_density, bin_edges = np.histogram(
            latent_distribution,
            density=True,
            bins=bins,
        )
        
        # smooth to avoid zero density → zero division
        hist_smoothed_density = hist_density + smoothing_fac
        
        # normalize into probabilities
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)
        
        # extend edges to catch all samples
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')
        
        # find which bin each sample falls into
        bin_idx = np.digitize(latent_distribution, bin_edges)
        
        # inverse density — rare samples get high weight
            # Common samples → small weight
            # Rare samples → large weight
        
        # fancy index
        sample_density = hist_smoothed_density[bin_idx - 1]
        # equivalent:
        #sample_density = np.array([hist_smoothed_density[idx - 1] for idx in bin_idx])
        
        prob_den = 1.0 / sample_density
        prob_den /= prob_den.sum()
        
        # update sampling probabilities by considering whether the newly
        #     computed p is greater than the existing sampling probabilities
        training_sample_p = np.maximum(training_sample_p, prob_den)
    
    # final normalization
    training_sample_p /= np.sum(training_sample_p)
    
    return training_sample_p

    