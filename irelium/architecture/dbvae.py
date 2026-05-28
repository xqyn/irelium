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
from model.cnn import cnnConvNorm
from model.vae import cnnDecoder
from utils import as_tensor

#Path(__file__).parent.parent / "config" / "color" / "colors.yaml"

class DB_VAE(nn.Module):
    '''
    debiasing variational autoencoder (DB-VAE).

    Combines a CNN encoder, VAE reparameterization, and CNN decoder.
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
    
    def __init__(self,
                 H: int,
                 W: int,
                 latent_dim: int = 100,
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = cnnConvNorm(n_outputs=2 * latent_dim + 1, H=H, W=W)
        self.decoder = cnnDecoder()
        
    def encode(self,
               x: torch.Tensor) -> tuple:
        '''
        Encode input x → classification logit + latent distribution.
        x: Input image tensor [B, 3, H, W].
        Returns:
            y_logit:    Classification logit    [B, 1].
            z_mean:     Latent mean             [B, latent_dim].
            z_logsigma: Latent log std dev       [B, latent_dim].
        '''
        # encoder outputs flat tensor [B, 2*latent_dim+1]
        encoder_output = self.encoder(x)
        
        # split into 3 parts:
        # [0]              → classification logit
        # [1:latent_dim+1] → latent mean
        # [latent_dim+1:]  → latent log std dev
        # classification prediction
        y_logit = encoder_output[:, 0].unsqueeze(-1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1 : self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1 :]
        return y_logit, z_mean, z_logsigma
    
    def predict(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        '''
        classification only — skips decoder for inference speed.

        x: Input image tensor [B, 3, H, W].
        Returns:
        y_logit: Classification logit [B, 1].
        '''
        y_logit, _, _ = self.encode(x)
        return y_logit
    
    def forward(
        self,
        x: torch.Tensor
    ) -> tuple:
        '''
        Full forward pass: encode → reparameterize → decode.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            y_logit:    Classification logit        [B, 1].
            z_mean:     Latent mean                 [B, latent_dim].
            z_logsigma: Latent log std dev           [B, latent_dim].
            recon:      Reconstructed image          [B, 3, H, W].
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
    images: torch.Tensor,
    model: nn.Module,
    batch_size: int
) -> np.ndarray:
    '''
    Extract latent means for all images without loading all into GPU at once.

    Args:
        images:     Input images [N, C, H, W].
        model:      Trained DB_VAE model.
        batch_size: Number of images per forward pass.

    Returns:
        z_mean: Latent means [N, latent_dim].
    '''
    # turn model to eval mode
    model.eval()
    
    # transfer image to model's device
    images_t = as_tensor(images, model=model)
    assert images_t.shape[-1] in (1,3), \
        f'Expected channels-last [N,H,W,C], got shape {images_t.shape}'
    
    all_z_mean = []
    
    with torch.inference_mode():
        for start in range(0, len(images_t), batch_size):
            batch = image_t[start:start + batch_size].permute(0, 3, 1, 2)
            _, z_mean, _ = model.encode(batch)
            all_z_mean.append(z_mean.cpu())
    
    # concatenate all partial z_mean
    mu = torch.cat(all_z_mean, dim=0).numpy()
    return mu
    

# --- get training sample based on sample probability
def get_train_sample_probability(
    images: np.ndarray | torch.Tensor,
    model: DB_VAE,
    bins: int = 10,
    batch_size: int = 64,
    smoothing_fac: float = 0.001,
) -> np.array:
    '''
    Function that recomputes the sampling probabilities for images within a batch
    based on how they distribute across the training data
    '''
    
    latent_dim = model.latent_dim
    
    # run input batch and get latent variabe means
    images = as_tensor(images, model=model)
    mu = get_latent_mu(images, model=model, batch_size=batch_size)
    
    # sample probabilities for the images
    training_sample_p = np.zeros(mu.shape[0], dtype=np.float64)
    
    # consider the distribution for each latent variable
    for i in range(latent_dim):
        latent_distribution = mu[:, i]
        
        # histogram distribution over this latent dimension
        hist_density, bin_edges = np.histogram(
            latent_distribution,
            density=True,
            bins=bins
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
        
        sample_density = np.array([hist_smoothed_density[idx - 1] for idx in bin_idx])
        # or fancy index
        # sample_density = hist_smoothed_density[bin_idx - 1]
        prob_den = 1.0 / sample_density
        prob_den /= prob_den.sum()
        
        # update sampling probabilities by considering whether the newly
        #     computed p is greater than the existing sampling probabilities
        training_sample_p = np.maximum(training_sample_p, prob_den)
    
    # final normalization
    training_sample_p /= np.sum(training_sample_p)
    
    return training_sample_p

    