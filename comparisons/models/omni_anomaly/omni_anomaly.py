import torch
import torch.nn as nn
from .modules import GRUEncoder, GRUDecoder

class OmniAnomaly(nn.Module):
    def __init__(
        self, 
        input_dim, 
        window_size, 
        hidden_dim=500, 
        latent_dim=3, 
        num_layers=1,
        device='cpu'
    ):
        super(OmniAnomaly, self).__init__()
        self.window_size = window_size
        self.device = device
        
        self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = GRUDecoder(latent_dim, hidden_dim, input_dim, num_layers)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        mu, log_var = self.encoder(x)
        
        # Reparameterization
        z = self.reparameterize(mu, log_var)
        
        # Decoder
        recon_x = self.decoder(z, self.window_size)
        
        return recon_x, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, kld_weight=1.0):
        # Reconstruction Loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL Divergence Loss
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kld_weight * kld_loss