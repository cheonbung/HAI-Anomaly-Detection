import torch
import torch.nn as nn
from .modules import Encoder, Decoder, Discriminator

class DAEMON(nn.Module):
    def __init__(self, n_features, window_size, z_dim=32, ndf=32, ngf=32, device='cpu'):
        super(DAEMON, self).__init__()
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        
        # Generator (Encoder-Decoder)
        # Encoder Input: (Batch, Features, Window)
        self.encoder = Encoder(n_features, ndf, z_dim)
        
        # Decoder Input: (Batch, Z_Dim, Reduced_Window)
        # Note: Decoder structure needs to match Encoder's reduction
        # Here we simplify for compatibility. Original DAEMON uses specific sizes.
        self.decoder = Decoder(z_dim, ngf, n_features)
        
        # Discriminators
        # D_Rec: Discriminates Real Data vs Reconstructed Data
        self.d_rec = Discriminator(n_features, ndf)
        
        # D_Lat: Discriminates Latent Code vs Prior (Gaussian)
        self.d_lat = Discriminator(z_dim, ndf)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (Batch, Features, Window)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        
        # Adjust Output Size if necessary (padding/trimming due to ConvTranspose)
        if recon_x.shape[2] != x.shape[2]:
            recon_x = nn.functional.interpolate(recon_x, size=x.shape[2], mode='linear', align_corners=True)
            
        return recon_x, z, mu, log_var