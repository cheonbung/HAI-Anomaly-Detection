import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Generator, Discriminator

class MADGAN(nn.Module):
    """
    Wrapper class for Generator, Discriminator and Anomaly Scoring Logic
    """
    def __init__(self, n_features, window_size, latent_dim=15, hidden_dim=64, device='cpu'):
        super(MADGAN, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.n_features = n_features
        
        self.generator = Generator(latent_dim, hidden_dim, n_features).to(device)
        self.discriminator = Discriminator(n_features, hidden_dim).to(device)

    def compute_anomaly_score(self, x, lambda_p=0.5):
        """
        Calculates DR-Score:
        Score = lambda * Rec_Error + (1-lambda) * Disc_Score
        """
        self.generator.eval()
        self.discriminator.eval()
        
        # 1. Discriminator Score (Detection)
        with torch.no_grad():
            d_out = self.discriminator(x)
            # Take mean over time steps for the score
            discrimination_loss = torch.mean(-torch.log(d_out + 1e-8), dim=[1, 2])
            
        # 2. Reconstruction Loss (Optimization of Z)
        # Find optimal Z that generates X
        z = torch.randn(x.size(0), self.window_size, self.latent_dim, 
                        device=self.device, requires_grad=True)
        
        z_optimizer = torch.optim.Adam([z], lr=0.01)
        
        # Iteratively optimize Z
        for _ in range(50): # Steps for Z optimization
            z_optimizer.zero_grad()
            gen_x = self.generator(z)
            
            # Reconstruction Error
            rec_loss = torch.mean(torch.abs(x - gen_x), dim=[1, 2]) # L1 norm sum/mean
            
            # Similarity Loss (Discriminator Feature Matching - Simplified here to Rec Loss)
            # Original MAD-GAN uses combined loss for Z optimization, here we use Rec Loss
            
            total_z_loss = rec_loss.sum()
            total_z_loss.backward()
            z_optimizer.step()
            
        with torch.no_grad():
            gen_x_opt = self.generator(z)
            reconstruction_loss = torch.mean(torch.abs(x - gen_x_opt), dim=[1, 2])
            
        # 3. Combine Scores
        # Normalize to balance scales if necessary, here we use raw combination
        total_score = lambda_p * reconstruction_loss + (1 - lambda_p) * discrimination_loss
        
        return total_score