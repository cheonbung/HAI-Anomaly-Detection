import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, ndf, z_dim):
        super(Encoder, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv1d(input_dim, ndf, 4, 2, 1),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output to Latent (Mu, LogVar)
        # Note: Input length dependence. Assuming window size allows 3 downsamplings (div by 8)
        self.mu_conv = nn.Conv1d(ndf * 4, z_dim, 1) # Reduce channels
        self.logvar_conv = nn.Conv1d(ndf * 4, z_dim, 1)

    def forward(self, x):
        # x: (Batch, Features, Window) -> Transposed input
        h = self.main(x)
        mu = self.mu_conv(h)
        log_var = self.logvar_conv(h)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, z_dim, ngf, output_dim):
        super(Decoder, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose1d(z_dim, ngf * 4, 4, 2, 1), # Upsample 1
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1), # Upsample 2
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1), # Upsample 3
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(ngf, output_dim, 3, 1, 1), # Final adjustment (Keep size)
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, ndf):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv1d(input_dim, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Conv1d(ndf * 4, 1, 1) # Output probability map
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.main(x)
        out = self.sigmoid(self.classifier(h))
        return out, h # Return features for feature matching loss