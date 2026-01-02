import torch
import torch.nn as nn
from .modules import Encoder, Decoder

class USAD(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
  
    def forward(self, batch):
        # batch shape: (Batch, Window * Features)
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1)) # Adversarial Path
        return w1, w2, w3

    def training_step(self, batch, epoch):
        # USAD Specific Adversarial Training Loss
        w1, w2, w3 = self(batch)
        
        # Loss for AE1
        # 1/n * (X - w1)^2 + (1 - 1/n) * (X - w3)^2
        loss1 = (1.0 / epoch) * torch.mean((batch - w1)**2) + (1.0 - 1.0 / epoch) * torch.mean((batch - w3)**2)
        
        # Loss for AE2 (Adversarial)
        # 1/n * (X - w2)^2 - (1 - 1/n) * (X - w3)^2
        loss2 = (1.0 / epoch) * torch.mean((batch - w2)**2) - (1.0 - 1.0 / epoch) * torch.mean((batch - w3)**2)
        
        return loss1, loss2