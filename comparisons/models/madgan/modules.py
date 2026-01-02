import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers=2):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (Batch, Window, Latent_Dim)
        out, _ = self.lstm(z)
        out = self.linear(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1
        )
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Window, Features)
        out, _ = self.lstm(x)
        out = self.linear(out)
        return self.sigmoid(out)