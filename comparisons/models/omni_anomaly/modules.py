import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_var = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # x: (Batch, Window, Features)
        _, h_n = self.gru(x)
        # Use the last hidden state
        h_n = h_n[-1] 
        mu = self.linear_mu(h_n)
        log_var = self.linear_var(h_n)
        return mu, log_var

class GRUDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=1):
        super(GRUDecoder, self).__init__()
        self.linear_input = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, z, window_size):
        # z: (Batch, Latent)
        # Expand z to match time steps for decoder input
        # (Batch, Window, Hidden)
        hidden_input = self.linear_input(z).unsqueeze(1).repeat(1, window_size, 1)
        
        output, _ = self.gru(hidden_input)
        reconstruction = self.linear_out(output)
        return reconstruction