import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.preprocessing import get_preprocessed_data
from utils.metrics import find_best_threshold, apply_moving_average
from comparisons.models.madgan.madgan import MADGAN

# --- Config & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('configs/config.json') as f:
    config = json.load(f)

# MAD-GAN Specific Config
config['batch_size'] = 64
config['epochs'] = 30
config['window_size'] = 60
config['latent_dim'] = 15 # Latent Dimension
config['hidden_dim'] = 128

output_dir = os.path.join(config['output_dir'], 'madgan')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Data Loading ---
print("Loading Data...")
tf_train, tf_val, tf_test1, tf_test2 = get_preprocessed_data(config)

def to_torch_loader(tf_tensor, batch_size, shuffle=False):
    data_np = tf_tensor.numpy()
    tensor_x = torch.from_numpy(data_np).float()
    dataset = TensorDataset(tensor_x) 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), data_np.shape[2]

train_loader, n_features = to_torch_loader(tf_train, config['batch_size'], shuffle=True)
# GANs don't typically use validation sets for early stopping in the same way, but useful for monitoring
val_loader, _ = to_torch_loader(tf_val, config['batch_size'], shuffle=False)
test1_loader, _ = to_torch_loader(tf_test1, config['batch_size'], shuffle=False)
test2_loader, _ = to_torch_loader(tf_test2, config['batch_size'], shuffle=False)

# --- 2. Model Init ---
model = MADGAN(
    n_features=n_features,
    window_size=config['window_size'],
    latent_dim=config['latent_dim'],
    hidden_dim=config['hidden_dim'],
    device=device
)

# Optimizers
opt_g = torch.optim.Adam(model.generator.parameters(), lr=0.0001)
opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=0.0001)

criterion = nn.BCELoss()

# --- 3. Training Loop ---
print("Starting Training...")
history = {'g_loss': [], 'd_loss': []}

for epoch in range(config['epochs']):
    model.generator.train()
    model.discriminator.train()
    
    g_losses, d_losses = [], []
    
    for (real_x,) in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch_size = real_x.size(0)
        real_x = real_x.to(device)
        
        # Labels
        real_labels = torch.ones(batch_size, config['window_size'], 1).to(device)
        fake_labels = torch.zeros(batch_size, config['window_size'], 1).to(device)
        
        # =======================
        # Train Discriminator
        # =======================
        opt_d.zero_grad()
        
        # Real
        out_real = model.discriminator(real_x)
        d_loss_real = criterion(out_real, real_labels)
        
        # Fake
        z = torch.randn(batch_size, config['window_size'], config['latent_dim']).to(device)
        fake_x = model.generator(z)
        out_fake = model.discriminator(fake_x.detach()) # Detach G
        d_loss_fake = criterion(out_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_d.step()
        
        # =======================
        # Train Generator
        # =======================
        opt_g.zero_grad()
        
        # Generate Fake again (for gradient flow) - or reuse if memory allows but detach logic matters
        # Here we reuse fake_x graph but passing to D again
        out_fake_g = model.discriminator(fake_x)
        
        # Generator wants D to think these are Real
        g_loss = criterion(out_fake_g, real_labels)
        g_loss.backward()
        opt_g.step()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
    avg_g_loss = np.mean(g_losses)
    avg_d_loss = np.mean(d_losses)
    history['g_loss'].append(avg_g_loss)
    history['d_loss'].append(avg_d_loss)
    
    print(f"Epoch {epoch+1} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
    
    # Save both models
    torch.save({
        'generator': model.generator.state_dict(),
        'discriminator': model.discriminator.state_dict()
    }, os.path.join(output_dir, "madgan_best.pth"))

# Plot Loss
plt.plot(history['g_loss'], label='Generator')
plt.plot(history['d_loss'], label='Discriminator')
plt.title('MAD-GAN Training Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# --- 4. Evaluation ---
print("Evaluating (This may take time due to Z-optimization)...")

def get_anomaly_scores_madgan(loader):
    scores = []
    # Batch processing for optimization
    for (x,) in tqdm(loader, desc="Calculating DR-Scores"):
        x = x.to(device)
        score = model.compute_anomaly_score(x, lambda_p=0.5)
        scores.append(score.cpu().numpy())
    return np.concatenate(scores)

label1 = pd.read_csv(config['data_dir'] + 'label-test1.csv')['label'][config['window_size']-1:].values
label2 = pd.read_csv(config['data_dir'] + 'label-test2.csv')['label'][config['window_size']-1:].values

for name, loader, true_labels in [("Test1", test1_loader, label1), ("Test2", test2_loader, label2)]:
    raw_scores = get_anomaly_scores_madgan(loader)
    
    # Moving Average
    scores_smooth = apply_moving_average(raw_scores, config['moving_average_window'])
    
    # Thresholding
    threshold_range = np.arange(0, np.max(scores_smooth), config['threshold_step'] * 10)
    metrics = find_best_threshold(scores_smooth, true_labels, threshold_range)
    
    print(f"[{name}] Results:")
    print(f"  Best F1: {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    
    np.save(os.path.join(output_dir, f"{name}_scores.npy"), scores_smooth)

print("MAD-GAN Experiments Completed.")