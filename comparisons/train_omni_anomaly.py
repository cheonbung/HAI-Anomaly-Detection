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

# 프로젝트 루트 경로 추가 (utils import 용)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.preprocessing import get_preprocessed_data
from utils.metrics import find_best_threshold, apply_moving_average
from comparisons.models.omni_anomaly.omni_anomaly import OmniAnomaly

# --- Config & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('configs/config.json') as f:
    config = json.load(f)

# OmniAnomaly specific overrides
config['batch_size'] = 256
config['epochs'] = 30
config['window_size'] = 60
config['latent_dim'] = 3
config['hidden_dim'] = 500

output_dir = os.path.join(config['output_dir'], 'omni_anomaly')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Data Loading ---
print("Loading Data...")
tf_train, tf_val, tf_test1, tf_test2 = get_preprocessed_data(config)

def to_torch_loader(tf_tensor, batch_size, shuffle=False):
    data_np = tf_tensor.numpy()
    tensor_x = torch.from_numpy(data_np).float()
    dataset = TensorDataset(tensor_x) # AE-style: Input only
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), data_np.shape[2]

train_loader, n_features = to_torch_loader(tf_train, config['batch_size'], shuffle=True)
val_loader, _ = to_torch_loader(tf_val, config['batch_size'], shuffle=False)
test1_loader, _ = to_torch_loader(tf_test1, config['batch_size'], shuffle=False)
test2_loader, _ = to_torch_loader(tf_test2, config['batch_size'], shuffle=False)

# --- 2. Model Init ---
model = OmniAnomaly(
    input_dim=n_features,
    window_size=config['window_size'],
    hidden_dim=config['hidden_dim'],
    latent_dim=config['latent_dim'],
    device=device
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# --- 3. Training ---
print("Starting Training...")
history = {'train_loss': [], 'val_loss': []}

for epoch in range(config['epochs']):
    model.train()
    epoch_loss = 0
    
    for (x,) in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x = x.to(device)
        optimizer.zero_grad()
        
        recon_x, mu, log_var = model(x)
        loss = model.loss_function(recon_x, x, mu, log_var, kld_weight=0.001)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader.dataset)
    history['train_loss'].append(avg_loss)
    
    # Validation
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)
            recon_x, mu, log_var = model(x)
            loss = model.loss_function(recon_x, x, mu, log_var, kld_weight=0.001)
            val_epoch_loss += loss.item()
    
    avg_val_loss = val_epoch_loss / len(val_loader.dataset)
    history['val_loss'].append(avg_val_loss)
    
    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    torch.save(model.state_dict(), os.path.join(output_dir, "omni_anomaly_best.pth"))

# Plot Loss
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.title('OmniAnomaly Training Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# --- 4. Evaluation ---
print("Evaluating...")
model.eval()

def get_anomaly_scores(loader):
    scores = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon_x, _, _ = model(x)
            # Reconstruction Probability Approximation using MSE
            # Original paper uses Monte Carlo sampling for prob, MSE is a fast approximation
            error = torch.mean((recon_x - x) ** 2, dim=2) # (Batch, Window)
            score = torch.mean(error, dim=1) # Mean over window
            scores.append(score.cpu().numpy())
    return np.concatenate(scores)

label1 = pd.read_csv(config['data_dir'] + 'label-test1.csv')['label'][config['window_size']-1:].values
label2 = pd.read_csv(config['data_dir'] + 'label-test2.csv')['label'][config['window_size']-1:].values

for name, loader, true_labels in [("Test1", test1_loader, label1), ("Test2", test2_loader, label2)]:
    raw_scores = get_anomaly_scores(loader)
    scores_smooth = apply_moving_average(raw_scores, config['moving_average_window'])
    
    threshold_range = np.arange(0, np.max(scores_smooth), config['threshold_step'] * 10)
    metrics = find_best_threshold(scores_smooth, true_labels, threshold_range)
    
    print(f"[{name}] Results:")
    print(f"  Best F1: {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    
    np.save(os.path.join(output_dir, f"{name}_scores.npy"), scores_smooth)

print("OmniAnomaly Experiments Completed.")