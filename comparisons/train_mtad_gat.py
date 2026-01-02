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

# 상위 폴더(루트) 경로 추가하여 기존 utils import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.preprocessing import get_preprocessed_data, load_raw_data
from utils.metrics import find_best_threshold, apply_moving_average
from comparisons.models.mtad_gat.mtad_gat import MTAD_GAT

# --- Config & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('configs/config.json') as f:
    config = json.load(f)

# Config overrides for MTAD-GAT specific
config['batch_size'] = 256
config['epochs'] = 30  # Adjust as needed
config['val_split'] = 0.2
config['window_size'] = 60 # MTAD-GAT usually works well with larger windows

# Output Directories
output_dir = os.path.join(config['output_dir'], 'mtad_gat')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Data Loading & Adaptation (TF -> PyTorch) ---
print("Loading and Preprocessing Data (using utils.preprocessing)...")
# TF Tensor로 반환됨: (Samples, Window, Features)
tf_train, tf_val, tf_test1, tf_test2 = get_preprocessed_data(config)

def to_torch_loader(tf_tensor, batch_size, shuffle=False):
    # Convert TF Tensor -> Numpy -> PyTorch Tensor
    data_np = tf_tensor.numpy()
    tensor_x = torch.from_numpy(data_np).float()
    
    # MTAD-GAT requires (x, y) where y is usually next step for forecasting
    # Here we simulate auto-regression: Input X, Target X (Reconstruction) + Next X (Forecast)
    # But for simplicity in anomaly detection, we often use X as target for reconstruction.
    # Original MTAD implementation uses specific forecasting horizon.
    # We will pass X as both input and target for reconstruction loss.
    
    dataset = TensorDataset(tensor_x, tensor_x) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, data_np.shape[2] # return loader and feature dim

train_loader, n_features = to_torch_loader(tf_train, config['batch_size'], shuffle=True)
val_loader, _ = to_torch_loader(tf_val, config['batch_size'], shuffle=False)
test1_loader, _ = to_torch_loader(tf_test1, config['batch_size'], shuffle=False)
test2_loader, _ = to_torch_loader(tf_test2, config['batch_size'], shuffle=False)

print(f"Data Loaded. Features: {n_features}, Window: {config['window_size']}")

# --- 2. Model Initialization ---
model = MTAD_GAT(
    n_features=n_features,
    window_size=config['window_size'],
    out_dim=n_features,
    kernel_size=7,
    use_gatv2=True,
    dropout=0.2,
    device=device
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
forecast_criterion = nn.MSELoss()
recon_criterion = nn.MSELoss()

# --- 3. Training Loop ---
print("Starting Training...")
history = {'train_loss': [], 'val_loss': []}

for epoch in range(config['epochs']):
    model.train()
    train_losses = []
    
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
        x = x.to(device)
        y = y.to(device) # Target is same as input for reconstruction
        
        optimizer.zero_grad()
        
        preds, recons = model(x)
        
        # MTAD-GAT Logic: Combine Forecast Loss and Reconstruction Loss
        # Forecast target: usually next step, but here we simplify to reconstruction task or self-prediction
        # To align with original paper perfectly, one should shift y, but for AE-style, x=y is common.
        
        loss_forecast = torch.sqrt(forecast_criterion(y, preds))
        loss_recon = torch.sqrt(recon_criterion(x, recons))
        loss = loss_forecast + loss_recon
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
    avg_train_loss = np.mean(train_losses)
    history['train_loss'].append(avg_train_loss)
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            preds, recons = model(x)
            loss = torch.sqrt(forecast_criterion(y, preds)) + torch.sqrt(recon_criterion(x, recons))
            val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses)
    history['val_loss'].append(avg_val_loss)
    
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save Checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, "mtad_gat_best.pth"))

# Plot History
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.title('MTAD-GAT Training Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# --- 4. Evaluation ---
print("Evaluating on Test Sets...")
model.eval()

# Helper to calculate Anomaly Score
def get_anomaly_scores(loader):
    scores = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds, recons = model(x)
            
            # Score = (Pred - True)^2 + Gamma * (Recon - True)^2
            # Gamma usually 1.0
            gamma = 1.0
            
            error_forecast = torch.mean((preds - y) ** 2, dim=2)
            error_recon = torch.mean((recons - x) ** 2, dim=2)
            
            score = error_forecast + gamma * error_recon
            scores.append(score.cpu().numpy())
    return np.concatenate(scores)

# Load Labels
label1 = pd.read_csv(config['data_dir'] + 'label-test1.csv')['label']
label2 = pd.read_csv(config['data_dir'] + 'label-test2.csv')['label']

# Adjust labels for windowing
label1 = label1[config['window_size']-1:].values
label2 = label2[config['window_size']-1:].values

for name, loader, true_labels in [("Test1", test1_loader, label1), ("Test2", test2_loader, label2)]:
    print(f"Evaluating {name}...")
    raw_scores = get_anomaly_scores(loader)
    
    # 1. Flatten if necessary (average over window)
    # shape: (Samples, Window). We take the mean over the window or just the last point
    scores_flat = np.mean(raw_scores, axis=1) 
    
    # 2. Moving Average
    scores_smooth = apply_moving_average(scores_flat, config['moving_average_window'])
    
    # 3. Find Best Threshold
    threshold_range = np.arange(0, np.max(scores_smooth), config['threshold_step'] * 10)
    metrics = find_best_threshold(scores_smooth, true_labels, threshold_range)
    
    print(f"[{name}] Results:")
    print(f"  Best F1: {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    # Save scores
    np.save(os.path.join(output_dir, f"{name}_scores.npy"), scores_smooth)

print("MTAD-GAT Experiments Completed.")