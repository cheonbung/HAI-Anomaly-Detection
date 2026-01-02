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
from comparisons.models.daemon.daemon import DAEMON
from comparisons.models.daemon.modules import Discriminator

# --- Config & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('configs/config.json') as f:
    config = json.load(f)

# DAEMON Specific Config
config['batch_size'] = 64
config['epochs'] = 30
config['window_size'] = 32 # Needs to be power of 2 for Conv layers ideally

output_dir = os.path.join(config['output_dir'], 'daemon')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Data Loading ---
print("Loading Data...")
tf_train, tf_val, tf_test1, tf_test2 = get_preprocessed_data(config)

def to_torch_loader(tf_tensor, batch_size, shuffle=False):
    data_np = tf_tensor.numpy()
    # Permute for Conv1D: (Batch, Window, Feats) -> (Batch, Feats, Window)
    tensor_x = torch.from_numpy(data_np).float().permute(0, 2, 1)
    dataset = TensorDataset(tensor_x) 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), data_np.shape[2]

train_loader, n_features = to_torch_loader(tf_train, config['batch_size'], shuffle=True)
val_loader, _ = to_torch_loader(tf_val, config['batch_size'], shuffle=False)
test1_loader, _ = to_torch_loader(tf_test1, config['batch_size'], shuffle=False)
test2_loader, _ = to_torch_loader(tf_test2, config['batch_size'], shuffle=False)

# --- 2. Model Init ---
model = DAEMON(
    n_features=n_features,
    window_size=config['window_size'],
    z_dim=32,
    device=device
).to(device)

# Optimizers
opt_G = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.001, betas=(0.5, 0.999))
opt_D_Rec = torch.optim.Adam(model.d_rec.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_D_Lat = torch.optim.Adam(model.d_lat.parameters(), lr=0.0001, betas=(0.5, 0.999))

criterion_ADV = nn.BCELoss() # Adversarial Loss
criterion_REC = nn.MSELoss() # Reconstruction Loss
criterion_LAT = nn.MSELoss() # Latent Feature Matching Loss

# --- 3. Training Loop ---
print("Starting Training...")
history = {'g_loss': [], 'd_rec_loss': [], 'd_lat_loss': []}

for epoch in range(config['epochs']):
    model.train()
    g_losses, dr_losses, dl_losses = [], [], []
    
    for (real_x,) in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        real_x = real_x.to(device)
        batch_size = real_x.size(0)
        
        # Generate Fake
        recon_x, z, _, _ = model(real_x)
        
        # ---------------------
        #  Train Discriminators
        # ---------------------
        
        # D_Rec (Reconstruction)
        opt_D_Rec.zero_grad()
        pred_real, _ = model.d_rec(real_x)
        pred_fake, _ = model.d_rec(recon_x.detach())
        
        loss_dr_real = criterion_ADV(pred_real, torch.ones_like(pred_real))
        loss_dr_fake = criterion_ADV(pred_fake, torch.zeros_like(pred_fake))
        loss_dr = (loss_dr_real + loss_dr_fake) * 0.5
        loss_dr.backward()
        opt_D_Rec.step()
        
        # D_Lat (Latent)
        opt_D_Lat.zero_grad()
        real_z = torch.randn_like(z).to(device) # Prior (Gaussian)
        pred_lat_real, _ = model.d_lat(real_z)
        pred_lat_fake, _ = model.d_lat(z.detach())
        
        loss_dl_real = criterion_ADV(pred_lat_real, torch.ones_like(pred_lat_real))
        loss_dl_fake = criterion_ADV(pred_lat_fake, torch.zeros_like(pred_lat_fake))
        loss_dl = (loss_dl_real + loss_dl_fake) * 0.5
        loss_dl.backward()
        opt_D_Lat.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        opt_G.zero_grad()
        
        # Adversarial Loss (Fool D_Rec & D_Lat)
        pred_fake_rec, feat_fake_rec = model.d_rec(recon_x)
        pred_fake_lat, feat_fake_lat = model.d_lat(z)
        
        # Feature Matching Loss (Optional but recommended in DAEMON)
        # Here simplified to standard GAN loss for stability
        loss_adv_rec = criterion_ADV(pred_fake_rec, torch.ones_like(pred_fake_rec))
        loss_adv_lat = criterion_ADV(pred_fake_lat, torch.ones_like(pred_fake_lat))
        
        loss_recon = criterion_REC(recon_x, real_x)
        
        # Total G Loss
        loss_g = loss_recon * 10 + loss_adv_rec + loss_adv_lat
        loss_g.backward()
        opt_G.step()
        
        g_losses.append(loss_g.item())
        dr_losses.append(loss_dr.item())
        dl_losses.append(loss_dl.item())
        
    print(f"Epoch {epoch+1} - G: {np.mean(g_losses):.4f}, D_Rec: {np.mean(dr_losses):.4f}, D_Lat: {np.mean(dl_losses):.4f}")
    
    torch.save(model.state_dict(), os.path.join(output_dir, "daemon_best.pth"))

# --- 4. Evaluation ---
print("Evaluating...")
model.eval()

def get_anomaly_scores(loader):
    scores = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon_x, _, _, _ = model(x)
            
            # Reconstruction Error per time step
            # x: (Batch, Feats, Window)
            error = torch.mean((recon_x - x) ** 2, dim=1) # (Batch, Window)
            score = torch.mean(error, dim=1) # Mean over window
            scores.append(score.cpu().numpy())
            
    return np.concatenate(scores)

label1 = pd.read_csv(config['data_dir'] + 'label-test1.csv')['label'][config['window_size']-1:].values
label2 = pd.read_csv(config['data_dir'] + 'label-test2.csv')['label'][config['window_size']-1:].values

for name, loader, true_labels in [("Test1", test1_loader, label1), ("Test2", test2_loader, label2)]:
    raw_scores = get_anomaly_scores(loader)
    
    # Moving Average Filter
    scores_smooth = apply_moving_average(raw_scores, config['moving_average_window'])
    
    # Thresholding
    threshold_range = np.arange(0, np.max(scores_smooth), config['threshold_step'] * 10)
    metrics = find_best_threshold(scores_smooth, true_labels, threshold_range)
    
    print(f"[{name}] Results:")
    print(f"  Best F1: {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    np.save(os.path.join(output_dir, f"{name}_scores.npy"), scores_smooth)

print("DAEMON Experiments Completed.")