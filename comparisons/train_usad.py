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
from comparisons.models.usad.usad import USAD

# --- Config & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('configs/config.json') as f:
    config = json.load(f)

# USAD Specific Config
config['batch_size'] = 4096  # USAD papers often use large batch sizes
config['epochs'] = 50
config['window_size'] = 10
config['latent_size'] = 100  # Dimension of z

output_dir = os.path.join(config['output_dir'], 'usad')
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
val_loader, _ = to_torch_loader(tf_val, config['batch_size'], shuffle=False)
test1_loader, _ = to_torch_loader(tf_test1, config['batch_size'], shuffle=False)
test2_loader, _ = to_torch_loader(tf_test2, config['batch_size'], shuffle=False)

# Input dimension for USAD is Flattened Window (Window * Features)
input_size = config['window_size'] * n_features

# --- 2. Model Init ---
model = USAD(
    w_size=input_size,
    z_size=config['latent_size']
).to(device)

# USAD uses two optimizers
optimizer1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=config['learning_rate'])
optimizer2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=config['learning_rate'])

# --- 3. Training Loop ---
print("Starting Training...")
history = {'loss1': [], 'loss2': []}

for epoch in range(config['epochs']):
    model.train()
    epoch_loss1, epoch_loss2 = [], []
    
    for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # Flatten Input: (Batch, Window, Feats) -> (Batch, Window*Feats)
        batch = batch.view(batch.size(0), -1).to(device)
        
        # Train AE1
        loss1, _ = model.training_step(batch, epoch + 1)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        # Train AE2
        _, loss2 = model.training_step(batch, epoch + 1)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        
        epoch_loss1.append(loss1.item())
        epoch_loss2.append(loss2.item())
        
    avg_loss1 = np.mean(epoch_loss1)
    avg_loss2 = np.mean(epoch_loss2)
    history['loss1'].append(avg_loss1)
    history['loss2'].append(avg_loss2)
    
    print(f"Epoch {epoch+1} - Loss1: {avg_loss1:.6f}, Loss2: {avg_loss2:.6f}")
    
    torch.save(model.state_dict(), os.path.join(output_dir, "usad_best.pth"))

# Plot Loss
plt.plot(history['loss1'], label='Loss1')
plt.plot(history['loss2'], label='Loss2')
plt.title('USAD Training Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# --- 4. Evaluation ---
print("Evaluating...")
model.eval()

def get_anomaly_scores(loader, alpha=0.5, beta=0.5):
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.view(batch.size(0), -1).to(device)
            
            w1 = model.decoder1(model.encoder(batch))
            w2 = model.decoder2(model.encoder(w1))
            
            # Reconstruction Error
            # Alpha * Recon(AE1) + Beta * Recon(Adversarial AE2)
            score = alpha * torch.mean((batch - w1)**2, axis=1) + beta * torch.mean((batch - w2)**2, axis=1)
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

print("USAD Experiments Completed.")