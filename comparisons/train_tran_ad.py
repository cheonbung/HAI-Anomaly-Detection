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
from comparisons.models.tran_ad.tran_ad import TranAD

# --- Config & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('configs/config.json') as f:
    config = json.load(f)

# TranAD Specific Config
config['batch_size'] = 128
config['epochs'] = 10  # TranAD converges relatively fast
config['window_size'] = 10  # TranAD typically uses small windows (e.g., 10)

output_dir = os.path.join(config['output_dir'], 'tran_ad')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Data Loading ---
print("Loading Data...")
tf_train, tf_val, tf_test1, tf_test2 = get_preprocessed_data(config)

def to_torch_loader(tf_tensor, batch_size, shuffle=False):
    data_np = tf_tensor.numpy()
    tensor_x = torch.from_numpy(data_np).float()
    # TranAD input: Input Only (AutoEncoder style)
    dataset = TensorDataset(tensor_x) 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), data_np.shape[2]

train_loader, n_features = to_torch_loader(tf_train, config['batch_size'], shuffle=True)
# Validation set is not strictly used in TranAD original code training loop, but we keep it for monitoring
val_loader, _ = to_torch_loader(tf_val, config['batch_size'], shuffle=False)
test1_loader, _ = to_torch_loader(tf_test1, config['batch_size'], shuffle=False)
test2_loader, _ = to_torch_loader(tf_test2, config['batch_size'], shuffle=False)

# --- 2. Model Init ---
model = TranAD(
    feats=n_features,
    window_size=config['window_size'],
    device=device
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
criterion = nn.MSELoss(reduction='none')

# --- 3. Training Loop ---
print("Starting Training...")
history = {'train_loss': []}

for epoch in range(config['epochs']):
    model.train()
    epoch_losses = []
    n = epoch + 1  # Epoch index for weighting
    
    for (d,) in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # TranAD expects (Seq_Len, Batch, Feats)
        # d shape: (Batch, Window, Feats) -> Permute to (Window, Batch, Feats)
        window = d.permute(1, 0, 2).to(device)
        
        # Target element (usually the last element of the window)
        # shape: (1, Batch, Feats)
        elem = window[-1, :, :].unsqueeze(0)
        
        optimizer.zero_grad()
        
        x1, x2 = model(window, elem)
        
        # Loss calculation (Phase 1 & Phase 2 weighted combination)
        loss1 = criterion(x1, elem)
        loss2 = criterion(x2, elem)
        
        loss = (1 / n) * loss1 + (1 - 1 / n) * loss2
        loss = torch.mean(loss)
        
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
        
    scheduler.step()
    avg_loss = np.mean(epoch_losses)
    history['train_loss'].append(avg_loss)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}")
    
    torch.save(model.state_dict(), os.path.join(output_dir, "tran_ad_best.pth"))

# Plot Loss
plt.plot(history['train_loss'], label='Train Loss')
plt.title('TranAD Training Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# --- 4. Evaluation ---
print("Evaluating...")
model.eval()

def get_anomaly_scores(loader):
    scores = []
    with torch.no_grad():
        for (d,) in loader:
            window = d.permute(1, 0, 2).to(device)
            elem = window[-1, :, :].unsqueeze(0)
            
            x1, x2 = model(window, elem)
            
            # Reconstruction Error from Phase 2 (x2)
            loss = criterion(x2, elem) # (1, Batch, Feats)
            loss = loss[0] # (Batch, Feats)
            
            # Feature aggregation (Mean over features)
            score = torch.mean(loss, dim=1)
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

print("TranAD Experiments Completed.")