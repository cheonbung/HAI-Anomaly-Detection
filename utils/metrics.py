import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

def flatten_data(data):
    # (samples, timesteps, features) -> (samples, features)
    # 마지막 타임스텝의 값만 사용
    if len(data.shape) == 3:
        return data[:, -1, :]
    return data

def calculate_reconstruction_error(model, data):
    preds = model.predict(data, verbose=0)
    
    data_flat = flatten_data(data)
    preds_flat = flatten_data(preds)
    
    mse = np.mean(np.power(data_flat - preds_flat, 2), axis=1)
    return mse

def apply_moving_average(mse, window_size):
    return pd.Series(mse).rolling(window=window_size).mean().fillna(0).values

def find_best_threshold(mse, true_labels, threshold_range):
    best_metrics = {'threshold': 0, 'f1': 0}
    
    for threshold in threshold_range:
        pred_labels = (mse > threshold).astype(int)
        f1 = f1_score(true_labels, pred_labels)
        
        if f1 > best_metrics['f1']:
            best_metrics = {
                'threshold': threshold,
                'f1': f1,
                'precision': precision_score(true_labels, pred_labels),
                'recall': recall_score(true_labels, pred_labels),
                'accuracy': accuracy_score(true_labels, pred_labels),
                'roc_auc': roc_auc_score(true_labels, mse)
            }
            
    return best_metrics