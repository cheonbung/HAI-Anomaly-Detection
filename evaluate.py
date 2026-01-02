import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.preprocessing import load_data, normalize_data, apply_window
from utils.metrics import calculate_reconstruction_error, apply_moving_average, find_best_threshold
from models.layers import AttentionLayer, FeatureAttentionLayer
from tqdm import tqdm

def main():
    with open('configs/config.json') as f:
        config = json.load(f)
        
    print("Loading Test Data...")
    train_df, test1_df, test2_df = load_data(config['data_dir'])
    
    # Label 로드
    label1 = pd.read_csv(config['data_dir'] + 'label-test1.csv')['label']
    label2 = pd.read_csv(config['data_dir'] + 'label-test2.csv')['label']
    
    # 정규화
    _, test1_scaled, test2_scaled = normalize_data(train_df, test1_df, test2_df)
    
    # Window 적용
    print("Applying Window to Test Data...")
    test1_tensor = apply_window(test1_scaled, config['window_size'])
    test2_tensor = apply_window(test2_scaled, config['window_size'])
    
    # 윈도우로 인한 라벨 길이 조정
    label1 = label1[config['window_size']-1:].values
    label2 = label2[config['window_size']-1:].values
    
    model_files = [f for f in os.listdir(config['model_checkpoint_dir']) if f.endswith('.h5')]
    
    results = []
    
    for model_file in model_files:
        model_name = model_file.replace('_best.h5', '')
        print(f"\n🔍 Evaluating {model_name}...")
        
        # 모델 로드 (Custom Layer 포함)
        model = load_model(
            os.path.join(config['model_checkpoint_dir'], model_file),
            custom_objects={
                'AttentionLayer': AttentionLayer,
                'FeatureAttentionLayer': FeatureAttentionLayer
            }
        )
        
        # 평가 루프 (Test1, Test2)
        for test_name, tensor, labels in [('Test1', test1_tensor, label1), ('Test2', test2_tensor, label2)]:
            print(f"  - Dataset: {test_name}")
            
            # 1. Reconstruction Error
            mse = calculate_reconstruction_error(model, tensor)
            
            # 2. Moving Average
            mse_smooth = apply_moving_average(mse, config['moving_average_window'])
            
            # 3. Find Best Threshold
            threshold_range = np.arange(0, 0.01, config['threshold_step'])
            metrics = find_best_threshold(mse_smooth, labels, threshold_range)
            
            metrics['model'] = model_name
            metrics['dataset'] = test_name
            results.append(metrics)
            
            print(f"    Best F1: {metrics['f1']:.4f} (Thresh: {metrics['threshold']:.5f})")

    # 결과 저장
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(config['output_dir'], 'evaluation_results.csv'), index=False)
    print("\n✅ Evaluation Complete. Results saved.")

if __name__ == "__main__":
    import os
    main()