import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_raw_data(data_dir):
    """
    CSV 파일을 로드하고 Timestamp를 파싱하여 Train/Val/Test로 분리합니다.
    (원본 노트의 날짜 기반 분리 로직 반영)
    """
    print(f"Loading data from {data_dir}...")
    try:
        train1 = pd.read_csv(os.path.join(data_dir, 'hai-train1.csv'))
        train2 = pd.read_csv(os.path.join(data_dir, 'hai-train2.csv'))
        train3 = pd.read_csv(os.path.join(data_dir, 'hai-train3.csv'))
        train4 = pd.read_csv(os.path.join(data_dir, 'hai-train4.csv'))
        
        test1 = pd.read_csv(os.path.join(data_dir, 'hai-test1.csv'))
        test2 = pd.read_csv(os.path.join(data_dir, 'hai-test2.csv'))
    except FileNotFoundError:
        print("Error: 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None, None, None, None

    # Train 데이터 합치기
    train_all = pd.concat([train1, train2, train3, train4], axis=0).reset_index(drop=True)
    
    # Timestamp 변환
    train_all['timestamp'] = pd.to_datetime(train_all['timestamp'])
    test1['timestamp'] = pd.to_datetime(test1['timestamp'])
    test2['timestamp'] = pd.to_datetime(test2['timestamp'])

    # Validation Set 분리 (24일, 25일 데이터)
    val_df = train_all[(train_all['timestamp'].dt.day == 24) | (train_all['timestamp'].dt.day == 25)].copy()
    train_df = train_all[train_all['timestamp'].dt.day != 24].copy() # 24일 제외하고 나머지

    # Timestamp 및 불필요 컬럼 제거 및 컬럼명 변경 (C0, C1...)
    def clean_df(df):
        df = df.drop(['timestamp', 'Timestamp', 'attack', 'attack_P1', 'attack_P2', 'attack_P3'], axis=1, errors='ignore')
        df.columns = [f'C{i}' for i in range(len(df.columns))]
        return df

    return clean_df(train_df), clean_df(val_df), clean_df(test1), clean_df(test2)

def remove_zero_std_columns(train_df, val_df, test1, test2):
    """
    표준편차가 0인(값이 변하지 않는) 컬럼을 제거합니다.
    """
    description = train_df.describe()
    zero_std_cols = description.loc['std'][description.loc['std'] == 0].index
    
    if len(zero_std_cols) > 0:
        print(f"Removing columns with zero std: {list(zero_std_cols)}")
        train_df = train_df.drop(columns=zero_std_cols)
        val_df = val_df.drop(columns=zero_std_cols)
        test1 = test1.drop(columns=zero_std_cols)
        test2 = test2.drop(columns=zero_std_cols)
        
    return train_df, val_df, test1, test2

def remove_high_vif_columns(train_df, val_df, test1, test2, threshold=10):
    """
    VIF(분산 팽창 요인)가 높은 변수를 제거하여 다중공선성을 줄입니다.
    (주의: 데이터가 크면 시간이 오래 걸릴 수 있으므로 샘플링하여 계산)
    """
    print("Calculating VIF to remove multicollinearity...")
    
    # 속도를 위해 데이터 샘플링 (예: 10000개만 사용)
    sample_df = train_df.sample(n=min(10000, len(train_df)), random_state=42)
    
    # MinMax Scaling for VIF calculation
    scaler = MinMaxScaler()
    sample_scaled = scaler.fit_transform(sample_df)
    
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(sample_scaled, i) for i in tqdm(range(sample_scaled.shape[1]), desc="VIF")]
    vif["features"] = sample_df.columns
    
    high_vif_cols = vif[vif["VIF Factor"] >= threshold]["features"].values
    
    print(f"Removing {len(high_vif_cols)} columns with VIF >= {threshold}")
    
    train_df = train_df.drop(columns=high_vif_cols)
    val_df = val_df.drop(columns=high_vif_cols)
    test1 = test1.drop(columns=high_vif_cols)
    test2 = test2.drop(columns=high_vif_cols)
    
    return train_df, val_df, test1, test2

def normalize_data(train_df, val_df, test1_df, test2_df):
    """
    Train 데이터를 기준으로 Min-Max Normalization 수행
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test1_scaled = scaler.transform(test1_df)
    test2_scaled = scaler.transform(test2_df)
    
    return train_scaled, val_scaled, test1_scaled, test2_scaled

def apply_window(data, window_size):
    """
    Sliding Window를 적용하여 (Samples, Window, Features) 형태의 3D 텐서로 변환
    """
    data_numpy = tf.reshape(data, (data.shape[0], -1))
    
    # TensorFlow의 dataset 유틸리티를 사용하면 더 효율적입니다.
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data_numpy,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=256 
    )
    
    # 전체 데이터를 메모리에 텐서로 올리기 (데이터가 너무 크면 이 부분 조정 필요)
    reshaped_data = []
    for batch in tqdm(ds, desc="Applying Window"):
        reshaped_data.append(batch)
        
    return tf.concat(reshaped_data, axis=0)

def get_preprocessed_data(config):
    """
    전체 전처리 파이프라인을 실행하는 함수
    """
    # 1. Load & Split
    train_df, val_df, test1_df, test2_df = load_raw_data(config['data_dir'])
    
    # 2. Remove Zero Std
    train_df, val_df, test1_df, test2_df = remove_zero_std_columns(train_df, val_df, test1_df, test2_df)
    
    # 3. Remove High VIF (Optional: config에서 제어 가능하도록 설정 추천)
    # train_df, val_df, test1_df, test2_df = remove_high_vif_columns(train_df, val_df, test1_df, test2_df)
    
    # 4. Normalize
    train_scaled, val_scaled, test1_scaled, test2_scaled = normalize_data(train_df, val_df, test1_df, test2_df)
    
    # 5. Windowing
    print("Converting to Tensor (Windowing)...")
    train_tensor = apply_window(train_scaled, config['window_size'])
    val_tensor = apply_window(val_scaled, config['window_size'])
    test1_tensor = apply_window(test1_scaled, config['window_size'])
    test2_tensor = apply_window(test2_scaled, config['window_size'])
    
    return train_tensor, val_tensor, test1_tensor, test2_tensor