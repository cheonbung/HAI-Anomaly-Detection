import json
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from utils.preprocessing import get_preprocessed_data
from models.architectures import get_model

def plot_history(history, save_path):
    """
    학습 과정의 Loss 변화를 그래프로 그려 저장하는 함수
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"  - Loss plot saved to {save_path}")

def main(args):
    # 1. Config 로드
    config_path = 'configs/config.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # CLI 인자로 Config 덮어쓰기 (실험 시 유용)
    model_name = args.model if args.model else config['target_models'][0]
    epochs = args.epochs if args.epochs else config['epochs']
    batch_size = args.batch_size if args.batch_size else config['batch_size']
    learning_rate = args.lr if args.lr else config['learning_rate']
    
    print(f"\n🚀 Start Training Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Window Size: {config['window_size']}")
    print("-" * 40)

    # 2. 데이터 전처리 파이프라인 실행
    # get_preprocessed_data는 (train, val, test1, test2) 4개의 텐서를 반환합니다.
    # 학습 단계에서는 train과 val만 필요하므로 나머지는 _로 받습니다.
    print("\n🔄 Running Data Preprocessing Pipeline...")
    train_tensor, val_tensor, _, _ = get_preprocessed_data(config)
    
    print(f"  - Train Tensor Shape: {train_tensor.shape}")
    print(f"  - Val Tensor Shape: {val_tensor.shape}")

    # 3. 모델 빌드
    # 입력 형태: (Window Size, Feature Dimension)
    input_shape = (train_tensor.shape[1], train_tensor.shape[2])
    model = get_model(model_name, input_shape)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    print(f"\n🏗️ Model Architecture: {model_name}")
    model.summary()

    # 4. 체크포인트 및 콜백 설정
    # 디렉토리 생성
    if not os.path.exists(config['model_checkpoint_dir']):
        os.makedirs(config['model_checkpoint_dir'])
    
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
        
    checkpoint_path = os.path.join(config['model_checkpoint_dir'], f"{model_name}_best.h5")
    
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
    ]

    # 5. 학습 실행
    print("\n🔥 Starting Training...")
    history = model.fit(
        train_tensor, train_tensor,  # AutoEncoder는 입력 == 타겟
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_tensor, val_tensor),
        callbacks=callbacks,
        shuffle=True
    )
    
    # 6. 결과 저장
    # 학습 History(Loss 값 등)를 pickle 파일로 저장
    history_path = os.path.join(config['output_dir'], f"{model_name}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
        
    # Loss 그래프 저장
    plot_path = os.path.join(config['output_dir'], f"{model_name}_loss.png")
    plot_history(history, plot_path)
    
    print(f"\n✅ Training Finished Successfully.")
    print(f"  - Best Model saved at: {checkpoint_path}")
    print(f"  - History saved at: {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Anomaly Detection Models on HAI Dataset")
    
    parser.add_argument("--model", type=str, default=None, 
                        help="Name of the model to train (e.g., Conv_BiLSTM_AE, Conv_BiLSTM_AE_Attention)")
    parser.add_argument("--epochs", type=int, default=None, 
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=None, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=None, 
                        help="Learning rate")
    
    args = parser.parse_args()
    
    # GPU 설정 확인 (메모리 증가 할당)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} device(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ GPU NOT Available. Training will use CPU.")

    main(args)