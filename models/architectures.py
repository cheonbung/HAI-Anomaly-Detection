import tensorflow as tf
from tensorflow.keras import layers, models
from .layers import AttentionLayer, FeatureAttentionLayer

def build_conv_bilstm_ae(input_shape):
    n_steps, n_features = input_shape
    
    inputs = layers.Input(shape=(n_steps, n_features))
    
    # Encoder
    x = layers.Conv1D(filters=128, kernel_size=48, padding='same', dilation_rate=1, activation="linear")(inputs)
    x = layers.Dense(128)(x)
    x = layers.Bidirectional(layers.LSTM(64, activation="relu", return_sequences=False))(x)
    
    # Decoder
    x = layers.RepeatVector(n_steps)(x)
    x = layers.Bidirectional(layers.LSTM(64, activation="relu", return_sequences=True))(x)
    x = layers.Dense(128)(x)
    x = layers.Conv1D(filters=128, kernel_size=48, padding='same', activation="linear")(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features, activation='linear'))(x)
    
    return models.Model(inputs, outputs, name="Conv_BiLSTM_AE")

def build_conv_bilstm_ae_attention(input_shape):
    n_steps, n_features = input_shape
    inputs = layers.Input(shape=(n_steps, n_features))
    
    # Encoder
    x = layers.Conv1D(128, 48, padding='same', activation="linear")(inputs)
    x = layers.Dense(128)(x)
    x = layers.Bidirectional(layers.LSTM(64, activation="relu", return_sequences=True))(x)
    
    # Attention
    x_pooled, attention_weights = AttentionLayer(name='attention')(x)
    
    # Decoder
    x = layers.RepeatVector(n_steps)(x_pooled)
    x = layers.Bidirectional(layers.LSTM(64, activation="relu", return_sequences=True))(x)
    x = layers.Dense(128)(x)
    x = layers.Conv1D(128, 48, padding='same', activation="linear")(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features, activation='linear'))(x)
    
    model = models.Model(inputs, outputs, name="Conv_BiLSTM_AE_Attention")
    return model

def get_model(model_name, input_shape):
    if model_name == "Conv_BiLSTM_AE":
        return build_conv_bilstm_ae(input_shape)
    elif model_name == "Conv_BiLSTM_AE_Attention":
        return build_conv_bilstm_ae_attention(input_shape)
    # 다른 모델들도 여기에 추가
    else:
        raise ValueError(f"Unknown model name: {model_name}")