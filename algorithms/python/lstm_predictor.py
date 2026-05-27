# -*- coding: utf-8 -*-
"""
LSTM/GRU 时间序列预测 — Python 实现
基于 Keras/TensorFlow

用法：
    pip install tensorflow
    python lstm_predictor.py
"""

import numpy as np


def create_sequences(data, seq_length):
    """将时间序列转为监督学习格式"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def _rnn_forecast(series, cell_type='lstm', seq_length=12, forecast_num=6,
                  units=64, epochs=100, batch_size=16, verbose=True):
    """RNN 预测的共享实现"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    cell_cls = layers.LSTM if cell_type == 'lstm' else layers.GRU

    data = np.asarray(series, dtype=float).reshape(-1, 1)
    mean, std = np.mean(data), np.std(data)
    data_norm = (data - mean) / (std + 1e-10)

    X, y = create_sequences(data_norm, seq_length)
    X = X.reshape(-1, seq_length, 1)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = keras.Sequential([
        cell_cls(units, activation='tanh', return_sequences=True,
                 input_shape=(seq_length, 1)),
        layers.Dropout(0.2),
        cell_cls(units // 2, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    early_stop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop], verbose=0)

    if verbose:
        print(f"  [{cell_type.upper()}] 训练 {len(history.history['loss'])} epochs, "
              f"val_loss: {min(history.history['val_loss']):.6f}")

    fitted_norm = model.predict(X, verbose=0).flatten()
    fitted = fitted_norm * std + mean
    fitted_full = np.concatenate([np.full(seq_length, np.nan), fitted])

    last_seq = data_norm[-seq_length:].reshape(1, seq_length, 1)
    forecast_norm = []
    for _ in range(forecast_num):
        pred = model.predict(last_seq, verbose=0)[0, 0]
        forecast_norm.append(pred)
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0, -1, 0] = pred

    forecast = np.array(forecast_norm) * std + mean

    return {
        'fitted': fitted_full,
        'forecast': forecast,
        'history': history,
        'model': model,
    }


def lstm_forecast(series, seq_length=12, forecast_num=6, units=64,
                  epochs=100, batch_size=16, verbose=True):
    """LSTM 时间序列预测"""
    return _rnn_forecast(series, 'lstm', seq_length, forecast_num,
                         units, epochs, batch_size, verbose)


def gru_forecast(series, seq_length=12, forecast_num=6, units=64, epochs=100):
    """GRU 时间序列预测"""
    return _rnn_forecast(series, 'gru', seq_length, forecast_num,
                         units, epochs, batch_size=16, verbose=True)


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("LSTM/GRU 时间序列预测示例")
    print("=" * 60)

    t = np.arange(200)
    trend = 0.1 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 0.8, 200)
    series = 30 + trend + seasonal + noise

    print(f"序列长度: {len(series)}")

    try:
        result = lstm_forecast(series, seq_length=24, forecast_num=12,
                              epochs=50, units=32, verbose=True)
        print(f"LSTM 未来12期预测: {result['forecast'].round(2)[:5]}...")
    except ImportError:
        print("TensorFlow 未安装 (pip install tensorflow)")
    except Exception as e:
        print(f"错误: {e}")
