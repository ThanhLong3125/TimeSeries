import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
import logging

def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu theo trục thời gian không xáo trộn: 70% Train, 15% Val, 15% Test.
    """
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    if logger:
        logger.info(f"Phân chia thời gian (Không shuffle):")
        logger.info(f" - Train      : {len(df_train):,} dòng ({df_train.index.min()} -> {df_train.index.max()})")
        logger.info(f" - Validation : {len(df_val):,} dòng ({df_val.index.min()} -> {df_val.index.max()})")
        logger.info(f" - Test       : {len(df_test):,} dòng ({df_test.index.min()} -> {df_test.index.max()})")

    return df_train, df_val, df_test


def create_xgb_features(
    df_scaled: pd.DataFrame,
    lag_window: int = 24,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "load"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tạo vector phẳng 96 đặc trưng trễ (24h * 4 biến) kèm Rolling features cho XGBoost.
    Mỗi dòng t dùng [t-24 : t-1] để dự báo y_t = load_t.
    """
    if feature_cols is None:
        feature_cols = ["load", "temperature", "humidity", "wind_speed"]

    df_feat = pd.DataFrame(index=df_scaled.index)

    # 1. Tạo 24 lag features cho mỗi biến (24 * 4 = 96 lags)
    for col in feature_cols:
        for lag in range(1, lag_window + 1):
            df_feat[f"{col}_lag_{lag}"] = df_scaled[col].shift(lag)

    # 2. Tạo Rolling Features trên shift(1) của target để tránh look-ahead bias
    target_shifted = df_scaled[target_col].shift(1)
    df_feat["load_rolling_mean_6h"] = target_shifted.rolling(window=6).mean()
    df_feat["load_rolling_std_24h"] = target_shifted.rolling(window=24).std()
    
    # 3. Load trend: load_{t-1} - load_{t-24} (hoặc load_t - load_{t-24})
    df_feat["load_trend"] = target_shifted - df_scaled[target_col].shift(lag_window)

    # Cột nhãn mục tiêu y_t (load tại bước hiện tại t)
    y = df_scaled[target_col].copy()

    # Loại bỏ các dòng NaN do lag window
    valid_idx = df_feat.dropna().index
    X = df_feat.loc[valid_idx]
    y = y.loc[valid_idx]

    return X, y


def create_lstm_sequences(
    df_scaled: pd.DataFrame,
    lag_window: int = 24,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "load"
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Tạo Tensor 3D cho LSTM: (N, 24 timesteps, 4 features) và nhãn y: (N,).
    Với mỗi index i (từ lag_window đến len-1):
      X[i] = df[i - lag_window : i] (24 bước thời gian)
      y[i] = df[i, target_col] (giá trị load ở bước tiếp theo)
    """
    if feature_cols is None:
        feature_cols = ["load", "temperature", "humidity", "wind_speed"]

    target_idx = feature_cols.index(target_col)
    data_matrix = df_scaled[feature_cols].values

    num_samples = len(data_matrix) - lag_window
    X = np.zeros((num_samples, lag_window, len(feature_cols)), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.float32)

    for i in range(num_samples):
        X[i] = data_matrix[i : i + lag_window]
        y[i] = data_matrix[i + lag_window, target_idx]

    timestamps = df_scaled.index[lag_window:]
    return X, y, timestamps
