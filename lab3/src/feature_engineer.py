import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

def create_features(
    df: pd.DataFrame,
    target_col: str = "AEP_MW",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[Dict]] = None,
    calendar: Optional[List[str]] = None,
    features_output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Tạo các nhóm đặc trưng cho chuỗi thời gian: Lags, Rolling statistics và Calendar features.
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168]
    if rolling_windows is None:
        rolling_windows = [{"window": 24, "metrics": ["mean", "std"]}, {"window": 168, "metrics": ["mean", "std"]}]
    if calendar is None:
        calendar = ["hour", "dayofweek", "quarter", "month", "year", "dayofyear", "is_weekend"]

    if logger:
        logger.info(f"Bắt đầu trích xuất đặc trưng (Lags: {lags}, Rolling: {rolling_windows}, Calendar: {calendar})")

    df_feat = df.copy()

    # 1. Tạo Lag features
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    # 2. Tạo Rolling statistics (trên shift(1) để chống rò rỉ target hiện tại)
    shifted_target = df_feat[target_col].shift(1)
    for rw in rolling_windows:
        win = rw["window"]
        for metric in rw.get("metrics", ["mean"]):
            col_name = f"rolling_{metric}_{win}h"
            if metric == "mean":
                df_feat[col_name] = shifted_target.rolling(window=win).mean()
            elif metric == "std":
                df_feat[col_name] = shifted_target.rolling(window=win).std()

    # 3. Tạo Calendar features
    for cal in calendar:
        if cal == "hour":
            df_feat["hour"] = df_feat.index.hour
        elif cal == "dayofweek":
            df_feat["dayofweek"] = df_feat.index.dayofweek
        elif cal == "quarter":
            df_feat["quarter"] = df_feat.index.quarter
        elif cal == "month":
            df_feat["month"] = df_feat.index.month
        elif cal == "year":
            df_feat["year"] = df_feat.index.year
        elif cal == "dayofyear":
            df_feat["dayofyear"] = df_feat.index.dayofyear
        elif cal == "is_weekend":
            df_feat["is_weekend"] = (df_feat.index.dayofweek >= 5).astype(int)

    # Loại bỏ giá trị NaN do lag / rolling window ở đầu chuỗi
    initial_len = len(df_feat)
    df_feat = df_feat.dropna()
    dropped_rows = initial_len - len(df_feat)

    if logger:
        logger.info(f"Đã loại bỏ {dropped_rows} dòng NaN ban đầu do trượt cửa sổ. Kích thước ma trận đặc trưng: {df_feat.shape}")

    if features_output_path:
        os.makedirs(os.path.dirname(features_output_path), exist_ok=True)
        df_feat.to_csv(features_output_path)
        if logger:
            logger.info(f"Đã lưu ma trận đặc trưng tại: {features_output_path}")

    return df_feat


def split_time_series(
    df: pd.DataFrame,
    split_date: str = "2017-01-01",
    target_col: str = "AEP_MW",
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Phân chia Train/Test theo trục thời gian không xáo trộn.
    """
    feature_cols = [c for c in df.columns if c != target_col]

    train_mask = df.index < split_date
    test_mask = df.index >= split_date

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    if logger:
        logger.info(f"Phân chia Train/Test tại mốc: {split_date}")
        logger.info(f" - Train set: {len(X_train):,} mẫu ({X_train.index.min()} -> {X_train.index.max()})")
        logger.info(f" - Test set : {len(X_test):,} mẫu ({X_test.index.min()} -> {X_test.index.max()})")

    return X_train, y_train, X_test, y_test
