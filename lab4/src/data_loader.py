import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import joblib
import logging

def load_and_validate_data(
    raw_path: str,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "load",
    clean_output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Tải dữ liệu chuỗi thời gian đa biến, kiểm tra missing, outlier và sắp xếp datetime.
    """
    if feature_cols is None:
        feature_cols = ["load", "temperature", "humidity", "wind_speed"]

    if logger:
        logger.info(f"Bắt đầu đọc dữ liệu từ: {raw_path}")

    df = pd.read_csv(raw_path)
    
    # Xác định cột Datetime
    dt_col = None
    for col in df.columns:
        if col.lower() in ["datetime", "date", "time", "timestamp"]:
            dt_col = col
            break

    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(dt_col).set_index(dt_col)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            # Tạo Datetime index nếu là chuỗi chỉ số
            df.index = pd.date_range("2017-01-01 00:00:00", periods=len(df), freq="1h")

    # Chỉ giữ các cột quan sát
    df = df[feature_cols].copy()

    # Kiểm tra missing values
    missing_sum = df.isna().sum().sum()
    if missing_sum > 0:
        if logger:
            logger.warning(f"Phát hiện {missing_sum} giá trị khuyết. Tiến hành nội suy tuyến tính...")
        df = df.interpolate(method="time").ffill().bfill()
    else:
        if logger:
            logger.info("Dữ liệu hoàn toàn sạch, không có mốc khuyết (0 missing values).")

    if clean_output_path:
        os.makedirs(os.path.dirname(clean_output_path), exist_ok=True)
        df.to_csv(clean_output_path)
        if logger:
            logger.info(f"Đã lưu dữ liệu tiền xử lý tại: {clean_output_path}")

    if logger:
        logger.info(f"Kích thước dữ liệu đa biến: {len(df):,} dòng, {len(feature_cols)} cột")

    return df


class DataScaler:
    def __init__(self, feature_cols: List[str], target_col: str = "load"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.target_idx = feature_cols.index(target_col)
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame):
        self.scaler.fit(df_train[self.feature_cols])
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Scaler chưa được fit trên dữ liệu Train!")
        scaled_vals = self.scaler.transform(df[self.feature_cols])
        return pd.DataFrame(scaled_vals, index=df.index, columns=self.feature_cols)

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(df_train)
        return self.transform(df_train)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Khôi phục thang đo gốc cho riêng biến target (load).
        y_orig = y_scaled * std + mean
        """
        mean = self.scaler.mean_[self.target_idx]
        scale = self.scaler.scale_[self.target_idx]
        return y_scaled * scale + mean

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str):
        return joblib.load(file_path)
