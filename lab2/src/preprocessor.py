import pandas as pd
import numpy as np
from .logger import logger

def preprocess_time_series(df, date_col, target_col, frequency="MS", apply_log=False):
    """
    Tiền xử lý chuỗi thời gian: Căn chỉnh DatetimeIndex và áp dụng Log transformation nếu cần.
    """
    series = df.set_index(date_col)[target_col].copy()
    series = series.asfreq(frequency)
    
    if series.isnull().any():
        null_count = series.isnull().sum()
        logger.warning(f"Interpolating {null_count} missing values using forward fill + linear interpolation")
        series = series.ffill().interpolate(method="time")
        
    if apply_log:
        logger.info("Applying natural logarithm transform: y_log = ln(y) to stabilize variance (Heteroskedasticity fix)")
        if (series <= 0).any():
            logger.error("Cannot apply log transform to non-positive values!")
            raise ValueError("Series contains values <= 0, log transform invalid")
        series = np.log(series)
        
    return series

def split_train_test(series, train_ratio=0.8):
    """
    Phân chia Train/Test theo thứ tự thời gian nghiêm ngặt (tuyệt đối không shuffle).
    """
    split_idx = int(len(series) * train_ratio)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    
    logger.info(
        f"Train/Test split: total={len(series)}, train={len(train)} ({train.index.min().strftime('%Y-%m')} -> {train.index.max().strftime('%Y-%m')}), "
        f"test={len(test)} ({test.index.min().strftime('%Y-%m')} -> {test.index.max().strftime('%Y-%m')})"
    )
    return train, test
