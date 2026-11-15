import pandas as pd
import os
from .logger import logger
from .tracker import calculate_file_md5

def load_and_validate_data(data_path, date_col, target_col, date_format=None, frequency="MS"):
    """
    Nạp dữ liệu từ file CSV và thực hiện kiểm tra toàn vẹn chuỗi thời gian (Integrity Checks).
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file does not exist: {data_path}")
        raise FileNotFoundError(f"File not found: {data_path}")
        
    data_md5 = calculate_file_md5(data_path)
    logger.info(f"Ingesting raw dataset: path={data_path}, md5={data_md5}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded: rows={len(df)}, columns={df.columns.tolist()}")
    
    if date_col not in df.columns:
        logger.error(f"Date column '{date_col}' not found in dataframe!")
        raise KeyError(f"Missing date column: {date_col}")
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in dataframe!")
        raise KeyError(f"Missing target column: {target_col}")
        
    # Chuyển đổi Datetime
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    else:
        df[date_col] = pd.to_datetime(df[date_col])
        
    # Sắp xếp theo thứ tự thời gian tăng dần
    df = df.sort_values(by=date_col).reset_index(drop=True)
    is_monotonic = df[date_col].is_monotonic_increasing
    
    # Kiểm tra tính liên tục của thời gian (Missing Date Gaps)
    expected_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=frequency)
    missing_dates = expected_range.difference(df[date_col])
    gap_count = len(missing_dates)
    
    # Kiểm tra giá trị thiếu và giá trị âm
    null_count = df[target_col].isnull().sum()
    min_val = float(df[target_col].min())
    max_val = float(df[target_col].max())
    has_negatives = (min_val < 0)
    
    logger.info(
        f"Data Validation: time_span={df[date_col].min().strftime('%Y-%m')} to {df[date_col].max().strftime('%Y-%m')}, "
        f"monotonic={is_monotonic}, freq={frequency}, missing_gaps={gap_count}, nulls={null_count}, "
        f"min={min_val:.2f}, max={max_val:.2f}, has_negatives={has_negatives}"
    )
    
    if gap_count > 0:
        logger.warning(f"Detected {gap_count} missing time periods in series: {list(missing_dates[:3])}...")
        
    return df
