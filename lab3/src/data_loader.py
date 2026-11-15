import os
import pandas as pd
from typing import Optional
import logging

def load_and_clean_data(
    raw_path: str,
    datetime_col: str = "Datetime",
    target_col: str = "AEP_MW",
    freq: str = "h",
    clean_output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Tải dữ liệu chuỗi thời gian, xử lý datetime index, khử trùng lặp và nội suy mốc khuyết.
    """
    if logger:
        logger.info(f"Bắt đầu đọc dữ liệu từ: {raw_path}")
        
    df = pd.read_csv(raw_path)
    total_raw = len(df)
    if logger:
        logger.info(f"Kích thước ban đầu: {total_raw:,} dòng, {df.shape[1]} cột")
        
    # Chuyển đổi và sắp xếp Datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    
    # Xử lý trùng lặp mốc giờ (DST Autumn)
    dup_count = df[datetime_col].duplicated().sum()
    if dup_count > 0:
        if logger:
            logger.warning(f"Phát hiện {dup_count} mốc thời gian bị trùng lặp (do đổi giờ DST). Đang khử trùng...")
        df = df.drop_duplicates(subset=[datetime_col], keep="first")
        
    df = df.set_index(datetime_col)
    
    # Đồng bộ tần số chuỗi thời gian
    df = df.asfreq(freq)
    missing_count = df[target_col].isna().sum()
    if missing_count > 0:
        if logger:
            logger.info(f"Phát hiện {missing_count} mốc giờ khuyết ({(missing_count/len(df))*100:.3f}%). Đang nội suy chuỗi...")
        df[target_col] = df[target_col].interpolate(method="time")
        
    if clean_output_path:
        os.makedirs(os.path.dirname(clean_output_path), exist_ok=True)
        df.to_csv(clean_output_path)
        if logger:
            logger.info(f"Đã lưu dữ liệu làm sạch tại: {clean_output_path}")
            
    if logger:
        logger.info(f"Dữ liệu sau làm sạch: {len(df):,} quan sát liên tục từ {df.index.min()} đến {df.index.max()}")
        
    return df
