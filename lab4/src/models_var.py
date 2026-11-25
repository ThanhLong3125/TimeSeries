import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import logging

class VARModelWrapper:
    def __init__(self, maxlags: int = 24, ic: str = "aic", logger: Optional[logging.Logger] = None):
        self.maxlags = maxlags
        self.ic = ic
        self.logger = logger
        self.fitted_model = None
        self.optimal_lag = 1
        self.needs_diff = False
        self.last_train_values = None

    def check_stationarity(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Kiểm định Augmented Dickey-Fuller (ADF) cho từng biến trong hệ đa biến.
        """
        adf_results = {}
        all_stationary = True
        for col in df.columns:
            res = adfuller(df[col].dropna())
            is_stat = (res[1] < 0.05) and (res[0] < res[4]["5%"])
            adf_results[col] = {
                "adf_stat": float(res[0]),
                "p_value": float(res[1]),
                "used_lag": int(res[2]),
                "is_stationary": bool(is_stat)
            }
            if not is_stat:
                all_stationary = False
            if self.logger:
                status_str = "DỪNG" if is_stat else "KHÔNG DỪNG"
                self.logger.info(f"ADF test [{col}]: ADF={res[0]:.4f}, p-value={res[1]:.4e} -> {status_str}")

        self.needs_diff = not all_stationary
        return adf_results

    def fit(self, df_train: pd.DataFrame):
        """
        Huấn luyện mô hình VAR trên dữ liệu Train.
        """
        data = df_train.copy()
        if self.needs_diff:
            if self.logger:
                self.logger.info("Chuỗi không dừng -> Tiến hành lấy sai phân bậc 1 (d=1) cho mô hình VAR.")
            self.last_train_values = data.iloc[-1].copy()
            data = data.diff().dropna()

        var_model = VAR(data)
        lag_order_results = var_model.select_order(maxlags=min(self.maxlags, len(data) // 5))
        self.optimal_lag = int(getattr(lag_order_results, self.ic, 1))
        if self.optimal_lag == 0:
            self.optimal_lag = 1

        if self.logger:
            self.logger.info(f"Bậc trễ tối ưu được chọn theo {self.ic.upper()}: p = {self.optimal_lag}")

        self.fitted_model = var_model.fit(self.optimal_lag)
        return self

    def predict_one_step(self, history_window: np.ndarray) -> np.ndarray:
        """
        Dự báo 1 bước tiếp theo cho toàn bộ vector [load, temp, humidity, wind].
        history_window có shape (p, num_features).
        """
        if self.fitted_model is None:
            raise ValueError("Mô hình VAR chưa được fit!")
        forecast = self.fitted_model.forecast(y=history_window[-self.optimal_lag:], steps=1)
        return forecast[0]

    def predict_test_set(self, df_full: pd.DataFrame, test_indices: pd.DatetimeIndex, target_col: str = "load") -> np.ndarray:
        """
        Thực hiện dự báo 1 bước (one-step ahead) cuộn qua toàn bộ tập Test (Walk-forward).
        """
        target_idx = df_full.columns.get_loc(target_col)
        preds = []

        data_vals = df_full.values
        feature_cols = df_full.columns

        # Vị trí của mốc test đầu tiên trong df_full
        start_pos = df_full.index.get_loc(test_indices[0])

        for pos in range(start_pos, len(df_full)):
            # Lấy p bước thời gian ngay trước pos
            hist = data_vals[pos - self.optimal_lag : pos]
            if self.needs_diff:
                # Nếu sai phân, ta lấy sai phân của cửa sổ
                hist_diff = np.diff(data_vals[pos - self.optimal_lag - 1 : pos], axis=0)
                diff_pred = self.fitted_model.forecast(y=hist_diff[-self.optimal_lag:], steps=1)[0]
                # Hoàn trả sai phân: y_{t} = y_{t-1} + diff
                pred_val = data_vals[pos - 1, target_idx] + diff_pred[target_idx]
            else:
                pred_vec = self.fitted_model.forecast(y=hist, steps=1)[0]
                pred_val = pred_vec[target_idx]

            preds.append(pred_val)

        return np.array(preds)

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str):
        return joblib.load(file_path)
