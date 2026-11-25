import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import logging

class ModelEvaluator:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        return {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        }

    def benchmark_models(
        self,
        y_true_orig: np.ndarray,
        predictions_orig: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Tổng hợp bảng so sánh hiệu năng RMSE, MAE, MAPE trên thang đo gốc (Original Scale: kW).
        """
        results = []
        for name, y_pred in predictions_orig.items():
            metrics = self.calculate_metrics(y_true_orig, y_pred)
            metrics["Model"] = name
            results.append(metrics)
            if self.logger:
                self.logger.info(f"Kết quả kiểm thử [{name}]: RMSE={metrics['RMSE']:.2f} kW, MAE={metrics['MAE']:.2f} kW, MAPE={metrics['MAPE']:.2f}%")

        df_bench = pd.DataFrame(results)[["Model", "RMSE", "MAE", "MAPE"]]
        return df_bench

    @staticmethod
    def walk_forward_multistep(
        model_type: str,
        model_obj: Any,
        initial_history: np.ndarray,
        future_weather: np.ndarray,
        horizon: int = 24
    ) -> np.ndarray:
        """
        Mô phỏng Walk-forward Validation đa bước (predict t+1 -> update -> predict t+2 ...).
        initial_history: shape (24, 4) chứa [load, temp, humidity, wind] của 24h gần nhất.
        future_weather: shape (horizon, 3) chứa [temp, humidity, wind] của các giờ tương lai (dự báo thời tiết đã biết).
        """
        history = initial_history.copy()
        preds = []

        for step in range(horizon):
            w_next = future_weather[step] # [temp, hum, wind]
            if model_type == "var":
                vec_next = model_obj.predict_one_step(history)
                load_pred = vec_next[0]
            elif model_type == "xgb":
                # Làm phẳng history để tạo đặc trưng
                # Đơn giản hóa dự báo đa bước
                load_pred = float(model_obj.predict(pd.DataFrame([history.flatten()])))
            elif model_type == "lstm":
                x_tensor = history[np.newaxis, :, :].astype(np.float32)
                load_pred = float(model_obj.predict(x_tensor)[0])
            else:
                load_pred = history[-1, 0]

            preds.append(load_pred)
            # Cập nhật bước mới vào chuỗi lịch sử (FIFO trượt 24 bước)
            new_row = np.array([load_pred, w_next[0], w_next[1], w_next[2]])
            history = np.vstack([history[1:], new_row])

        return np.array(preds)
