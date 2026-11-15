import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class ModelEvaluator:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }

    def benchmark_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred)
            metrics["Model"] = name
            results.append(metrics)
            if self.logger:
                self.logger.info(f"Kết quả kiểm thử [{name}]: MAE={metrics['MAE']:.2f} MW, RMSE={metrics['RMSE']:.2f} MW, MAPE={metrics['MAPE']:.2f}%")

        df_bench = pd.DataFrame(results)[["Model", "MAE", "RMSE", "MAPE"]]
        return df_bench

    @staticmethod
    def get_feature_importance(model: Any, feature_names: List[str]) -> pd.Series:
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            return pd.Series(np.abs(model.coef_), index=feature_names).sort_values(ascending=False)
        else:
            return pd.Series()

    def recursive_forecast(
        self,
        model: Any,
        df_clean: pd.DataFrame,
        feature_names: List[str],
        start_time: str,
        horizon: int = 24,
        target_col: str = "AEP_MW"
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start_time)
        end_ts = start_ts + pd.Timedelta(hours=horizon - 1)

        history_window = df_clean.loc[df_clean.index < start_ts, target_col].tail(168).copy()
        actual_future = df_clean.loc[start_ts:end_ts, target_col].copy()

        current_series = history_window.copy()
        preds = []

        for step in range(horizon):
            cur_time = start_ts + pd.Timedelta(hours=step)
            feat_dict = {
                "lag_1": current_series.iloc[-1],
                "lag_2": current_series.iloc[-2],
                "lag_3": current_series.iloc[-3],
                "lag_24": current_series.iloc[-24],
                "lag_48": current_series.iloc[-48],
                "lag_168": current_series.iloc[-168],
                "rolling_mean_24h": current_series.tail(24).mean(),
                "rolling_std_24h": current_series.tail(24).std(),
                "rolling_mean_168h": current_series.tail(168).mean(),
                "rolling_std_168h": current_series.tail(168).std(),
                "hour": cur_time.hour,
                "dayofweek": cur_time.dayofweek,
                "quarter": cur_time.quarter,
                "month": cur_time.month,
                "year": cur_time.year,
                "dayofyear": cur_time.dayofyear,
                "is_weekend": int(cur_time.dayofweek >= 5)
            }
            X_step = pd.DataFrame([feat_dict])[feature_names]
            step_pred = float(model.predict(X_step)[0])
            preds.append(step_pred)
            current_series[cur_time] = step_pred

        df_res = pd.DataFrame({
            "Actual": actual_future.values,
            "Recursive_Forecast": preds
        }, index=actual_future.index)

        metrics = self.calculate_metrics(df_res["Actual"], df_res["Recursive_Forecast"].values)
        if self.logger:
            self.logger.info(f"Kết quả Recursive Multi-Step ({horizon} bước): MAE={metrics['MAE']:.2f} MW, MAPE={metrics['MAPE']:.2f}%")

        return df_res
