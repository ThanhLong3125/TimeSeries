import os
import time
import joblib
import pandas as pd
from typing import Dict, Any, Optional
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class ModelTrainer:
    def __init__(self, models_dir: str = "lab3/artifacts/models", logger: Optional[logging.Logger] = None):
        self.models_dir = models_dir
        self.logger = logger
        os.makedirs(models_dir, exist_ok=True)
        self.trained_models = {}

    def get_model_instance(self, model_name: str, config: Dict[str, Any]):
        if model_name == "linear_regression":
            return LinearRegression(**config)
        elif model_name == "random_forest":
            return RandomForestRegressor(**config)
        elif model_name == "xgboost":
            return XGBRegressor(**config)
        else:
            raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")

    def train_and_save(
        self,
        model_name: str,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        if self.logger:
            self.logger.info(f"Bắt đầu huấn luyện mô hình: [{model_name}] với {len(X_train):,} mẫu...")

        model = self.get_model_instance(model_name, config)
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time

        if self.logger:
            self.logger.info(f"Hoàn thành huấn luyện [{model_name}] sau {duration:.2f}s")

        model_file = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_file)

        if self.logger:
            self.logger.info(f"Đã lưu checkpoint mô hình tại: {model_file}")

        self.trained_models[model_name] = model
        return model
