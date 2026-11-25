import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from xgboost import XGBRegressor
import logging

class XGBoostModelWrapper:
    def __init__(self, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 30,
            "random_state": 42,
            "n_jobs": -1
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        if self.logger:
            self.logger.info(f"Bắt đầu huấn luyện XGBoost ({len(X_train):,} mẫu Train, {len(X_val):,} mẫu Val, {X_train.shape[1]} đặc trưng)...")

        self.model = XGBRegressor(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        best_iter = getattr(self.model, "best_iteration", self.params["n_estimators"])
        if self.logger:
            self.logger.info(f"Hoàn thành huấn luyện XGBoost. Best iteration: {best_iter}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Mô hình XGBoost chưa được fit!")
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list) -> pd.Series:
        if self.model is None:
            return pd.Series()
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str):
        return joblib.load(file_path)
