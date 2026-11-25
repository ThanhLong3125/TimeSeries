import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch

from lab4.src.data_loader import load_and_validate_data, DataScaler
from lab4.src.feature_engineer import split_chronological, create_xgb_features, create_lstm_sequences
from lab4.src.models_var import VARModelWrapper
from lab4.src.models_xgb import XGBoostModelWrapper
from lab4.src.models_lstm import LSTMModelWrapper
from lab4.src.evaluator import ModelEvaluator
from lab4.src.tracker import ExperimentTracker

class TestLab4Pipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        dates = pd.date_range("2020-01-01 00:00:00", periods=200, freq="1h")
        self.df_dummy = pd.DataFrame({
            "Datetime": dates,
            "load": 30000.0 + np.sin(np.linspace(0, 20, 200)) * 5000 + np.random.randn(200) * 100,
            "temperature": 20.0 + np.cos(np.linspace(0, 10, 200)) * 10,
            "humidity": 60.0 + np.sin(np.linspace(0, 15, 200)) * 20,
            "wind_speed": 2.0 + np.random.rand(200) * 3
        })
        self.raw_csv = os.path.join(self.test_dir, "test_raw.csv")
        self.df_dummy.to_csv(self.raw_csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_data_loader_and_scaler(self):
        df = load_and_validate_data(self.raw_csv)
        self.assertEqual(len(df), 200)
        self.assertEqual(df.shape[1], 4)

        scaler = DataScaler(feature_cols=["load", "temperature", "humidity", "wind_speed"])
        df_scaled = scaler.fit_transform(df)
        self.assertAlmostEqual(df_scaled["load"].mean(), 0.0, places=1)
        self.assertAlmostEqual(df_scaled["load"].std(), 1.0, places=1)

        # Kiểm tra inverse transform
        inv = scaler.inverse_transform_target(df_scaled["load"].values)
        np.testing.assert_allclose(inv, df["load"].values, rtol=1e-4)

    def test_feature_engineering_and_split(self):
        df = load_and_validate_data(self.raw_csv)
        train, val, test = split_chronological(df, 0.7, 0.15, 0.15)
        self.assertEqual(len(train), 140)
        self.assertEqual(len(val), 30)
        self.assertEqual(len(test), 30)
        self.assertTrue(train.index.max() < val.index.min() < test.index.min())

        # Test XGB features
        X_xgb, y_xgb = create_xgb_features(df, lag_window=24)
        self.assertEqual(X_xgb.shape[1], 24 * 4 + 3) # 96 lags + 3 rolling
        self.assertEqual(len(X_xgb), len(y_xgb))

        # Test LSTM tensor
        X_lstm, y_lstm, ts = create_lstm_sequences(df, lag_window=24)
        self.assertEqual(X_lstm.shape[1], 24)
        self.assertEqual(X_lstm.shape[2], 4)
        self.assertEqual(len(X_lstm), len(y_lstm))

    def test_models_quick(self):
        df = load_and_validate_data(self.raw_csv)
        scaler = DataScaler(feature_cols=["load", "temperature", "humidity", "wind_speed"])
        df_scaled = scaler.fit_transform(df)

        # 1. VAR
        var_wrapper = VARModelWrapper(maxlags=4)
        var_wrapper.fit(df_scaled.iloc[:100])
        var_file = os.path.join(self.test_dir, "var.joblib")
        var_wrapper.save(var_file)
        self.assertTrue(os.path.exists(var_file))

        # 2. XGBoost
        X_xgb, y_xgb = create_xgb_features(df_scaled, lag_window=12)
        xgb_wrapper = XGBoostModelWrapper(params={"n_estimators": 10, "max_depth": 3, "n_jobs": 1})
        xgb_wrapper.fit(X_xgb.iloc[:80], y_xgb.iloc[:80], X_xgb.iloc[80:100], y_xgb.iloc[80:100])
        preds_xgb = xgb_wrapper.predict(X_xgb.iloc[100:])
        self.assertEqual(len(preds_xgb), len(X_xgb) - 100)

        # 3. LSTM
        X_lstm, y_lstm, _ = create_lstm_sequences(df_scaled, lag_window=12)
        lstm_wrapper = LSTMModelWrapper(input_size=4, hidden_dim1=16, hidden_dim2=8, num_epochs=2, batch_size=16)
        lstm_wrapper.fit(X_lstm[:80], y_lstm[:80], X_lstm[80:100], y_lstm[80:100])
        preds_lstm = lstm_wrapper.predict(X_lstm[100:])
        self.assertEqual(len(preds_lstm), len(X_lstm) - 100)

    def test_evaluator_metrics(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 300.0])
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["MAE"], 20.0 / 3.0, places=3)
        self.assertGreater(metrics["RMSE"], 0)
        self.assertGreater(metrics["MAPE"], 0)

if __name__ == "__main__":
    unittest.main()
