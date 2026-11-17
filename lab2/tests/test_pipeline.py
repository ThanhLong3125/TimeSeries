import unittest
import os
import pandas as pd
import numpy as np
from src.data_loader import load_and_validate_data
from src.preprocessor import preprocess_time_series, split_train_test
from src.stationarity import run_adf_with_autolag_logging, analyze_differencing_orders
from src.tracker import calculate_file_md5, record_audit_entry
from src.model_comparator import compare_champion_challenger

class TestTimeSeriesPipeline(unittest.TestCase):

    def test_data_loader_a10(self):
        df = load_and_validate_data(
            data_path="lab2/data/raw/a10.csv",
            date_col="date",
            target_col="value"
        )
        self.assertEqual(len(df), 204)
        self.assertTrue(df["date"].is_monotonic_increasing)
        self.assertFalse(df["value"].isnull().any())

    def test_preprocessor_split(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            "value": np.arange(10)
        })
        series = preprocess_time_series(df, date_col="date", target_col="value")
        train, test = split_train_test(series, train_ratio=0.8)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        self.assertTrue(train.index.max() < test.index.min())

    def test_stationarity_adf_autolag(self):
        np.random.seed(42)
        series = pd.Series(np.random.randn(100), index=pd.date_range("2000-01-01", periods=100, freq="D"))
        res = run_adf_with_autolag_logging(series, series_name="Synthetic Noise")
        self.assertIn("adf_stat", res)
        self.assertIn("autolag_history", res)
        self.assertGreater(len(res["autolag_history"]), 0)
        self.assertTrue(res["is_stationary"])

    def test_differencing_analysis(self):
        t = np.arange(100)
        series = pd.Series(2.5 * t + np.random.randn(100))
        diff_res = analyze_differencing_orders(series, max_order=2)
        self.assertIn(0, diff_res)
        self.assertIn(1, diff_res)
        self.assertIn(2, diff_res)
        self.assertLess(diff_res[1]["variance"], diff_res[0]["variance"])

    def test_tracker_md5(self):
        md5_hash = calculate_file_md5("lab2/data/raw/a10.csv")
        self.assertIsNotNone(md5_hash)
        self.assertEqual(len(md5_hash), 32)

    def test_model_comparator_gate(self):
        # 1. Test Initial Champion
        metrics_new = {
            "test_performance": {"RMSE": 4.5, "MAE": 3.5, "MAPE": 15.0},
            "aic": 500.0, "bic": 510.0
        }
        res_init = compare_champion_challenger(metrics_new, "lab2/artifacts/metrics/non_existent.json")
        self.assertTrue(res_init["is_promoted"])
        self.assertEqual(res_init["verdict"], "INITIAL_CHAMPION")

        # 2. Test Superior Challenger
        import tempfile
        import json
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            champ_data = {
                "test_performance": {"RMSE": 5.0, "MAE": 4.0, "MAPE": 18.0},
                "aic": 520.0, "bic": 530.0
            }
            json.dump(champ_data, tmp)
            tmp_path = tmp.name

        try:
            # Challenger tốt hơn (RMSE 4.5 < 5.0)
            res_better = compare_champion_challenger(metrics_new, tmp_path)
            self.assertTrue(res_better["is_promoted"])
            self.assertEqual(res_better["verdict"], "PROMOTED_SUPERIOR_FORECAST")

            # Challenger kém hơn (RMSE 6.0 > 5.0)
            metrics_worse = {
                "test_performance": {"RMSE": 6.0, "MAE": 5.0, "MAPE": 22.0},
                "aic": 550.0, "bic": 560.0
            }
            res_worse = compare_champion_challenger(metrics_worse, tmp_path)
            self.assertFalse(res_worse["is_promoted"])
            self.assertEqual(res_worse["verdict"], "REJECTED_INFERIOR_ACCURACY")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
