import unittest
import os
import pandas as pd
import numpy as np
from src.data_loader import load_and_validate_data
from src.preprocessor import preprocess_time_series, split_train_test
from src.stationarity import run_adf_with_autolag_logging, analyze_differencing_orders
from src.tracker import calculate_file_md5, record_audit_entry

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

if __name__ == "__main__":
    unittest.main()
