import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from lab3.src.data_loader import load_and_clean_data
from lab3.src.feature_engineer import create_features, split_time_series
from lab3.src.trainer import ModelTrainer
from lab3.src.evaluator import ModelEvaluator
from lab3.src.tracker import ExperimentTracker
from lab3.src.visualizer import generate_all_plots

class TestLab3Pipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Tạo dữ liệu giả lập có trùng lặp và khuyết timestamp
        dates = pd.date_range("2020-01-01 00:00:00", periods=200, freq="h")
        self.df_raw = pd.DataFrame({
            "Datetime": dates,
            "AEP_MW": 1000.0 + np.sin(np.linspace(0, 20, 200)) * 200 + np.random.randn(200) * 10
        })
        # Gây lỗi trùng lặp tại mốc 10
        self.df_raw.loc[11, "Datetime"] = self.df_raw.loc[10, "Datetime"]
        # Gây lỗi khuyết 1 mốc tại mốc 50
        self.df_raw = self.df_raw.drop(index=50).reset_index(drop=True)

        self.raw_csv = os.path.join(self.test_dir, "test_raw.csv")
        self.df_raw.to_csv(self.raw_csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_data_loader(self):
        clean_csv = os.path.join(self.test_dir, "test_clean.csv")
        df_clean = load_and_clean_data(
            raw_path=self.raw_csv,
            datetime_col="Datetime",
            target_col="AEP_MW",
            freq="h",
            clean_output_path=clean_csv
        )
        self.assertTrue(os.path.exists(clean_csv))
        self.assertEqual(df_clean.index.duplicated().sum(), 0)
        self.assertFalse(df_clean["AEP_MW"].isna().any())
        self.assertEqual(df_clean.index.freqstr, "h")

    def test_feature_engineering_and_split(self):
        clean_csv = os.path.join(self.test_dir, "test_clean.csv")
        df_clean = load_and_clean_data(self.raw_csv, clean_output_path=clean_csv)
        
        df_feat = create_features(
            df=df_clean,
            target_col="AEP_MW",
            lags=[1, 2, 24],
            rolling_windows=[{"window": 24, "metrics": ["mean", "std"]}],
            calendar=["hour", "dayofweek", "is_weekend"]
        )
        self.assertIn("lag_1", df_feat.columns)
        self.assertIn("lag_24", df_feat.columns)
        self.assertIn("rolling_mean_24h", df_feat.columns)
        self.assertIn("hour", df_feat.columns)
        self.assertIn("is_weekend", df_feat.columns)
        self.assertFalse(df_feat.isna().any().any())

        # Test Train/Test Split
        split_date = df_feat.index[100].strftime("%Y-%m-%d %H:%M:%S")
        X_train, y_train, X_test, y_test = split_time_series(df_feat, split_date=split_date)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertTrue(X_train.index.max() < X_test.index.min())

    def test_trainer_and_evaluator(self):
        clean_csv = os.path.join(self.test_dir, "test_clean.csv")
        df_clean = load_and_clean_data(self.raw_csv, clean_output_path=clean_csv)
        df_feat = create_features(df_clean, lags=[1, 2], rolling_windows=[{"window": 6, "metrics": ["mean"]}])
        
        split_point = df_feat.index[120].strftime("%Y-%m-%d %H:%M:%S")
        X_train, y_train, X_test, y_test = split_time_series(df_feat, split_date=split_point)

        trainer = ModelTrainer(models_dir=self.test_dir)
        model = trainer.train_and_save(
            model_name="linear_regression",
            config={"fit_intercept": True},
            X_train=X_train,
            y_train=y_train
        )
        saved_file = os.path.join(self.test_dir, "linear_regression.joblib")
        self.assertTrue(os.path.exists(saved_file))

        evaluator = ModelEvaluator()
        df_bench = evaluator.benchmark_models({"LinearRegression": model}, X_test, y_test)
        self.assertEqual(len(df_bench), 1)
        self.assertGreater(df_bench.loc[0, "MAE"], 0)
        self.assertGreater(df_bench.loc[0, "RMSE"], 0)

    def test_tracker_hash(self):
        tracker = ExperimentTracker(audit_file=os.path.join(self.test_dir, "audit.jsonl"))
        h1 = tracker.calculate_file_hash(self.raw_csv)
        self.assertIsInstance(h1, str)
        self.assertEqual(len(h1), 32) # MD5 string length

    def test_visualizer(self):
        clean_csv = os.path.join(self.test_dir, "test_clean.csv")
        df_clean = load_and_clean_data(self.raw_csv, clean_output_path=clean_csv)
        df_bench = pd.DataFrame([{"Model": "Linear", "MAE": 10.0, "RMSE": 12.0, "MAPE": 1.5}])
        feat_imp = pd.Series([0.6, 0.4], index=["lag_1", "lag_2"])
        y_test = df_clean["AEP_MW"].tail(20)
        preds = {"Linear": y_test.values}
        df_rec = pd.DataFrame({"Actual": [100.0, 105.0], "Recursive_Forecast": [99.0, 104.0]}, index=pd.date_range("2020-01-02", periods=2, freq="h"))

        out_dir = os.path.join(self.test_dir, "test_figures")
        generate_all_plots(df_clean, df_bench, feat_imp, y_test, preds, df_rec, output_dir=out_dir)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "lab3_01_eda_raw_series.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "lab3_02_eda_daily_pattern.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "lab3_03_model_benchmark.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "lab3_04_feature_importance.png")))

if __name__ == "__main__":
    unittest.main()
