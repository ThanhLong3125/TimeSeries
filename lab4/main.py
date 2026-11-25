import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import joblib

# Thêm đường dẫn project vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.logger import setup_logger
from src.tracker import ExperimentTracker
from src.data_loader import load_and_validate_data, DataScaler
from src.feature_engineer import split_chronological, create_xgb_features, create_lstm_sequences
from src.models_var import VARModelWrapper
from src.models_xgb import XGBoostModelWrapper
from src.models_lstm import LSTMModelWrapper
from src.evaluator import ModelEvaluator
from src.visualizer import (
    plot_eda_time_series,
    plot_load_vs_temperature,
    plot_correlation_matrix,
    plot_model_benchmark,
    plot_lstm_loss_curve,
    plot_feature_importance,
    plot_test_forecast_comparison
)

def parse_args():
    parser = argparse.ArgumentParser(description="Lab 4: Multivariate Time Series Pipeline (VAR, XGBoost, LSTM)")
    parser.add_argument(
        "--config",
        type=str,
        default="lab4/configs/multivariate_forecast.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training and load saved models from disk"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Nạp cấu hình
    if not os.path.exists(args.config):
        print(f"Lỗi: Không tìm thấy file cấu hình tại {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Khởi tạo Logger và Audit Tracker
    logger = setup_logger(name="Multivariate_TS_Pipeline", log_dir=cfg["outputs"]["logs_dir"])
    tracker = ExperimentTracker(audit_file=cfg["outputs"]["audit_file"])

    logger.info("=" * 75)
    logger.info("BẮT ĐẦU CHẠY PIPELINE LAB 4: DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (LOAD + WEATHER)")
    logger.info("=" * 75)

    tracker.log_event("pipeline_start", {
        "config_path": args.config,
        "project": cfg["project"]["name"],
        "version": cfg["project"]["version"]
    })

    # 3. Nạp và kiểm định dữ liệu thô
    raw_hash = tracker.calculate_file_hash(cfg["data"]["raw_path"])
    logger.info(f"Checksum dữ liệu gốc (MD5): {raw_hash}")

    df_raw = load_and_validate_data(
        raw_path=cfg["data"]["raw_path"],
        feature_cols=cfg["data"]["feature_columns"],
        target_col=cfg["data"]["target_column"],
        clean_output_path=cfg["data"]["processed_path"],
        logger=logger
    )

    # 4. Trực quan hóa Khám phá Dữ liệu (EDA Figures)
    figures_dir = cfg["outputs"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Tạo và lưu biểu đồ EDA vào: {figures_dir}")
    plot_eda_time_series(df_raw, figures_dir, target_col=cfg["data"]["target_column"])
    plot_load_vs_temperature(df_raw, figures_dir)
    plot_correlation_matrix(df_raw, figures_dir)

    # 5. Phân chia tập dữ liệu theo thứ tự thời gian (70/15/15)
    df_train, df_val, df_test = split_chronological(
        df=df_raw,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        logger=logger
    )

    # 6. Chuẩn hóa dữ liệu bằng StandardScaler
    logger.info("Chuẩn hóa toàn bộ 4 biến bằng StandardScaler (fit trên tập Train)...")
    scaler = DataScaler(feature_cols=cfg["data"]["feature_columns"], target_col=cfg["data"]["target_column"])
    scaler.fit(df_train)
    scaler_path = os.path.join(cfg["outputs"]["models_dir"], "data_scaler.joblib")
    scaler.save(scaler_path)

    df_train_scaled = scaler.transform(df_train)
    df_val_scaled = scaler.transform(df_val)
    df_test_scaled = scaler.transform(df_test)
    df_full_scaled = scaler.transform(df_raw)

    # 7. Kỹ nghệ đặc trưng cho XGBoost (Tabular Lags + Rolling)
    lag_win = cfg["feature_engineering"]["lag_window"]
    logger.info(f"Trích xuất đặc trưng cho XGBoost (Cửa sổ trễ L={lag_win}, 96 lags + rolling)...")
    X_train_xgb, y_train_xgb = create_xgb_features(df_train_scaled, lag_window=lag_win, feature_cols=cfg["data"]["feature_columns"], target_col=cfg["data"]["target_column"])
    X_val_xgb, y_val_xgb = create_xgb_features(df_val_scaled, lag_window=lag_win, feature_cols=cfg["data"]["feature_columns"], target_col=cfg["data"]["target_column"])
    X_test_xgb, y_test_xgb = create_xgb_features(df_test_scaled, lag_window=lag_win, feature_cols=cfg["data"]["feature_columns"], target_col=cfg["data"]["target_column"])

    # 8. Kỹ nghệ chuỗi cho LSTM (Tensor 3D: N, 24, 4)
    logger.info(f"Đóng gói Tensor 3D cho LSTM: (N, {lag_win}, 4)...")
    X_train_lstm, y_train_lstm, _ = create_lstm_sequences(df_train_scaled, lag_window=lag_win, feature_cols=cfg["data"]["feature_columns"], target_col=cfg["data"]["target_column"])
    X_val_lstm, y_val_lstm, _ = create_lstm_sequences(df_val_scaled, lag_window=lag_win, feature_cols=cfg["data"]["feature_columns"], target_col=cfg["data"]["target_column"])
    X_test_lstm, y_test_lstm, test_timestamps = create_lstm_sequences(df_test_scaled, lag_window=lag_win, feature_cols=cfg["data"]["feature_columns"], target_col=cfg["data"]["target_column"])

    # Căn chỉnh test indices chung cho cả 3 mô hình
    common_test_idx = test_timestamps
    y_test_scaled = y_test_lstm
    y_test_orig = scaler.inverse_transform_target(y_test_scaled)

    # 9. Huấn luyện hoặc Tải Checkpoint các Mô hình
    models_dir = cfg["outputs"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    var_path = os.path.join(models_dir, "var_model.joblib")
    xgb_path = os.path.join(models_dir, "xgboost_model.joblib")
    lstm_path = os.path.join(models_dir, "lstm_model.pt")

    if not args.skip_train:
        # === 9.1 Huấn luyện VAR ===
        logger.info("-" * 50)
        logger.info("HUẤN LUYỆN MÔ HÌNH 1: VAR (Vector Autoregression)...")
        var_wrapper = VARModelWrapper(maxlags=cfg["models"]["var"]["maxlags"], ic=cfg["models"]["var"]["ic"], logger=logger)
        var_wrapper.check_stationarity(df_train_scaled)
        var_wrapper.fit(df_train_scaled)
        var_wrapper.save(var_path)
        tracker.log_event("model_trained", {"model": "VAR", "optimal_lag": var_wrapper.optimal_lag, "artifact": var_path})

        # === 9.2 Huấn luyện XGBoost ===
        logger.info("-" * 50)
        logger.info("HUẤN LUYỆN MÔ HÌNH 2: XGBoost Regressor...")
        xgb_wrapper = XGBoostModelWrapper(params=cfg["models"]["xgboost"], logger=logger)
        xgb_wrapper.fit(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
        xgb_wrapper.save(xgb_path)
        tracker.log_event("model_trained", {"model": "XGBoost", "artifact": xgb_path})

        # === 9.3 Huấn luyện LSTM ===
        logger.info("-" * 50)
        logger.info("HUẤN LUYỆN MÔ HÌNH 3: LSTM Deep Learning...")
        lstm_cfg = cfg["models"]["lstm"]
        lstm_wrapper = LSTMModelWrapper(
            input_size=lstm_cfg["input_size"],
            hidden_dim1=lstm_cfg["hidden_dim1"],
            hidden_dim2=lstm_cfg["hidden_dim2"],
            output_dim=lstm_cfg["output_dim"],
            learning_rate=lstm_cfg["learning_rate"],
            batch_size=lstm_cfg["batch_size"],
            num_epochs=lstm_cfg["num_epochs"],
            early_stopping_patience=lstm_cfg["early_stopping_patience"],
            logger=logger
        )
        lstm_wrapper.fit(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
        lstm_wrapper.save(lstm_path)
        tracker.log_event("model_trained", {"model": "LSTM", "artifact": lstm_path})

    else:
        logger.info("Chế độ --skip-train kích hoạt. Nạp mô hình từ artifacts...")
        var_wrapper = VARModelWrapper.load(var_path)
        xgb_wrapper = XGBoostModelWrapper.load(xgb_path)
        lstm_wrapper = LSTMModelWrapper.load(lstm_path, logger=logger)
        logger.info("Đã nạp thành công 3 checkpoint mô hình!")

    # 10. Đánh giá Mô hình trên Tập Test (Walk-forward / One-step Ahead)
    logger.info("=" * 55)
    logger.info("TIẾN HÀNH ĐÁNH GIÁ VÀ SO CHUẨN TRÊN TẬP TEST:")
    logger.info("=" * 55)

    # Dự báo VAR
    pred_var_scaled = var_wrapper.predict_test_set(df_full_scaled, common_test_idx, target_col=cfg["data"]["target_column"])
    pred_var_orig = scaler.inverse_transform_target(pred_var_scaled)

    # Dự báo XGBoost
    pred_xgb_scaled = xgb_wrapper.predict(X_test_xgb.loc[common_test_idx])
    pred_xgb_orig = scaler.inverse_transform_target(pred_xgb_scaled)

    # Dự báo LSTM
    pred_lstm_scaled = lstm_wrapper.predict(X_test_lstm)
    pred_lstm_orig = scaler.inverse_transform_target(pred_lstm_scaled)

    predictions_orig = {
        "VAR (Statistical)": pred_var_orig,
        "XGBoost (ML)": pred_xgb_orig,
        "LSTM (Deep Learning)": pred_lstm_orig
    }

    evaluator = ModelEvaluator(logger=logger)
    df_bench = evaluator.benchmark_models(y_test_orig, predictions_orig)

    logger.info("\n" + "=" * 65)
    logger.info("BẢNG SO CHUẨN HIỆU NĂNG 3 MÔ HÌNH TRÊN TẬP TEST (Thang đo kW):")
    logger.info("=" * 65)
    for _, row in df_bench.iterrows():
        logger.info(f" -> {row['Model']:<22} | RMSE: {row['RMSE']:>8.2f} kW | MAE: {row['MAE']:>8.2f} kW | MAPE: {row['MAPE']:>5.2f}%")
    logger.info("=" * 65)

    # Lưu metrics JSON
    metrics_dir = cfg["outputs"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "benchmark_metrics.json")
    df_bench.to_json(metrics_path, orient="records", indent=2)
    logger.info(f"Đã lưu kết quả đánh giá tại: {metrics_path}")

    tracker.log_event("benchmark_evaluation", {"metrics": df_bench.to_dict(orient="records")})

    # 11. Trực quan hóa Kết Quả Đánh Giá (Figures)
    logger.info("Tạo các biểu đồ so sánh kết quả và phân tích...")
    plot_model_benchmark(df_bench, figures_dir)
    plot_lstm_loss_curve(lstm_wrapper.train_losses, lstm_wrapper.val_losses, figures_dir)

    feat_imp = xgb_wrapper.get_feature_importance(list(X_train_xgb.columns))
    plot_feature_importance(feat_imp, figures_dir)

    plot_test_forecast_comparison(
        timestamps=common_test_idx,
        y_true=y_test_orig,
        preds_dict=predictions_orig,
        output_dir=figures_dir,
        num_hours=168
    )

    logger.info("=" * 75)
    logger.info("PIPELINE LAB 4 HOÀN TẤT THÀNH CÔNG VÀ CHUẨN XÁC!")
    logger.info("=" * 75)

if __name__ == "__main__":
    main()
