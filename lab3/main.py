import os
import sys
import yaml
import argparse
import joblib
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.logger import setup_logger
from src.tracker import ExperimentTracker
from src.data_loader import load_and_clean_data
from src.feature_engineer import create_features, split_time_series
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import generate_all_plots

def parse_args():
    parser = argparse.ArgumentParser(description="Lab 3 ML Time Series Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="lab3/configs/xgboost_aep.yaml",
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
    
    # 1. Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Không tìm thấy file cấu hình tại {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Setup logger and audit tracker
    logger = setup_logger(
        name="AEP_XGBoost_Pipeline",
        log_dir=cfg["outputs"]["logs_dir"]
    )
    tracker = ExperimentTracker(audit_file=cfg["outputs"]["audit_file"])
    
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU CHẠY PIPELINE LAB 3: MACHINE LEARNING DỰ BÁO PHỤ TẢI ĐIỆN")
    logger.info("=" * 70)
    
    # Audit log config
    tracker.log_event("pipeline_start", {
        "config_path": args.config,
        "project": cfg["project"]["name"],
        "version": cfg["project"]["version"]
    })

    # 3. Data Ingestion & Cleaning
    raw_hash = tracker.calculate_file_hash(cfg["data"]["raw_path"])
    logger.info(f"Checksum dữ liệu gốc (MD5): {raw_hash}")
    
    df_clean = load_and_clean_data(
        raw_path=cfg["data"]["raw_path"],
        datetime_col=cfg["data"]["datetime_column"],
        target_col=cfg["data"]["target_column"],
        freq=cfg["data"]["freq"],
        clean_output_path=cfg["data"]["clean_path"],
        logger=logger
    )
    
    # 4. Feature Engineering
    df_feat = create_features(
        df=df_clean,
        target_col=cfg["data"]["target_column"],
        lags=cfg["features"]["lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        calendar=cfg["features"]["calendar"],
        features_output_path=cfg["data"]["features_path"],
        logger=logger
    )
    
    # 5. Train/Test Split
    X_train, y_train, X_test, y_test = split_time_series(
        df=df_feat,
        split_date=cfg["split"]["split_date"],
        target_col=cfg["data"]["target_column"],
        logger=logger
    )

    feature_names = list(X_train.columns)

    # 6. Model Training & Serialization
    trainer = ModelTrainer(models_dir=cfg["outputs"]["models_dir"], logger=logger)
    trained_models = {}

    if not args.skip_train:
        for m_name, m_params in cfg["models"].items():
            model = trainer.train_and_save(
                model_name=m_name,
                config=m_params,
                X_train=X_train,
                y_train=y_train
            )
            trained_models[m_name] = model
            # Audit log model artifact
            model_path = os.path.join(cfg["outputs"]["models_dir"], f"{m_name}.joblib")
            tracker.log_event("model_trained", {
                "model_name": m_name,
                "artifact_path": model_path,
                "md5": tracker.calculate_file_hash(model_path)
            })
    else:
        logger.info("Chế độ --skip-train kích hoạt. Đang nạp các mô hình đã huấn luyện từ artifacts...")
        for m_name in cfg["models"].keys():
            model_path = os.path.join(cfg["outputs"]["models_dir"], f"{m_name}.joblib")
            if os.path.exists(model_path):
                trained_models[m_name] = joblib.load(model_path)
                logger.info(f"Đã nạp checkpoint: {model_path}")
            else:
                logger.error(f"Không tìm thấy checkpoint: {model_path}")
                sys.exit(1)

    # 7. Model Evaluation & Benchmarking
    evaluator = ModelEvaluator(logger=logger)
    df_bench = evaluator.benchmark_models(
        models=trained_models,
        X_test=X_test,
        y_test=y_test
    )
    
    logger.info("\n" + "=" * 55)
    logger.info("BẢNG SO CHUẨN CÁC MÔ HÌNH TRÊN TẬP TEST (2017 - 2018):")
    logger.info("=" * 55)
    for _, row in df_bench.iterrows():
        logger.info(f" -> {row['Model']:<20} | MAE: {row['MAE']:>7.2f} MW | RMSE: {row['RMSE']:>7.2f} MW | MAPE: {row['MAPE']:>5.2f}%")
    logger.info("=" * 55)

    tracker.log_event("evaluation_benchmark", {
        "metrics": df_bench.to_dict(orient="records")
    })

    # Save metrics JSON into artifacts/metrics/
    metrics_dir = cfg["outputs"].get("metrics_dir", "lab3/artifacts/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "benchmark_metrics.json")
    df_bench.to_json(metrics_file, orient="records", indent=2)
    logger.info(f"Đã lưu bảng số liệu đánh giá tại: {metrics_file}")

    # 8. Feature Importance (XGBoost)
    feat_imp = None
    if "xgboost" in trained_models:
        feat_imp = evaluator.get_feature_importance(trained_models["xgboost"], feature_names)
        logger.info("TOP 5 ĐẶC TRƯNG QUAN TRỌNG NHẤT CỦA XGBOOST:")
        for idx, (f, score) in enumerate(feat_imp.head(5).items(), 1):
            logger.info(f"   {idx}. {f:<20}: {score*100:.2f}%")

    # 9. Recursive Multi-step Forecasting
    logger.info("Tiến hành dự báo đa bước đệ quy (Recursive Multi-Step Forecasting)...")
    df_rec = evaluator.recursive_forecast(
        model=trained_models["xgboost"],
        df_clean=df_clean,
        feature_names=feature_names,
        start_time=cfg["forecasting"]["start_forecast_time"],
        horizon=cfg["forecasting"]["recursive_horizon"],
        target_col=cfg["data"]["target_column"]
    )
    rec_metrics = evaluator.calculate_metrics(df_rec["Actual"], df_rec["Recursive_Forecast"].values)
    tracker.log_event("recursive_forecast_completed", {
        "horizon": cfg["forecasting"]["recursive_horizon"],
        "start_time": cfg["forecasting"]["start_forecast_time"],
        "metrics": rec_metrics
    })

    # 10. Generate Visualizations (Artifact Figures)
    figures_dir = cfg["outputs"].get("figures_dir", "lab3/artifacts/figures")
    logger.info(f"Đang tạo và lưu toàn bộ biểu đồ vào: {figures_dir}...")
    test_preds = {name: m.predict(X_test) for name, m in trained_models.items()}
    generate_all_plots(
        df_clean=df_clean,
        df_bench=df_bench,
        feat_imp=feat_imp,
        y_test=y_test,
        predictions=test_preds,
        df_rec=df_rec,
        output_dir=figures_dir,
        target_col=cfg["data"]["target_column"]
    )
    logger.info("Đã tạo thành công 6 biểu đồ trực quan hóa!")

    logger.info("=" * 70)
    logger.info("PIPELINE LAB 3 HOÀN TẤT THÀNH CÔNG VÀ AN TOÀN!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
