#!/usr/bin/env python3
import argparse
import time
import yaml
import os
from src.logger import logger
from src.data_loader import load_and_validate_data
from src.preprocessor import preprocess_time_series, split_train_test
from src.stationarity import run_adf_with_autolag_logging, analyze_differencing_orders
from src.trainer import train_time_series_model, save_model_artifact
from src.evaluator import (
    evaluate_test_performance,
    run_residual_diagnostics,
    export_metrics_json,
    plot_residual_diagnostics
)
from src.forecaster import generate_forecasts, plot_forecast_vs_actual
from src.tracker import record_audit_entry
from src.eda import plot_eda_raw_series, plot_eda_differencing, plot_eda_acf_pacf

def run_pipeline(config_path, dry_run=False):
    t_start = time.time()
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Missing config: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    exp_name = cfg.get("experiment_name", "unnamed_experiment")
    temp_run_id = f"run_{int(time.time())}"
    
    logger.info(f"================================================================================")
    logger.info(f"STARTING TIME SERIES PIPELINE: {exp_name}")
    logger.info(f"Config: {config_path} | DryRun={dry_run}")
    logger.info(f"================================================================================")
    
    # 1. Ingestion & Validation
    t0 = time.time()
    df = load_and_validate_data(
        data_path=cfg["data_path"],
        date_col=cfg["date_col"],
        target_col=cfg["target_col"],
        date_format=cfg.get("date_format"),
        frequency=cfg.get("frequency", "MS")
    )
    time_load = time.time() - t0
    
    # 2. Preprocessing & Train/Test Split
    t0 = time.time()
    series = preprocess_time_series(
        df=df,
        date_col=cfg["date_col"],
        target_col=cfg["target_col"],
        frequency=cfg.get("frequency", "MS"),
        apply_log=cfg.get("apply_log", False)
    )
    train_series, test_series = split_train_test(
        series=series,
        train_ratio=cfg.get("train_ratio", 0.8)
    )
    time_prep = time.time() - t0
    
    # 3. EDA Visualizations (Tự động xuất bản biểu đồ EDA)
    eda_raw_plot = plot_eda_raw_series(series, cfg, temp_run_id)
    
    # 4. Stationarity & Differencing Analysis
    t0 = time.time()
    adf_raw = run_adf_with_autolag_logging(
        series=train_series,
        series_name=f"{exp_name} (Train Raw)",
        autolag=cfg.get("stationarity", {}).get("autolag", "AIC"),
        alpha=cfg.get("stationarity", {}).get("alpha", 0.05)
    )
    diff_analysis = analyze_differencing_orders(
        series=train_series,
        max_order=cfg.get("stationarity", {}).get("max_diff_order", 2)
    )
    time_stat = time.time() - t0
    
    # Xuất bản biểu đồ EDA sai phân & ACF/PACF
    eda_diff_plot = plot_eda_differencing(diff_analysis, cfg, temp_run_id)
    eda_acf_plot = plot_eda_acf_pacf(diff_analysis[1]["series"], cfg, temp_run_id)
    
    if dry_run:
        logger.info("[DRY-RUN MODE] Data ingestion, validation, and EDA completed successfully. Halting before training.")
        return
        
    # 5. Model Training & Serialization
    t0 = time.time()
    model = train_time_series_model(
        train_series=train_series,
        model_type=cfg["model"]["type"],
        order=tuple(cfg["model"]["order"]),
        seasonal_order=tuple(cfg["model"]["seasonal_order"]) if cfg["model"].get("seasonal_order") else None
    )
    model_path = save_model_artifact(model, exp_name, temp_run_id)
    time_train = time.time() - t0
    
    # 6. Out-of-sample Forecast & Evaluation
    t0 = time.time()
    actual_test, forecast_test = generate_forecasts(
        fitted_model=model,
        test_series=test_series,
        apply_log=cfg.get("apply_log", False)
    )
    test_metrics = evaluate_test_performance(actual=actual_test, predicted=forecast_test)
    residual_diag = run_residual_diagnostics(
        residuals=model.resid,
        diagnostic_lags=cfg.get("evaluation", {}).get("diagnostic_lags", [10, 12, 24])
    )
    
    # 7. Export Artifacts (Metrics JSON & Figures)
    full_metrics = {
        "experiment": exp_name,
        "model": cfg["model"],
        "aic": float(model.aic),
        "bic": float(model.bic),
        "test_performance": test_metrics,
        "residual_diagnostics": residual_diag,
        "stationarity_tests": {
            "raw_adf": {k: v for k, v in adf_raw.items() if k != "autolag_history"}
        }
    }
    
    metrics_path = export_metrics_json(full_metrics, exp_name, temp_run_id)
    residuals_plot_path = plot_residual_diagnostics(model.resid, exp_name, temp_run_id)
    forecast_plot_path = plot_forecast_vs_actual(train_series, actual_test, forecast_test, test_metrics, cfg, temp_run_id)
    time_eval = time.time() - t0
    
    total_time = time.time() - t_start
    
    # 8. Record Immutable Audit Trail
    audit_run_id = record_audit_entry(
        config=cfg,
        metrics=test_metrics,
        model_path=model_path,
        execution_time_sec=total_time
    )
    
    # Đồng bộ đổi tên toàn bộ artifacts (cả EDA, Model, Metrics, Plots) với audit_run_id chính thức
    final_artifacts = {}
    for label, old_path in [
        ("Model", model_path),
        ("Metrics", metrics_path),
        ("EDA_Raw", eda_raw_plot),
        ("EDA_Diff", eda_diff_plot),
        ("EDA_ACF", eda_acf_plot),
        ("Residual_Plot", residuals_plot_path),
        ("Forecast_Plot", forecast_plot_path)
    ]:
        new_path = old_path.replace(temp_run_id, audit_run_id)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
        final_artifacts[label] = new_path
    
    logger.info(f"================================================================================")
    logger.info(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"Audit Run ID   : {audit_run_id}")
    logger.info(f"Model Artifact : {final_artifacts['Model']}")
    logger.info(f"Metrics JSON   : {final_artifacts['Metrics']}")
    logger.info(f"EDA Plots      : 1) {final_artifacts['EDA_Raw']}")
    logger.info(f"                 2) {final_artifacts['EDA_Diff']}")
    logger.info(f"                 3) {final_artifacts['EDA_ACF']}")
    logger.info(f"Eval Plots     : 1) {final_artifacts['Residual_Plot']}")
    logger.info(f"                 2) {final_artifacts['Forecast_Plot']}")
    logger.info(
        f"Execution Time : Total={total_time:.3f}s (Load={time_load:.3f}s, Prep={time_prep:.3f}s, "
        f"Stat={time_stat:.3f}s, Train={time_train:.3f}s, Eval={time_eval:.3f}s)"
    )
    logger.info(f"================================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Production Pipeline - Lab 2")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Run validation and stationarity tests only")
    args = parser.parse_args()
    
    run_pipeline(config_path=args.config, dry_run=args.dry_run)
