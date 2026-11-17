#!/usr/bin/env python3
import argparse
import time
import yaml
import os
import json
import joblib
from src.logger import logger
from src.data_loader import load_and_validate_data
from src.preprocessor import preprocess_time_series, split_train_test
from src.stationarity import run_adf_with_autolag_logging, analyze_differencing_orders
from src.trainer import train_time_series_model
from src.evaluator import (
    evaluate_test_performance,
    run_residual_diagnostics,
    plot_residual_diagnostics
)
from src.forecaster import generate_forecasts, plot_forecast_vs_actual
from src.tracker import record_audit_entry
from src.eda import plot_eda_raw_series, plot_eda_differencing, plot_eda_acf_pacf
from src.model_comparator import compare_champion_challenger

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
    
    # 3. EDA Visualizations (Ghi đè cố định lên ảnh cũ)
    eda_raw_plot = plot_eda_raw_series(series, cfg, run_id=None)
    
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
    
    # Xuất bản biểu đồ EDA sai phân & ACF/PACF (Ghi đè cố định lên ảnh cũ)
    eda_diff_plot = plot_eda_differencing(diff_analysis, cfg, run_id=None)
    eda_acf_plot = plot_eda_acf_pacf(diff_analysis[1]["series"], cfg, run_id=None)
    
    if dry_run:
        logger.info("[DRY-RUN MODE] Data ingestion, validation, and EDA completed successfully. Halting before training.")
        return
        
    # 5. Model Training (Challenger)
    t0 = time.time()
    model = train_time_series_model(
        train_series=train_series,
        model_type=cfg["model"]["type"],
        order=tuple(cfg["model"]["order"]),
        seasonal_order=tuple(cfg["model"]["seasonal_order"]) if cfg["model"].get("seasonal_order") else None,
        dataset_name=cfg.get("dataset_name", exp_name),
        experiment_name=exp_name
    )
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
    
    # 7. Model Evaluation Gate: Champion vs Challenger Comparison
    models_dir = "lab2/artifacts/models"
    metrics_dir = "lab2/artifacts/metrics"
    figures_dir = "lab2/artifacts/figures"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    champ_model_path = os.path.join(models_dir, f"{exp_name}_champion.joblib")
    champ_metrics_path = os.path.join(metrics_dir, f"{exp_name}_champion.json")
    
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
    
    gate_result = compare_champion_challenger(
        new_metrics=full_metrics,
        champion_metrics_path=champ_metrics_path
    )
    
    is_promoted = gate_result["is_promoted"]
    
    if is_promoted:
        # THĂNG HẠNG: Ghi đè file model, metrics và toàn bộ biểu đồ Champion
        joblib.dump(model, champ_model_path)
        full_metrics["champion_gate"] = gate_result
        with open(champ_metrics_path, "w", encoding="utf-8") as f:
            json.dump(full_metrics, f, indent=2, ensure_ascii=False)
            
        residuals_plot_path = plot_residual_diagnostics(model.resid, exp_name, run_id=None)
        forecast_plot_path = plot_forecast_vs_actual(train_series, actual_test, forecast_test, test_metrics, cfg, run_id=None)
        logger.info(f"🏆 [CHAMPION PROMOTED] {gate_result['reason']}")
    else:
        # TỪ CHỐI: Giữ nguyên Champion cũ, không ghi đè model và biểu đồ
        logger.warning(f"🛑 [CHALLENGER REJECTED] {gate_result['reason']}")
        logger.warning(f"   Current Champion artifacts preserved. Model and Figures NOT overwritten.")
        residuals_plot_path = os.path.join(figures_dir, f"{exp_name}_residuals.png")
        forecast_plot_path = os.path.join(figures_dir, f"{exp_name}_forecast.png")
        
    time_eval = time.time() - t0
    total_time = time.time() - t_start
    
    # 8. Record Immutable Audit Trail (Ghi nhận toàn bộ lần chạy kèm kết quả phân xử Gate)
    audit_run_id = record_audit_entry(
        config=cfg,
        metrics=test_metrics,
        model_path=champ_model_path if is_promoted else f"rejected_{temp_run_id}",
        execution_time_sec=total_time,
        gate_decision=gate_result["verdict"],
        gate_reason=gate_result["reason"]
    )
    
    logger.info(f"================================================================================")
    logger.info(f"PIPELINE EXECUTION COMPLETED! Result: {gate_result['verdict']}")
    logger.info(f"Audit Run ID   : {audit_run_id}")
    logger.info(f"Gate Verdict   : {'PROMOTED' if is_promoted else 'REJECTED'}")
    logger.info(f"Gate Reason    : {gate_result['reason']}")
    logger.info(f"Active Model   : {champ_model_path}")
    logger.info(f"Active Metrics : {champ_metrics_path}")
    logger.info(f"EDA Plots      : 1) {eda_raw_plot}")
    logger.info(f"                 2) {eda_diff_plot}")
    logger.info(f"                 3) {eda_acf_plot}")
    logger.info(f"Eval Plots     : 1) {residuals_plot_path}")
    logger.info(f"                 2) {forecast_plot_path}")
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
