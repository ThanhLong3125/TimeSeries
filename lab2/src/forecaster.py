import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .logger import logger

def generate_forecasts(fitted_model, test_series, apply_log=False):
    """
    Sinh dự báo ngoài mẫu trên tập Test và hoàn nguyên (invert) log nếu cần.
    """
    n_steps = len(test_series)
    logger.info(f"Generating out-of-sample forecast for {n_steps} future steps...")
    
    forecast_res = fitted_model.forecast(steps=n_steps)
    forecast_series = pd.Series(forecast_res.values, index=test_series.index)
    
    if apply_log:
        logger.info("Inverting log transformation: actual = exp(y_log), forecast = exp(fc_log)")
        actual_original = np.exp(test_series)
        forecast_original = np.exp(forecast_series)
        return actual_original, forecast_original
    else:
        return test_series, forecast_series

def plot_forecast_vs_actual(train_series, actual_test, forecast_series, metrics, config, run_id=None, output_dir="lab2/artifacts/figures"):
    """
    Vẽ biểu đồ đối chiếu ngoài mẫu (Forecast vs Actuals) theo chuẩn Senior Data Analyst.
    Ghi đè trực tiếp lên ảnh cũ nếu không truyền run_id.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_name = config.get("experiment_name", "model")
    model_type = config["model"]["type"]
    order = config["model"]["order"]
    s_order = config["model"].get("seasonal_order")
    model_label = f"{model_type}{order}" if not s_order else f"{model_type}{order}x{s_order}"
    unit = config.get("evaluation", {}).get("unit", "Value")
    
    # Chuẩn hóa train series nếu có log
    train_plot = np.exp(train_series) if config.get("apply_log") else train_series
    
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(train_plot.index, train_plot, color="#1F3A60", lw=1.5, label=f"Train ({train_plot.index.min().strftime('%Y-%m')} – {train_plot.index.max().strftime('%Y-%m')})")
    ax.plot(actual_test.index, actual_test, color="#1A252C", lw=2.0, label="Actual Test")
    ax.plot(forecast_series.index, forecast_series, color="#E74C3C", lw=2.2, linestyle="--", label=f"Forecast ({model_label})")
    
    # Vùng Test phủ nền xám nhạt
    ax.axvspan(actual_test.index[0], actual_test.index[-1], color="#F2F4F4", alpha=0.8, zorder=0)
    ax.axvline(actual_test.index[0], color="#BDC3C7", linestyle="--", lw=1.2)
    
    # Điểm sai lệch lớn nhất (Peak Bias)
    max_err_ts = pd.to_datetime(metrics["max_error"]["timestamp"])
    actual_max = metrics["max_error"]["actual"]
    pred_max = metrics["max_error"]["predicted"]
    delta_max = metrics["max_error"]["value"]
    
    ax.scatter([max_err_ts], [actual_max], color="#E74C3C", s=50, zorder=5)
    ax.annotate(
        f"Peak Error ({max_err_ts.strftime('%Y-%m')})\\nActual: {actual_max:.1f} vs FC: {pred_max:.1f} (Δ {delta_max:.1f})",
        xy=(max_err_ts, actual_max),
        xytext=(max_err_ts - pd.DateOffset(months=max(6, len(actual_test)//3)), actual_max + (actual_max * 0.05)),
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2),
        fontsize=9, color="#922B21", fontweight="bold"
    )
    
    # Thanh KPI Banner tinh tế góc trên bên trái
    mae = metrics["MAE"]
    rmse = metrics["RMSE"]
    mape = metrics["MAPE"]
    kpi_banner = f"{model_label} Evaluation  —  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}  |  MAPE: {mape:.1f}%"
    ax.text(0.02, 0.94, kpi_banner, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color="#1F3A60", bbox=dict(boxstyle="square,pad=0.5", facecolor="#EAEDED", edgecolor="#BDC3C7", lw=1.0))
            
    ax.set_title(config.get("evaluation", {}).get("plot_title", f"{exp_name} Forecast"), fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(unit, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.76), frameon=True, framealpha=0.9, fontsize=9.5)
    
    plt.tight_layout()
    filename = f"{exp_name}_forecast_{run_id}.png" if run_id else f"{exp_name}_forecast.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Forecast evaluation plot saved (overwritten): path={filepath}")
    return filepath
