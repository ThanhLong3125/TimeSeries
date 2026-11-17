import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .logger import logger

def evaluate_test_performance(actual, predicted):
    """
    Tính toán các chỉ số đánh giá độ chính xác ngoài mẫu: MAE, RMSE, MAPE.
    """
    actual_clean = np.array(actual)
    pred_clean = np.array(predicted)
    
    mae = float(mean_absolute_error(actual_clean, pred_clean))
    rmse = float(np.sqrt(mean_squared_error(actual_clean, pred_clean)))
    
    # MAPE với kiểm tra chống chia cho 0
    non_zero_mask = (actual_clean != 0)
    if non_zero_mask.any():
        mape = float(np.mean(np.abs((actual_clean[non_zero_mask] - pred_clean[non_zero_mask]) / actual_clean[non_zero_mask])) * 100.0)
    else:
        mape = np.nan
        
    abs_errors = np.abs(actual - predicted)
    max_err_idx = abs_errors.idxmax()
    max_err_val = float(abs_errors.max())
    
    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "max_error": {
            "value": round(max_err_val, 4),
            "timestamp": max_err_idx.strftime("%Y-%m-%d"),
            "actual": float(actual[max_err_idx]),
            "predicted": float(predicted[max_err_idx])
        }
    }
    
    logger.info(
        f"Test Evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, "
        f"max_error={max_err_val:.2f} at {metrics['max_error']['timestamp']} "
        f"(actual={metrics['max_error']['actual']:.2f}, pred={metrics['max_error']['predicted']:.2f})"
    )
    return metrics

def run_residual_diagnostics(residuals, diagnostic_lags=[10, 12, 24]):
    """
    Thực hiện kiểm định thống kê toàn diện trên phần dư (Ljung-Box và Jarque-Bera).
    """
    clean_resid = residuals.dropna()
    mean_val = float(clean_resid.mean())
    std_val = float(clean_resid.std())
    skew_val = float(clean_resid.skew())
    kurt_val = float(clean_resid.kurtosis())
    
    # Kiểm định Jarque-Bera tính phân phối chuẩn
    jb_stat, jb_p = stats.jarque_bera(clean_resid)
    
    # Kiểm định Ljung-Box tự tương quan
    valid_lags = [lag for lag in diagnostic_lags if lag < len(clean_resid) // 2]
    lb_df = acorr_ljungbox(clean_resid, lags=valid_lags, return_df=True)
    
    logger.info(f"Residual Properties: N={len(clean_resid)}, Mean={mean_val:.4f}, Std={std_val:.4f}, Skew={skew_val:.3f}, Kurtosis={kurt_val:.3f}")
    logger.info(f"Jarque-Bera Normality Test: stat={jb_stat:.2f}, p_value={jb_p:.4e} -> {'NORMAL' if jb_p > 0.05 else 'NON-NORMAL (Fat tails)'}")
    
    lb_results = {}
    for lag in valid_lags:
        p_val = float(lb_df.loc[lag, "lb_pvalue"])
        stat_val = float(lb_df.loc[lag, "lb_stat"])
        lb_results[lag] = {"stat": round(stat_val, 2), "p_value": round(p_val, 4)}
        if p_val < 0.05:
            logger.warning(f"   [AUTOCORRELATION DETECTED] Ljung-Box at Lag {lag}: p-value={p_val:.4e} < 0.05 -> Residuals contain unmodeled correlation!")
        else:
            logger.info(f"   Ljung-Box at Lag {lag}: p-value={p_val:.4f} >= 0.05 -> White noise behavior at this lag.")
            
    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "skewness": round(skew_val, 3),
        "kurtosis": round(kurt_val, 3),
        "jarque_bera": {"stat": round(float(jb_stat), 2), "p_value": round(float(jb_p), 4)},
        "ljung_box": lb_results
    }

def export_metrics_json(metrics, experiment_name, run_id, output_dir="lab2/artifacts/metrics"):
    """Lưu toàn bộ số liệu đánh giá vào tệp JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{experiment_name}_{run_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics JSON persisted: path={filepath}")
    return filepath

def plot_residual_diagnostics(residuals, experiment_name, suffix=None, output_dir="lab2/artifacts/figures"):
    """
    Xuất bản bộ 4 đồ thị chẩn đoán phần dư theo chuẩn Senior Data Analyst.
    Hỗ trợ suffix='challenger' hoặc 'champion'. Nếu suffix=None thì lưu tên chuẩn.
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_resid = residuals.dropna()
    mean_val = float(clean_resid.mean())
    std_val = float(clean_resid.std())
    skew_val = float(clean_resid.skew())
    kurt_val = float(clean_resid.kurtosis())
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plt.subplots_adjust(hspace=0.28, wspace=0.20)
    
    # 1. Residuals over time
    axes[0, 0].plot(clean_resid.index, clean_resid, color="#2C3E50", lw=1.0, alpha=0.85, label="Residual $e_t$")
    axes[0, 0].axhline(0, color="black", lw=0.8, linestyle="-", alpha=0.5)
    axes[0, 0].axhline(mean_val, color="#E74C3C", lw=1.2, linestyle="--", label=f"Mean ({mean_val:.2f})")
    tag_str = f" [{suffix.upper()}]" if suffix else ""
    axes[0, 0].set_title(f"Residuals vs Time{tag_str}  |  Mean: {mean_val:.2f}, Std: {std_val:.2f}", fontsize=11, fontweight="bold", pad=8)
    axes[0, 0].set_ylabel("Residual", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.35)
    axes[0, 0].legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    
    # 2. Histogram & KDE vs Normal
    sns.histplot(clean_resid, kde=True, ax=axes[0, 1], color="#16A085", stat="density", bins=22, alpha=0.45, edgecolor="none")
    x_grid = np.linspace(clean_resid.min(), clean_resid.max(), 100)
    p_norm = stats.norm.pdf(x_grid, mean_val, std_val)
    axes[0, 1].plot(x_grid, p_norm, color="#7F8C8D", lw=1.5, linestyle="--", label="Normal $N(\\mu, \\sigma^2)$")
    axes[0, 1].set_title(f"Residual Distribution  |  Skew: {skew_val:.2f}, Kurt: {kurt_val:.2f}", fontsize=11, fontweight="bold", pad=8)
    axes[0, 1].set_xlabel("Residual", fontsize=10)
    axes[0, 1].set_ylabel("Density", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.35)
    axes[0, 1].legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    
    # 3. ACF of residuals
    plot_acf(clean_resid, lags=min(30, len(clean_resid)//2 - 1), ax=axes[1, 0], color="#2980B9", title="")
    axes[1, 0].axvline(12, color="#E74C3C", linestyle=":", lw=1.2, alpha=0.9, label="Lag 12 (Seasonal)")
    axes[1, 0].axvline(24, color="#E74C3C", linestyle=":", lw=1.2, alpha=0.9, label="Lag 24 (Seasonal)")
    axes[1, 0].set_title("Residual ACF  |  Examine Spikes at Lags 12 & 24", fontsize=11, fontweight="bold", pad=8)
    axes[1, 0].set_xlabel("Lag", fontsize=10)
    axes[1, 0].set_ylabel("ACF", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.35)
    axes[1, 0].legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    
    # 4. Normal Q-Q plot
    qqplot(clean_resid, line="s", ax=axes[1, 1], markerfacecolor="#2980B9", markeredgecolor="none", alpha=0.6, markersize=5)
    axes[1, 1].get_lines()[1].set_color("#E74C3C")
    axes[1, 1].get_lines()[1].set_linewidth(1.3)
    axes[1, 1].set_title("Normal Q-Q Plot  |  Sample vs Theoretical Quantiles", fontsize=11, fontweight="bold", pad=8)
    axes[1, 1].set_xlabel("Theoretical Quantiles", fontsize=10)
    axes[1, 1].set_ylabel("Sample Quantiles", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.35)
    
    plt.tight_layout()
    filename = f"{experiment_name}_residuals_{suffix}.png" if suffix else f"{experiment_name}_residuals.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Residual diagnostics plot saved (overwritten): path={filepath}")
    return filepath
