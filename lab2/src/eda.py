import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from .logger import logger

def plot_eda_raw_series(series, config, run_id=None, output_dir="lab2/artifacts/figures"):
    """
    Vẽ biểu đồ EDA chuỗi thời gian gốc: Mean, Min, Max, và xu hướng phát triển.
    Ghi đè trực tiếp lên ảnh cũ nếu không truyền run_id.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_name = config.get("experiment_name", "model")
    unit = config.get("evaluation", {}).get("unit", "Value")
    
    mean_val = float(series.mean())
    min_val = float(series.min())
    max_val = float(series.max())
    min_idx = series.idxmin()
    max_idx = series.idxmax()
    
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(series.index, series.values, color="#1F3A60", lw=1.8, label="Dữ liệu thô thực tế")
    
    # Đường mức trung bình
    ax.axhline(mean_val, color="#E74C3C", linestyle="--", lw=1.3, label=f"Mean: {mean_val:.2f} {unit}")
    
    # Điểm cực trị Min & Max
    ax.scatter([min_idx], [min_val], color="#8E44AD", s=60, zorder=5)
    ax.annotate(f"Min: {min_val:.2f} ({min_idx.strftime('%Y-%m')})", xy=(min_idx, min_val),
                xytext=(min_idx, min_val + (max_val - min_val) * 0.08),
                arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1.2),
                fontsize=9, fontweight="bold", color="#8E44AD")
                
    ax.scatter([max_idx], [max_val], color="#27AE60", s=60, zorder=5)
    ax.annotate(f"Max: {max_val:.2f} ({max_idx.strftime('%Y-%m')})", xy=(max_idx, max_val),
                xytext=(max_idx - pd.DateOffset(months=18), max_val - (max_val - min_val) * 0.08),
                arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2),
                fontsize=9, fontweight="bold", color="#27AE60")
                
    # Vùng phân tán Mean +- 1 Std
    std_val = float(series.std())
    ax.axhspan(mean_val - std_val, mean_val + std_val, color="#E74C3C", alpha=0.08, label=f"Mean ± 1 Std ({std_val:.2f})")
    
    ax.set_title(f"EDA Chuỗi Thời Gian Gốc — {config.get('dataset_name', exp_name)}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Thời gian (Năm)", fontsize=11)
    ax.set_ylabel(unit, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    
    plt.tight_layout()
    filename = f"{exp_name}_eda_01_raw_{run_id}.png" if run_id else f"{exp_name}_eda_01_raw.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"EDA Raw series plot saved (overwritten): path={filepath}")
    return filepath

def plot_eda_differencing(diff_results, config, run_id=None, output_dir="lab2/artifacts/figures"):
    """
    Vẽ biểu đồ so sánh chuỗi sau sai phân bậc 1 (d=1) và bậc 2 (d=2) kèm theo dõi phương sai.
    Ghi đè trực tiếp lên ảnh cũ nếu không truyền run_id.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_name = config.get("experiment_name", "model")
    
    s_d1 = diff_results[1]["series"]
    var_d1 = diff_results[1]["variance"]
    r1_d1 = diff_results[1]["lag1_autocorr"]
    
    has_d2 = (2 in diff_results)
    if has_d2:
        s_d2 = diff_results[2]["series"]
        var_d2 = diff_results[2]["variance"]
        r1_d2 = diff_results[2]["lag1_autocorr"]
        var_change = ((var_d2 / var_d1) - 1.0) * 100.0
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
        
        # d=1
        ax1.plot(s_d1.index, s_d1.values, color="#2E7D32", lw=1.3, label="Sai phân bậc 1 (d=1)")
        ax1.axhline(0, color="black", linestyle="--", alpha=0.5, lw=0.9)
        ax1.set_title(f"Sai phân Bậc 1 (d=1) — Phương sai tối ưu: {var_d1:.4f} | Lag-1 ACF: {r1_d1:.3f}", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Giá trị d=1", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(loc="upper left", fontsize=9)
        
        # d=2
        ax2.plot(s_d2.index, s_d2.values, color="#00796B", lw=1.3, label="Sai phân bậc 2 (d=2)")
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5, lw=0.9)
        flag = " [Quá sai phân / Over-differencing]" if (var_d2 > var_d1 and r1_d2 < -0.5) else ""
        ax2.set_title(f"Sai phân Bậc 2 (d=2) — Phương sai: {var_d2:.4f} ({var_change:+.1f}%) | Lag-1 ACF: {r1_d2:.3f}{flag}", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Thời gian (Năm)", fontsize=10)
        ax2.set_ylabel("Giá trị d=2", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(loc="upper left", fontsize=9)
    else:
        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax1.plot(s_d1.index, s_d1.values, color="#2E7D32", lw=1.3, label="Sai phân bậc 1 (d=1)")
        ax1.axhline(0, color="black", linestyle="--", alpha=0.5, lw=0.9)
        ax1.set_title(f"Sai phân Bậc 1 (d=1) — Phương sai: {var_d1:.4f} | Lag-1 ACF: {r1_d1:.3f}", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Thời gian (Năm)", fontsize=10)
        ax1.set_ylabel("Giá trị d=1", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(loc="upper left", fontsize=9)
        
    plt.tight_layout()
    filename = f"{exp_name}_eda_02_differenced_{run_id}.png" if run_id else f"{exp_name}_eda_02_differenced.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"EDA Differenced series plot saved (overwritten): path={filepath}")
    return filepath

def plot_eda_acf_pacf(series, config, run_id=None, output_dir="lab2/artifacts/figures", max_lags=40):
    """
    Vẽ biểu đồ ACF và PACF trên chuỗi đã sai phân để nhận dạng bậc (p, d, q).
    Ghi đè trực tiếp lên ảnh cũ nếu không truyền run_id.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_name = config.get("experiment_name", "model")
    clean_series = series.dropna()
    actual_lags = min(max_lags, len(clean_series) // 2 - 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5))
    
    # ACF
    plot_acf(clean_series, lags=actual_lags, ax=ax1, color="#1F3A60", title="")
    ax1.axvline(12, color="#E74C3C", linestyle=":", lw=1.2, alpha=0.8, label="Seasonal Lag 12")
    ax1.axvline(24, color="#E74C3C", linestyle=":", lw=1.2, alpha=0.8, label="Seasonal Lag 24")
    ax1.set_title("Hàm Tự Tương Quan (ACF) — Xác định bậc Moving Average (q)", fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylabel("Hệ số ACF", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=9)
    
    # PACF
    plot_pacf(clean_series, lags=actual_lags, ax=ax2, color="#27AE60", method="ywm", title="")
    ax2.axvline(12, color="#E74C3C", linestyle=":", lw=1.2, alpha=0.8, label="Seasonal Lag 12")
    ax2.axvline(24, color="#E74C3C", linestyle=":", lw=1.2, alpha=0.8, label="Seasonal Lag 24")
    ax2.set_title("Hàm Tự Tương Quan Từng Phần (PACF) — Xác định bậc Autoregressive (p)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Độ trễ (Lags)", fontsize=10)
    ax2.set_ylabel("Hệ số PACF", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    filename = f"{exp_name}_eda_03_acf_pacf_{run_id}.png" if run_id else f"{exp_name}_eda_03_acf_pacf.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"EDA ACF/PACF plot saved (overwritten): path={filepath}")
    return filepath
