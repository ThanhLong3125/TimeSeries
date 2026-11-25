import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def save_fig(fig, filename: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_eda_time_series(df: pd.DataFrame, output_dir: str, target_col: str = "load"):
    """
    1. Biểu đồ load theo thời gian (nhận diện tính chu kỳ và xu hướng)
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df[target_col], color='#1f77b4', lw=0.7, alpha=0.85, label='Điện năng tiêu thụ thực tế (kW)')
    mean_val = df[target_col].mean()
    ax.axhline(mean_val, color='#d62728', linestyle='--', lw=1.5, label=f'Trung bình: {mean_val:,.0f} kW')
    ax.set_title('Biểu Đồ Chuỗi Thời Gian Điện Năng Tiêu Thụ Phụ Tải (Electricity Load)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Thời gian', fontsize=11)
    ax.set_ylabel('Phụ tải (kW)', fontsize=11)
    ax.legend(loc='upper right')
    save_fig(fig, "lab4_01_eda_load_timeseries.png", output_dir)


def plot_load_vs_temperature(df: pd.DataFrame, output_dir: str):
    """
    2. Biểu đồ quan hệ giữa Load và Temperature (hiện tượng U-shape / tương quan điều hòa)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['temperature'], df['load'], c=df['humidity'], cmap='viridis', alpha=0.4, edgecolors='none', s=20)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Độ ẩm (%)', fontsize=10)

    # Đường xu hướng bậc 2 (Polynomial fit để bắt quan hệ phi tuyến U-shape)
    z = np.polyfit(df['temperature'], df['load'], 2)
    p = np.poly1d(z)
    temp_range = np.linspace(df['temperature'].min(), df['temperature'].max(), 200)
    ax.plot(temp_range, p(temp_range), color='#d62728', lw=2.5, label='Đường hồi quy phi tuyến (U-shape Trend)')

    ax.set_title('Mối Quan Hệ Giữa Nhiệt Độ Và Điện Năng Tiêu Thụ (Load vs Temperature)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Nhiệt độ (°C)', fontsize=11)
    ax.set_ylabel('Điện năng tiêu thụ (kW)', fontsize=11)
    ax.legend(loc='upper left')
    save_fig(fig, "lab4_02_eda_load_vs_temperature.png", output_dir)


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str):
    """
    3. Ma trận tương quan (Correlation Matrix Heatmap) giữa 4 biến
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Ma Trận Tương Quan Đa Biến (Electricity Load & Weather Variables)', fontsize=12, fontweight='bold', pad=12)
    save_fig(fig, "lab4_03_eda_correlation_matrix.png", output_dir)


def plot_model_benchmark(df_bench: pd.DataFrame, output_dir: str):
    """
    4. Biểu đồ so chuẩn hiệu năng 3 mô hình (RMSE, MAE, MAPE)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics = ['RMSE', 'MAE', 'MAPE']
    colors = ['#4c72b0', '#55a868', '#c44e52']
    units = ['kW', 'kW', '%']

    for i, metric in enumerate(metrics):
        axes[i].bar(df_bench['Model'], df_bench[metric], color=colors[i], alpha=0.85)
        axes[i].set_title(f'So sánh {metric} ({units[i]})', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(f'{metric} ({units[i]})')
        for p in axes[i].patches:
            axes[i].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.9),
                             ha='center', va='center', color='white', fontweight='bold')
    save_fig(fig, "lab4_04_model_benchmark.png", output_dir)


def plot_lstm_loss_curve(train_losses: List[float], val_losses: List[float], output_dir: str):
    """
    5. Đường cong học tập của mạng LSTM (Train Loss vs Val Loss qua các Epochs)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train MSE Loss', color='#1f77b4', lw=2)
    ax.plot(epochs, val_losses, label='Validation MSE Loss', color='#ff7f0e', lw=2, linestyle='--')
    ax.set_title('Đường Cong Học Tập Của Mạng LSTM (Train Loss vs Validation Loss)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('MSE Loss (Normalized)', fontsize=11)
    ax.legend(loc='upper right')
    save_fig(fig, "lab4_05_lstm_loss_curve.png", output_dir)


def plot_feature_importance(feat_imp: pd.Series, output_dir: str, top_n: int = 15):
    """
    6. Độ quan trọng của các đặc trưng trong XGBoost (Lags điện năng vs Biến thời tiết)
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    top_feats = feat_imp.head(top_n)
    sns.barplot(x=top_feats.values, y=top_feats.index, hue=top_feats.index, palette='magma', legend=False, ax=ax)
    ax.set_title(f'Top {top_n} Đặc Trưng Đóng Góp Lớn Nhất Trong XGBoost (Gain Importance)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Tầm quan trọng tương đối', fontsize=11)
    save_fig(fig, "lab4_06_xgboost_feature_importance.png", output_dir)


def plot_test_forecast_comparison(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    output_dir: str,
    num_hours: int = 168 # 1 tuần mẫu
):
    """
    7. So sánh thực tế vs dự báo của VAR, XGBoost, LSTM trên 1 tuần mẫu của tập Test
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    sub_slice = slice(-num_hours, None)
    t = timestamps[sub_slice]

    ax.plot(t, y_true[sub_slice], label='Giá trị thực tế (Ground Truth)', color='black', lw=2.2, marker='.', markersize=4)

    styles = {
        'VAR (Statistical)': {'color': '#1f77b4', 'ls': '-.', 'lw': 1.6},
        'XGBoost (ML)': {'color': '#2ca02c', 'ls': '--', 'lw': 1.8},
        'LSTM (Deep Learning)': {'color': '#d62728', 'ls': ':', 'lw': 2.0}
    }

    for name, pred in preds_dict.items():
        style = styles.get(name, {'color': 'gray', 'ls': '-', 'lw': 1.5})
        ax.plot(t, pred[sub_slice], label=f'Dự báo {name}', color=style['color'], linestyle=style['ls'], lw=style['lw'])

    ax.set_title('So Sánh Chi Tiết Dự Báo Của 3 Mô Hình Trên Tập Test (Mẫu 1 Tuần = 168 Giờ)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Thời gian', fontsize=11)
    ax.set_ylabel('Điện năng tiêu thụ (kW)', fontsize=11)
    ax.legend(loc='upper right')
    save_fig(fig, "lab4_07_forecast_comparison.png", output_dir)
