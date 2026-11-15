import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, Optional

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def save_fig(fig, filename: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_all_plots(
    df_clean: pd.DataFrame,
    df_bench: pd.DataFrame,
    feat_imp: pd.Series,
    y_test: pd.Series,
    predictions: Dict[str, Any],
    df_rec: pd.DataFrame,
    output_dir: str = "lab3/artifacts/figures",
    target_col: str = "AEP_MW"
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. EDA Raw Series
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_clean.index, df_clean[target_col], color='#1f77b4', lw=0.6, alpha=0.8, label='AEP Hourly Demand (MW)')
    ax.axhline(df_clean[target_col].mean(), color='#d62728', linestyle='--', lw=1.5, label=f'Mean: {df_clean[target_col].mean():.0f} MW')
    ax.set_title('Biểu đồ Chuỗi Thời Gian Tiêu Thụ Phụ Tải Điện AEP (2004 - 2018)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Thời gian', fontsize=12)
    ax.set_ylabel('Công suất (MW)', fontsize=12)
    ax.legend(loc='upper right')
    save_fig(fig, "lab3_01_eda_raw_series.png", output_dir)

    # 2. EDA Diurnal Pattern
    df_eda = df_clean.copy()
    df_eda['hour'] = df_eda.index.hour
    hourly_stats = df_eda.groupby('hour')[target_col].agg(['mean', 'std'])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hourly_stats.index, hourly_stats['mean'], marker='o', color='#2ca02c', lw=2.5, label='Trung bình tiêu thụ theo giờ')
    ax.fill_between(hourly_stats.index, hourly_stats['mean'] - hourly_stats['std'], hourly_stats['mean'] + hourly_stats['std'], 
                    alpha=0.2, color='#2ca02c', label='Độ lệch chuẩn (±1 Std)')
    ax.set_title('Chu Kỳ Tiêu Thụ Điện Năng Trong Ngày (Diurnal Pattern: 24 Giờ)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Giờ trong ngày (0 - 23h)', fontsize=12)
    ax.set_ylabel('Phụ tải trung bình (MW)', fontsize=12)
    ax.set_xticks(range(0, 24))
    ax.annotate('Đáy thấp nhất (4 AM)', xy=(4, hourly_stats.loc[4, 'mean']), xytext=(1, 13000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    ax.annotate('Đỉnh chiều tối (19h - 20h)', xy=(19, hourly_stats.loc[19, 'mean']), xytext=(15, 18500),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    ax.legend(loc='lower right')
    save_fig(fig, "lab3_02_eda_daily_pattern.png", output_dir)

    # 3. Model Benchmark Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics = ['MAE', 'RMSE', 'MAPE']
    colors = ['#4c72b0', '#55a868', '#c44e52']
    units = ['MW', 'MW', '%']

    for i, metric in enumerate(metrics):
        axes[i].bar(df_bench['Model'], df_bench[metric], color=colors[i], alpha=0.85)
        axes[i].set_title(f'So sánh {metric} ({units[i]})', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(f'{metric} ({units[i]})')
        for p in axes[i].patches:
            axes[i].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.9),
                             ha='center', va='center', color='white', fontweight='bold')
    save_fig(fig, "lab3_03_model_benchmark.png", output_dir)

    # 4. Feature Importance
    if feat_imp is not None and not feat_imp.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette='viridis', legend=False, ax=ax)
        ax.set_title('Độ Quan Trọng Của Các Đặc Trưng Trong Mô Hình XGBoost (Gain Importance)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Tầm quan trọng tương đối (Importance Score)', fontsize=11)
        save_fig(fig, "lab3_04_feature_importance.png", output_dir)

    # 5. Forecast Sample 2 weeks
    sample_start = '2017-07-01'
    sample_end = '2017-07-14 23:00:00'
    mask_sample = (y_test.index >= sample_start) & (y_test.index <= sample_end)
    if mask_sample.any():
        sample_idx = y_test.index[mask_sample]
        y_true_sample = y_test.loc[sample_idx]
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(sample_idx, y_true_sample, label='Giá trị thực tế (Ground Truth)', color='black', lw=2, marker='.', markersize=4)
        if 'xgboost' in predictions:
            ax.plot(sample_idx, predictions['xgboost'][mask_sample], label='Dự báo XGBoost', color='#d62728', lw=1.8, linestyle='--')
        if 'random_forest' in predictions:
            ax.plot(sample_idx, predictions['random_forest'][mask_sample], label='Dự báo Random Forest', color='#2ca02c', lw=1.5, linestyle=':')
        ax.set_title('So Sánh Chi Tiết Giá Trị Phụ Tải Thực Tế vs Dự Báo (01/07/2017 - 14/07/2017)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Thời gian', fontsize=12)
        ax.set_ylabel('Phụ tải điện (MW)', fontsize=12)
        ax.legend(loc='upper right')
        save_fig(fig, "lab3_05_forecast_sample.png", output_dir)

    # 6. Recursive Multi-step Forecast
    if df_rec is not None and not df_rec.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_rec.index, df_rec['Actual'], marker='o', label='Thực tế (Actual)', color='black', lw=2)
        ax.plot(df_rec.index, df_rec['Recursive_Forecast'], marker='s', label='Dự báo lặp lại XGBoost (Recursive)', color='#d62728', lw=2, linestyle='--')
        ax.set_title('Kết Quả Dự Báo 24 Bước Đa Thời Điểm (Recursive Multi-Step Ahead Forecasting)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Thời gian (Giờ)', fontsize=11)
        ax.set_ylabel('Phụ tải (MW)', fontsize=11)
        ax.legend()
        save_fig(fig, "lab3_06_multistep_recursive.png", output_dir)
