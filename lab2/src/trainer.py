import os
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .logger import logger

def log_detailed_model_summary(fitted_model, train_series, model_type, order, seasonal_order=None, dataset_name="Time Series Data", experiment_name="model"):
    """
    Xuất bản báo cáo tóm tắt mô hình chuỗi thời gian toàn diện (Metadata + Statsmodels Summary + Diễn giải thống kê)
    chuẩn mực kinh tế lượng và kỹ thuật mô hình hóa.
    """
    n_obs = len(train_series)
    try:
        start_date = train_series.index.min().strftime("%Y-%m-%d %H:%M:%S")
        end_date = train_series.index.max().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        start_date = str(train_series.index.min())
        end_date = str(train_series.index.max())

    p, d, q = order
    if seasonal_order:
        P, D, Q, s = seasonal_order
        model_title = f"SARIMAX({p}, {d}, {q}) x ({P}, {D}, {Q})[{s}]"
    else:
        model_title = f"ARIMA({p}, {d}, {q})"

    sep_line = "=" * 80
    header_lines = [
        "",
        sep_line,
        f"{model_type.upper()} MODEL ESTIMATION SUMMARY — {experiment_name.upper()}",
        sep_line,
        f"Data                   : {dataset_name}",
        f"Date range             : {start_date} to {end_date}",
        f"Number of observations : {n_obs}",
        f"AR order (p)           : {p}",
        f"Differencing order (d) : {d}",
        f"MA order (q)           : {q}",
    ]

    if seasonal_order:
        header_lines.extend([
            f"Seasonal AR order (P)  : {P}",
            f"Seasonal diff (D)      : {D}",
            f"Seasonal MA order (Q)  : {Q}",
            f"Seasonal period (s)    : {s}",
        ])

    header_lines.extend([
        f"Model Specification    : {model_title}",
        sep_line,
        f"STATISTICAL ESTIMATION RESULTS (STATSMODELS)",
        sep_line,
        str(fitted_model.summary()),
        sep_line,
        "CHẨN ĐOÁN HỘI TỤ, TÍNH DỪNG & KHẢ NGHỊCH (STABILITY & INVERTIBILITY):"
    ])

    # 1. Trạng thái hội tụ
    converged = bool(getattr(fitted_model.mle_retvals, "converged", True))
    iterations = int(getattr(fitted_model.mle_retvals, "iterations", -1))
    header_lines.append(f" • Thuật toán tối ưu (MLE) : {'HỘI TỤ THÀNH CÔNG' if converged else 'CHƯA HỘI TỤ'} (Số vòng lặp: {iterations})")

    # 2. Kiểm tra tính dừng & khả nghịch qua nghiệm đặc trưng (Roots)
    if hasattr(fitted_model, "arroots") and len(fitted_model.arroots) > 0:
        ar_roots = fitted_model.arroots
        min_ar_mod = float(np.min(np.abs(ar_roots)))
        ar_status = "DỪNG (Stationary)" if min_ar_mod > 1.0 else "KHÔNG DỪNG (Cảnh báo nghiệm nằm trong/trên vòng tròn đơn vị!)"
        header_lines.append(f" • Nghiệm AR (Min Modulus) : {min_ar_mod:.4f} (> 1.0 -> {ar_status})")

    if hasattr(fitted_model, "maroots") and len(fitted_model.maroots) > 0:
        ma_roots = fitted_model.maroots
        min_ma_mod = float(np.min(np.abs(ma_roots)))
        ma_status = "KHẢ NGHỊCH (Invertible)" if min_ma_mod > 1.0 else "KHÔNG KHẢ NGHỊCH (Cảnh báo non-invertible!)"
        header_lines.append(f" • Nghiệm MA (Min Modulus) : {min_ma_mod:.4f} (> 1.0 -> {ma_status})")

    # 3. Tổng hợp ý nghĩa thống kê của các hệ số (p-value)
    if hasattr(fitted_model, "pvalues"):
        sig_params = []
        insig_params = []
        for param, pval in fitted_model.pvalues.items():
            if pval < 0.05:
                sig_params.append(f"{param} (p={pval:.4f})")
            else:
                insig_params.append(f"{param} (p={pval:.4f})")
        header_lines.append(f" • Tham số có ý nghĩa (p < 0.05)    : {', '.join(sig_params) if sig_params else 'None'}")
        if insig_params:
            header_lines.append(f" • Tham số không có ý nghĩa (p >= 0.05): {', '.join(insig_params)}")

    header_lines.append(sep_line)
    
    # Ghi toàn bộ nội dung summary đẹp mắt vào log
    logger.info("\n".join(header_lines))

def train_time_series_model(train_series, model_type="ARIMA", order=(1, 1, 1), seasonal_order=None, dataset_name="Time Series Data", experiment_name="model"):
    """
    Huấn luyện mô hình ARIMA hoặc SARIMAX và kiểm tra trạng thái nghiệm/hội tụ.
    """
    logger.info(f"Training Model: type={model_type}, order={order}, seasonal_order={seasonal_order}")
    
    if model_type.upper() == "ARIMA":
        model_instance = ARIMA(train_series, order=order)
        fitted_model = model_instance.fit()
    elif model_type.upper() == "SARIMAX":
        model_instance = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model_instance.fit(disp=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    # Trích xuất và ghi log tóm tắt chi tiết (giống định dạng chuyên gia tham khảo)
    log_detailed_model_summary(
        fitted_model=fitted_model,
        train_series=train_series,
        model_type=model_type,
        order=order,
        seasonal_order=seasonal_order,
        dataset_name=dataset_name,
        experiment_name=experiment_name
    )
            
    return fitted_model

def save_model_artifact(fitted_model, experiment_name, run_id, output_dir="lab2/artifacts/models"):
    """Lưu trữ mô hình đã huấn luyện thành file nhị phân .joblib."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{experiment_name}_{run_id}.joblib"
    filepath = os.path.join(output_dir, filename)
    joblib.dump(fitted_model, filepath)
    logger.info(f"Model artifact persisted: path={filepath}")
    return filepath
