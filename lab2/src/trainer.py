import os
import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .logger import logger

def train_time_series_model(train_series, model_type="ARIMA", order=(1, 1, 1), seasonal_order=None):
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
        
    # Trích xuất nhật ký tối ưu hóa
    aic_val = float(fitted_model.aic)
    bic_val = float(fitted_model.bic)
    llf_val = float(fitted_model.llf)
    converged = bool(getattr(fitted_model.mle_retvals, "converged", True))
    iterations = int(getattr(fitted_model.mle_retvals, "iterations", -1))
    
    logger.info(
        f"Optimization Result: converged={converged}, iterations={iterations}, "
        f"LogLikelihood={llf_val:.3f}, AIC={aic_val:.2f}, BIC={bic_val:.2f}"
    )
    
    # Kiểm tra nghiệm đặc trưng AR & MA (Characteristic roots)
    if hasattr(fitted_model, "arroots") and len(fitted_model.arroots) > 0:
        ar_roots = fitted_model.arroots
        min_ar_mod = float(np.min(np.abs(ar_roots)))
        logger.info(f"   AR Roots Modulus (min): {min_ar_mod:.4f} (Stationary if > 1.0)")
        if min_ar_mod <= 1.0:
            logger.warning("   [STABILITY WARNING] Minimum AR root modulus <= 1.0! Non-stationary process detected.")
            
    if hasattr(fitted_model, "maroots") and len(fitted_model.maroots) > 0:
        ma_roots = fitted_model.maroots
        min_ma_mod = float(np.min(np.abs(ma_roots)))
        logger.info(f"   MA Roots Modulus (min): {min_ma_mod:.4f} (Invertible if > 1.0)")
        if min_ma_mod <= 1.0:
            logger.warning("   [INVERTIBILITY WARNING] Minimum MA root modulus <= 1.0! Non-invertible process detected.")
            
    return fitted_model

def save_model_artifact(fitted_model, experiment_name, run_id, output_dir="lab2/artifacts/models"):
    """Lưu trữ mô hình đã huấn luyện thành file nhị phân .joblib."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{experiment_name}_{run_id}.joblib"
    filepath = os.path.join(output_dir, filename)
    joblib.dump(fitted_model, filepath)
    logger.info(f"Model artifact persisted: path={filepath}")
    return filepath
