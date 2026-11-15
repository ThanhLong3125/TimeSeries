import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from .logger import logger

def run_adf_with_autolag_logging(series, series_name="Time Series", autolag="AIC", alpha=0.05):
    """
    Thực hiện kiểm định ADF và trích xuất TOÀN BỘ nhật ký tính toán AIC của từng bậc lag.
    """
    clean_series = series.dropna()
    res = adfuller(clean_series, autolag=autolag, regresults=True)
    
    adf_stat = float(res[0])
    p_value = float(res[1])
    crit_vals = {k: float(v) for k, v in res[2].items()}
    resstore = res[3]
    
    used_lag = int(resstore.usedlag)
    n_obs = int(resstore.nobs)
    ic_best = float(resstore.icbest)
    
    logger.info(f"--- ADF Test [{series_name}] Autolag Evaluation (Criterion: {autolag}) ---")
    
    # Ghi log toàn bộ các bậc lag mà adfuller đã thử
    for col_idx, ols_res in resstore.autolag_results.items():
        actual_lag = col_idx - 2
        is_winner = (actual_lag == used_lag)
        flag = " [MINIMUM AIC - SELECTED]" if is_winner else ""
        logger.info(f"   Candidate Lag {actual_lag:>2}: AIC={ols_res.aic:10.4f}{flag}")
        
    is_stationary = (p_value < alpha) and (adf_stat < crit_vals["5%"])
    result_status = "STATIONARY (Bác bỏ H0)" if is_stationary else "NON-STATIONARY (Chưa đủ bằng chứng bác bỏ H0)"
    
    logger.info(
        f"ADF Result [{series_name}]: stat={adf_stat:.4f}, p_value={p_value:.4e}, "
        f"used_lag={used_lag}/{resstore.maxlag}, nobs={n_obs}, crit_5%={crit_vals['5%']:.4f} -> {result_status}"
    )
    
    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "nobs": n_obs,
        "crit_vals": crit_vals,
        "ic_best": ic_best,
        "is_stationary": is_stationary,
        "autolag_history": {col_idx - 2: float(ols_res.aic) for col_idx, ols_res in resstore.autolag_results.items()}
    }

def analyze_differencing_orders(series, max_order=2):
    """
    Phân tích biến động Phương sai và Tự tương quan Lag-1 qua từng bậc sai phân (d=0, d=1, d=2)
    để phát hiện bẫy Quá sai phân (Over-differencing).
    """
    logger.info("Evaluating Differencing Orders & Over-differencing Safeguard...")
    diff_results = {}
    current_series = series.copy()
    
    for d in range(max_order + 1):
        if d == 0:
            diff_s = current_series.dropna()
        else:
            current_series = current_series.diff()
            diff_s = current_series.dropna()
            
        var_val = float(diff_s.var())
        std_val = float(diff_s.std())
        r1_val = float(diff_s.autocorr(lag=1)) if len(diff_s) > 2 else np.nan
        
        diff_results[d] = {
            "variance": var_val,
            "std": std_val,
            "lag1_autocorr": r1_val,
            "series": diff_s
        }
        
        if d == 0:
            logger.info(f"   Order d=0 (Original): Variance={var_val:.4f}, Std={std_val:.4f}, Lag1_ACF={r1_val:.4f}")
        else:
            prev_var = diff_results[d - 1]["variance"]
            var_pct_change = ((var_val / prev_var) - 1.0) * 100.0
            logger.info(
                f"   Order d={d}: Variance={var_val:.4f} ({var_pct_change:+.2f}%), "
                f"Std={std_val:.4f}, Lag1_ACF={r1_val:.4f}"
            )
            
            # Kiểm tra Over-differencing
            if var_val > prev_var and r1_val < -0.5:
                logger.warning(
                    f"[OVER-DIFFERENCING DETECTED at d={d}] Variance increased by {var_pct_change:+.1f}% "
                    f"and Lag-1 Autocorrelation {r1_val:.4f} < -0.5! Model should NOT use d={d}."
                )
                
    return diff_results
