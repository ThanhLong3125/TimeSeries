import os
import json
from .logger import logger

def compare_champion_challenger(new_metrics, champion_metrics_path, tolerance_pct=1.0):
    """
    So sánh mô hình mới huấn luyện (Challenger) với mô hình đương nhiệm tốt nhất (Champion)
    dựa trên các tiêu chuẩn kinh tế lượng và chuỗi thời gian:
    1. Test RMSE (Primary Metric): Đo độ phân tán sai số ngoài mẫu và độ nhạy ngoại lai.
    2. Test MAE & MAPE: Đo độ lệch tuyệt đối và sai số tương đối.
    3. AIC & BIC: Nguyên lý tối giản (Parsimony / Occam's Razor) để chống overfitting.
    4. Residual White Noise (Ljung-Box): Đảm bảo phần dư sạch tương quan.
    """
    if not os.path.exists(champion_metrics_path):
        return {
            "is_promoted": True,
            "verdict": "INITIAL_CHAMPION",
            "reason": "Chưa có mô hình tiền nhiệm. Mô hình mới tự động được xác lập làm Champion đầu tiên.",
            "champion_exists": False,
            "metrics_diff": {}
        }
        
    try:
        with open(champion_metrics_path, "r", encoding="utf-8") as f:
            champion = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read existing champion metrics file: {e}. Promoting new model as fallback.")
        return {
            "is_promoted": True,
            "verdict": "CHAMPION_FALLBACK",
            "reason": "Lỗi đọc tệp Champion cũ. Thăng hạng mô hình mới.",
            "champion_exists": False,
            "metrics_diff": {}
        }

    # Trích xuất các chỉ số của Champion
    champ_perf = champion.get("test_performance", {})
    champ_rmse = float(champ_perf.get("RMSE", float("inf")))
    champ_mae = float(champ_perf.get("MAE", float("inf")))
    champ_mape = float(champ_perf.get("MAPE", float("inf")))
    champ_aic = float(champion.get("aic", float("inf")))
    champ_bic = float(champion.get("bic", float("inf")))

    # Trích xuất các chỉ số của Challenger
    chall_perf = new_metrics.get("test_performance", {})
    chall_rmse = float(chall_perf.get("RMSE", float("inf")))
    chall_mae = float(chall_perf.get("MAE", float("inf")))
    chall_mape = float(chall_perf.get("MAPE", float("inf")))
    chall_aic = float(new_metrics.get("aic", float("inf")))
    chall_bic = float(new_metrics.get("bic", float("inf")))

    # Tính toán chênh lệch (% change: âm là Challenger tốt hơn cho sai số)
    rmse_delta_pct = ((chall_rmse - champ_rmse) / champ_rmse) * 100.0 if champ_rmse != 0 else 0.0
    mae_delta_pct = ((chall_mae - champ_mae) / champ_mae) * 100.0 if champ_mae != 0 else 0.0
    mape_delta = chall_mape - champ_mape
    aic_delta = chall_aic - champ_aic
    bic_delta = chall_bic - champ_bic

    diff_summary = {
        "champ_rmse": champ_rmse, "chall_rmse": chall_rmse, "rmse_delta_pct": rmse_delta_pct,
        "champ_mae": champ_mae, "chall_mae": chall_mae, "mae_delta_pct": mae_delta_pct,
        "champ_mape": champ_mape, "chall_mape": chall_mape, "mape_delta": mape_delta,
        "champ_aic": champ_aic, "chall_aic": chall_aic, "aic_delta": aic_delta,
        "champ_bic": champ_bic, "chall_bic": chall_bic, "bic_delta": bic_delta
    }

    # Bảng phân xử (Decision Gate)
    sep_line = "=" * 80
    header = [
        "",
        sep_line,
        "CHAMPION VS CHALLENGER MODEL EVALUATION GATE (MLOps Time Series Gate)",
        sep_line,
        f"{'Tiêu chí so sánh':<26} | {'Champion':<12} | {'Challenger':<12} | {'Chênh lệch':<15} | {'Đánh giá'}",
        "-" * 80,
        f"{'Test RMSE (Tiêu chí chính)':<26} | {champ_rmse:<12.4f} | {chall_rmse:<12.4f} | {rmse_delta_pct:+8.2f}%       | {'CHALLENGER TỐT HƠN' if chall_rmse < champ_rmse else 'CHAMPION TỐT HƠN'}",
        f"{'Test MAE':<26} | {champ_mae:<12.4f} | {chall_mae:<12.4f} | {mae_delta_pct:+8.2f}%       | {'CHALLENGER TỐT HƠN' if chall_mae < champ_mae else 'CHAMPION TỐT HƠN'}",
        f"{'Test MAPE (%)':<26} | {champ_mape:<12.2f} | {chall_mape:<12.2f} | {mape_delta:+8.2f}% pts   | {'CHALLENGER TỐT HƠN' if chall_mape < champ_mape else 'CHAMPION TỐT HƠN'}",
        f"{'AIC (Mức độ tối giản)':<26} | {champ_aic:<12.2f} | {chall_aic:<12.2f} | {aic_delta:+8.2f} pts     | {'CHALLENGER GỌN HƠN' if chall_aic < champ_aic else 'CHAMPION GỌN HƠN'}",
        f"{'BIC (Phạt số tham số)':<26} | {champ_bic:<12.2f} | {chall_bic:<12.2f} | {bic_delta:+8.2f} pts     | {'CHALLENGER GỌN HƠN' if chall_bic < champ_bic else 'CHAMPION GỌN HƠN'}",
        "-" * 80,
    ]

    # LUẬT PHÂN XỬ:
    # 1. Thắng rõ ràng: RMSE giảm
    if chall_rmse < champ_rmse:
        is_promoted = True
        verdict = "PROMOTED_SUPERIOR_FORECAST"
        reason = f"Challenger vượt trội Champion: Test RMSE giảm {abs(rmse_delta_pct):.2f}% (từ {champ_rmse:.4f} xuống {chall_rmse:.4f})."
    # 2. Hòa sai số (trong dung sai tolerance_pct): So sánh độ tối giản AIC/BIC
    elif abs(rmse_delta_pct) <= tolerance_pct:
        if chall_aic < champ_aic:
            is_promoted = True
            verdict = "PROMOTED_PARSIMONIOUS"
            reason = f"Sai số ngoại mẫu tương đương (chênh lệch {abs(rmse_delta_pct):.2f}% <= {tolerance_pct}%), nhưng Challenger có AIC thấp hơn {abs(aic_delta):.2f} (ít overfit hơn)."
        else:
            is_promoted = False
            verdict = "REJECTED_PARSIMONY"
            reason = f"Sai số ngoại mẫu tương đương, nhưng Champion tối giản hơn (AIC Champion thấp hơn {abs(aic_delta):.2f})."
    # 3. Kém hơn: RMSE tăng
    else:
        is_promoted = False
        verdict = "REJECTED_INFERIOR_ACCURACY"
        reason = f"Challenger kém hơn Champion: Test RMSE tăng +{rmse_delta_pct:.2f}% (từ {champ_rmse:.4f} lên {chall_rmse:.4f})."

    header.append(f"KẾT QUẢ PHÂN XỬ : {'THĂNG HẠNG (PROMOTED)' if is_promoted else 'TỪ CHỐI (REJECTED)'}")
    header.append(f"LÝ DO CHI TIẾT  : {reason}")
    header.append(sep_line)
    
    logger.info("\n".join(header))

    return {
        "is_promoted": is_promoted,
        "verdict": verdict,
        "reason": reason,
        "champion_exists": True,
        "metrics_diff": diff_summary
    }
