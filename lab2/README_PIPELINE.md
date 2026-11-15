# 🚀 Hướng Dẫn Sử Dụng Time Series Production Pipeline (Lab 2)

Hệ thống Pipeline này chuyển đổi toàn bộ logic thực nghiệm từ Jupyter Notebook (`.ipynb`) sang kiến trúc module hóa hướng đối tượng chuẩn công nghiệp, hỗ trợ cấu hình declarative qua YAML, tự động ghi nhật ký đa tầng (Logging), truy vết thí nghiệm bất biến (Audit Trail), và xuất bản tài sản tự động (Artifacts).

---

## 1. Cấu Trúc Hệ Thống

```text
lab2/
├── configs/                      # Tầng cấu hình (YAML)
│   ├── base.yaml                 # Cấu hình ngưỡng mặc định
│   ├── arima_a10.yaml            # Cấu hình bài toán 1: a10 với ARIMA
│   └── sarima_airpassengers.yaml # Cấu hình bài toán 2: AirPassengers với SARIMA
├── src/                          # Tầng mã nguồn cốt lõi (Modules)
│   ├── logger.py                 # Centralized Rotating File Logger
│   ├── tracker.py                # Audit Trail & MD5 Data Hashing
│   ├── data_loader.py            # Ingestion & Schema Validation
│   ├── preprocessor.py           # Datetime alignment & Log Transformation
│   ├── stationarity.py           # ADF with Autolag AIC Logging & Over-diff Safeguard
│   ├── trainer.py                # ARIMA/SARIMAX fitting & Model Serialization
│   ├── evaluator.py              # Performance metrics & Residual Diagnostics
│   └── forecaster.py             # Multi-step Forecasting & Peak Error Detection
├── tests/                        # Tầng kiểm thử tự động
│   └── test_pipeline.py          # Unit test với Python unittest
├── logs/                         # Tầng nhật ký
│   ├── pipeline.log              # Raw log thời gian thực xoay vòng (tối đa 5MB)
│   └── audit_trail.jsonl         # Lịch sử truy vết từng lần chạy (JSON Lines)
├── artifacts/                    # Tầng tài sản đầu ra
│   ├── models/                   # File mô hình nhị phân (.joblib)
│   ├── metrics/                  # Kết quả kiểm thử (.json)
│   └── figures/                  # Biểu đồ chẩn đoán và dự báo ngoài mẫu (.png)
└── main.py                       # CLI Entry Point
```

---

## 2. Hướng Dẫn Chạy Pipeline Qua Dòng Lệnh (CLI)

### Chạy kiểm thử tự động (Unit Tests)
```bash
PYTHONPATH=lab2 python3 -m unittest lab2/tests/test_pipeline.py
```

### Chạy thử nghiệm Dry-run (Chỉ nạp dữ liệu, kiểm tra Schema và tính dừng, không fit model)
```bash
PYTHONPATH=lab2 python3 lab2/main.py --config lab2/configs/arima_a10.yaml --dry-run
```

### Chạy toàn diện Bài toán 1: Doanh số thuốc Úc (`a10` - ARIMA)
```bash
PYTHONPATH=lab2 python3 lab2/main.py --config lab2/configs/arima_a10.yaml
```

### Chạy toàn diện Bài toán 2: Hàng không (`AirPassengers` - SARIMA)
```bash
PYTHONPATH=lab2 python3 lab2/main.py --config lab2/configs/sarima_airpassengers.yaml
```

---

## 3. Các Điểm Nổi Bật Của Hệ Thống

1. **Minh bạch hóa phép tính ngầm của ADF (`regresults=True`):**
   * Hàm `run_adf_with_autolag_logging` bóc tách toàn bộ bảng điểm AIC từ Lag 0 đến Lag cực đại, giải thích vì sao máy tính chọn con số lag chiến thắng.
2. **Bẫy tự động chặn Quá sai phân (Over-differencing Safeguard):**
   * Tự động đo phương sai $\sigma^2(d=0) \rightarrow \sigma^2(d=1) \rightarrow \sigma^2(d=2)$.
   * Nếu phương sai tại $d=2$ tăng vọt và hệ số tự tương quan $r_1 < -0.5$, hệ thống tự động bắn cảnh báo `[WARNING]` vào log.
3. **Cơ chế Audit Trail bất biến (`logs/audit_trail.jsonl`):**
   * Tự động sinh `run_id` duy nhất và tính mã băm MD5 của file CSV đầu vào, đảm bảo tính toàn vẹn dữ liệu và khả năng tái lập 100%.
4. **Biểu đồ Senior Data Analyst:**
   * Không bị chồng chéo chữ (no chart junk), tích hợp chỉ số lên subheader và KPI banner tinh tế, xuất bản độ phân giải cao 300 DPI.
