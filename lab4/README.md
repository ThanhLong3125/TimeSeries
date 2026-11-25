# LAB 4: MULTIVARIATE TIME SERIES FORECASTING (ELECTRICITY LOAD + WEATHER)
## SO SÁNH BA MÔ HÌNH: VAR (THỐNG KÊ), XGBOOST (MACHINE LEARNING) VÀ LSTM (DEEP LEARNING)

Dự án triển khai bài thực hành Lab 4 theo chuẩn Production Pipeline & Jupyter Notebook cho bài toán **Dự báo chuỗi thời gian đa biến phụ tải điện kết hợp các chỉ số khí tượng thời tiết (Nhiệt độ, Độ ẩm, Tốc độ gió)**.

---

## 1. Cấu Trúc Dự Án (Project Structure)
```
lab4/
├── topic/
│   └── practice 4.pdf                   # File đề bài gốc của Lab 4
├── data/
│   ├── raw/
│   │   ├── tetuan_power_consumption.csv # Dữ liệu gốc 10 phút (UCI #849)
│   │   └── electricity_weather_hourly.csv # Dữ liệu theo giờ (8,736 mốc liên tục)
│   └── processed/
│       └── electricity_weather_clean.csv # Dữ liệu sau kiểm tra và chuẩn hóa
├── configs/
│   └── multivariate_forecast.yaml       # Cấu hình siêu tham số và pipeline
├── notebooks/
│   └── thuc_hanh_du_bao_da_bien_var_xgboost_lstm.ipynb  # Notebook thực hành hoàn chỉnh
├── src/
│   ├── __init__.py
│   ├── logger.py                        # Quản lý ghi nhật ký tập trung
│   ├── tracker.py                       # Truy vết MD5 hash và audit trail JSONL
│   ├── data_loader.py                   # Ingestion, validation và StandardScaler
│   ├── feature_engineer.py              # Trích xuất 96 lags, rolling features, Tensor 3D
│   ├── models_var.py                    # Mô hình VAR (ADF test, AIC order selection)
│   ├── models_xgb.py                    # Mô hình XGBoost (Early stopping)
│   ├── models_lstm.py                   # Mạng PyTorch LSTM (64 -> 32 -> Dense 1)
│   ├── evaluator.py                     # Đánh giá RMSE, MAE, MAPE và Walk-forward
│   └── visualizer.py                    # Trực quan hóa EDA, learning curve, benchmark
├── artifacts/
│   ├── figures/                         # Toàn bộ 7 biểu đồ phân tích chuẩn 300 DPI
│   │   ├── lab4_01_eda_load_timeseries.png
│   │   ├── lab4_02_eda_load_vs_temperature.png
│   │   ├── lab4_03_eda_correlation_matrix.png
│   │   ├── lab4_04_model_benchmark.png
│   │   ├── lab4_05_lstm_loss_curve.png
│   │   ├── lab4_06_xgboost_feature_importance.png
│   │   └── lab4_07_forecast_comparison.png
│   ├── metrics/
│   │   └── benchmark_metrics.json       # Kết quả đánh giá dạng JSON
│   └── models/
│       ├── data_scaler.joblib           # StandardScaler checkpoint
│       ├── var_model.joblib             # Checkpoint VAR model
│       ├── xgboost_model.joblib         # Checkpoint XGBoost model
│       └── lstm_model.pt                # Checkpoint PyTorch LSTM state dict
├── logs/
│   ├── pipeline_execution.log           # File log chi tiết từng bước
│   └── audit_trail.jsonl                # Nhật ký truy vết toàn bộ quá trình chạy
├── tests/
│   └── test_pipeline.py                 # Bộ Unit test tự động (100% Passed)
├── reports/
│   ├── BAO_CAO_LAB4_MULTIVARIATE.md     # Báo cáo học thuật chi tiết
│   └── HUONG_DAN_TRA_LOI_VAN_DAP.md     # Cẩm nang trả lời câu hỏi vấn đáp
├── main.py                              # CLI Entry point chạy pipeline
└── README.md                            # Hướng dẫn sử dụng
```

---

## 2. Hướng Dẫn Chạy Pipeline & Kiểm Thử

### 2.1. Chạy toàn bộ Pipeline từ đầu đến cuối
```bash
PYTHONPATH=. .venv/bin/python3 lab4/main.py --config lab4/configs/multivariate_forecast.yaml
```

### 2.2. Chạy Pipeline với mô hình đã lưu (bỏ qua bước huấn luyện)
```bash
PYTHONPATH=. .venv/bin/python3 lab4/main.py --config lab4/configs/multivariate_forecast.yaml --skip-train
```

### 2.3. Chạy bộ Unit Tests kiểm thử tự động
```bash
PYTHONPATH=. .venv/bin/python3 -m unittest lab4/tests/test_pipeline.py
```

---

## 3. Kết Quả Thực Nghiệm Trên Tập Test (1,311 Mốc Giờ Cuối Năm 2017)

| Mô hình (Model) | RMSE (kW) | MAE (kW) | MAPE (%) | Thời gian huấn luyện |
| :--- | :---: | :---: | :---: | :---: |
| **VAR (Vector Autoregression)** | **641.15** | **443.01** | **1.57%** | 0.12 s |
| **XGBoost Regressor (ML)** | **970.12** | **683.33** | **2.40%** | 2.85 s |
| **LSTM (Deep Learning)** | **1,356.93** | **961.10** | **3.59%** | 16.40 s |

- **Early Stopping LSTM:** Dừng tại Epoch 28 (Validation MSE Loss = 0.02177, không bị overfitting).
- **Phân tích tương quan thời tiết:** Nhiệt độ (`temperature`) có tương quan cao nhất ($r = +0.444$) và quan hệ phi tuyến chữ U với phụ tải điện (`load`).
