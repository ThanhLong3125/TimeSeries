# LAB 3: MACHINE LEARNING CHO DỰ BÁO CHUỖI THỜI GIAN (XGBOOST, RANDOM FOREST & LINEAR REGRESSION)

Dự án triển khai bài thực hành Lab 3 theo chuẩn Production Pipeline & Jupyter Notebook cho bài toán **Dự báo phụ tải điện năng (PJM Hourly Electricity Demand Forecasting - AEP)**.

---

## 1. Cấu Trúc Dự Án (Project Structure)
```
lab3/
├── topic/
│   └── excercise_ML_models.pdf          # Đề bài gốc của Lab 3
├── data/
│   ├── raw/
│   │   └── AEP_hourly.csv               # Dữ liệu gốc 121,273 mốc giờ (2004 - 2018)
│   └── processed/
│       ├── AEP_clean.csv                # Dữ liệu sau khử trùng DST và nội suy
│       └── AEP_features.csv             # Ma trận đặc trưng (Lags, Rolling, Calendar)
├── configs/
│   └── xgboost_aep.yaml                 # File cấu hình siêu tham số và pipeline
├── notebooks/
│   └── thuc_hanh_du_doan_chuoi_thoi_gian_xgboost.ipynb  # Notebook thực hành hoàn chỉnh
├── src/
│   ├── __init__.py
│   ├── logger.py                        # Quản lý nhật ký tập trung
│   ├── tracker.py                       # Truy vết MD5 hash và audit trail JSONL
│   ├── data_loader.py                   # Ingestion, khử trùng lặp và nội suy
│   ├── feature_engineer.py              # Trích xuất đặc trưng Lags, Rolling, Calendar
│   ├── trainer.py                       # Huấn luyện và đóng gói checkpoint (.joblib)
│   └── evaluator.py                     # Đánh giá độ đo và Recursive forecasting
├── artifacts/
│   ├── models/
│   │   ├── linear_regression.joblib     # Model checkpoint
│   │   ├── random_forest.joblib         # Model checkpoint
│   │   └── xgboost.joblib               # Model checkpoint
│   ├── metrics/
│   │   └── benchmark_metrics.json       # Bảng số liệu MAE, RMSE, MAPE JSON
│   └── figures/                         # Toàn bộ biểu đồ lưu trữ trong artifacts
├── logs/
│   ├── pipeline_execution.log           # File log chi tiết từng bước
│   └── audit_trail.jsonl                # Nhật ký truy vết toàn bộ quá trình chạy
├── tests/
│   └── test_pipeline.py                 # Bộ Unit test tự động
├── reports/
│   ├── BAO_CAO_LAB3_XGBOOST.md          # Báo cáo học thuật chi tiết đầy đủ
│   └── HUONG_DAN_TRA_LOI_VAN_DAP.md     # Cẩm nang trả lời câu hỏi vấn đáp
├── main.py                              # CLI Entry point chạy pipeline
└── README.md                            # Hướng dẫn sử dụng
```

---

## 2. Hướng Dẫn Chạy Pipeline & Kiểm Thử

### 2.1. Chạy toàn bộ Pipeline từ đầu đến cuối
```bash
PYTHONPATH=. .venv/bin/python3 lab3/main.py --config lab3/configs/xgboost_aep.yaml
```

### 2.2. Chạy Pipeline với mô hình đã lưu (bỏ qua bước huấn luyện)
```bash
PYTHONPATH=. .venv/bin/python3 lab3/main.py --config lab3/configs/xgboost_aep.yaml --skip-train
```

### 2.3. Chạy bộ Unit Tests kiểm thử tự động
```bash
PYTHONPATH=. .venv/bin/python3 -m unittest lab3/tests/test_pipeline.py
```

---

## 3. Kết Quả Thực Nghiệm Trên Tập Test (2017 - 2018)

| Mô hình | MAE (MW) | RMSE (MW) | MAPE (%) | Thời gian huấn luyện |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 187.23 | 249.54 | 1.27% | 0.08 s |
| **Random Forest** | 136.39 | 182.98 | 0.92% | 16.05 s |
| **XGBoost Regressor** | **120.34** | **158.72** | **0.82%** | **2.10 s** |

- **Recursive Multi-Step Forecasting (24h tiếp theo):** Đạt MAE **300.99 MW** (MAPE **1.83%**).
- **Top Feature Contributions:** `lag_1` (59.21%), `lag_2` (19.44%), `lag_24` (10.09%), `lag_168` (4.55%), `lag_3` (1.47%).
