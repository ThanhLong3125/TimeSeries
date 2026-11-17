# 📈 Time Series Analysis & Forecasting Portfolio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA%20%7C%20SARIMA-orange.svg)](https://www.statsmodels.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-red.svg)](https://xgboost.readthedocs.io/)
[![Code Style](https://img.shields.io/badge/Architecture-Production--Ready%20Pipeline-brightgreen.svg)]()

> **Kho lưu trữ toàn diện về Phân tích & Dự báo Chuỗi thời gian (Time Series Analysis & Forecasting)** — Triển khai từ các phương pháp Thống kê cổ điển (Classical Statistical Models: ARIMA, SARIMA) đến Mô hình Học máy hiện đại (Machine Learning: XGBoost, Random Forest, Feature Engineering), tuân thủ kiến trúc **Production-Grade MLOps Pipeline**, bộ Unit Test tự động và hệ thống báo cáo học thuật chuyên sâu.

---

## 📑 Mục lục (Table of Contents)

1. [Tổng quan các bài thực hành](#-tổng-quan-các-bài-thực-hành)
   - [Lab 1: Khám phá, Phân rã & Tiền xử lý dữ liệu](#-lab-1-khám-phá-phân-rã--tiền-xử-lý-chuỗi-thời-gian)
   - [Lab 2: Kiểm định Dừng (ADF) & Mô hình hóa ARIMA / SARIMA](#-lab-2-kiểm-định-adf--mô-hình-hóa-arima--sarima)
   - [Lab 3: Dự báo Phụ tải điện với Machine Learning (XGBoost)](#-lab-3-dự-báo-phụ-tải-điện-với-machine-learning-xgboost)
2. [Cấu trúc Repository](#-cấu-trúc-repository)
3. [Cài đặt môi trường](#-cài-đặt-môi-trường)
4. [Hướng dẫn thực thi](#-hướng-dẫn-thực-thi)
   - [Chạy Production Pipeline qua CLI](#1-chạy-production-pipeline-qua-cli)
   - [Chạy Unit Test kiểm thử chất lượng](#2-chạy-unit-test-kiểm-thử-chất-lượng)
   - [Khám phá với Jupyter Notebook](#3-khám-phá-với-jupyter-notebook)
5. [Tài liệu báo cáo chuyên sâu](#-tài-liệu-báo-cáo-chuyên-sâu)

---

## 🔬 Tổng quan các bài thực hành

### 🛫 Lab 1: Khám phá, Phân rã & Tiền xử lý Chuỗi thời gian
* **Dữ liệu thực nghiệm**: `AirPassengers` (1949 – 1960).
* **Nội dung trọng tâm**:
  * Trực quan hóa chuỗi thời gian, phân tích hàm tự tương quan (ACF) và tự tương quan riêng phần (PACF).
  * Phân rã chuỗi thời gian (Additive & Multiplicative Decomposition) thành Trend, Seasonality và Residuals.
  * Phân biệt chuỗi dừng (Stationary), chuỗi có xu hướng/mùa vụ với chuỗi Nhiễu trắng (White Noise).
  * Tiền xử lý dữ liệu: Phát hiện và xử lý giá trị khuyết thiếu (Missing Values) bằng Time-interpolation; phát hiện ngoại lai (Outliers) bằng kỹ thuật Rolling Statistics & IQR.
  * Phân tích đặc tính chu kỳ/mùa vụ qua Seasonal Subseries Plot và Heatmap trực quan.
* 👉 Chi tiết: [lab1/README.md](lab1/README.md)

---

### 📉 Lab 2: Kiểm định ADF & Mô hình hóa ARIMA / SARIMA
* **Dữ liệu thực nghiệm**: `a10` (Doanh số bán thuốc trị tiểu đường tại Úc) & `AirPassengers`.
* **Nội dung trọng tâm**:
  * **Kiểm định tính dừng**: Triển khai kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF) với cơ chế tự động tìm bậc trễ tối ưu dựa trên AIC (`autolag='AIC'`).
  * **Cơ chế chống Sai phân quá mức (Over-differencing Safeguard)**: Phân tích phương sai và hệ số tự tương quan bậc 1 ($ACF_1 < -0.5$) để tránh làm tăng phương sai khi lấy sai phân bậc cao ($d=2$).
  * **Mô hình hóa Box-Jenkins**: Triển khai mô hình ARIMA($p, d, q$) cho chuỗi đơn biến và SARIMA($p, d, q$)($P, D, Q$)$_s$ cho chuỗi có mùa vụ mạnh.
  * **Chẩn đoán phần dư chuẩn hóa**: Kiểm định Ljung-Box (độc lập thống kê), Jarque-Bera & D'Agostino-Pearson (phân phối chuẩn), Heteroskedasticity (phương sai đồng nhất).
  * **Kiến trúc Pipeline & Cổng thẩm định Champion vs. Challenger ($O(1)$)**: Tách biệt rõ ràng các module (`data_loader`, `preprocessor`, `stationarity`, `trainer`, `evaluator`, `forecaster`, `model_comparator`), quản lý siêu tham số bằng YAML (`configs/`), cơ chế Staging lưu song song tài sản Champion & Challenger phục vụ đối chiếu trực quan mà không gây phình to ổ đĩa, tích hợp Audit Trail JSONL và Unit tests tự động.
* 👉 Chi tiết: [lab2/README.md](lab2/README.md) | [lab2/README_PIPELINE.md](lab2/README_PIPELINE.md)

---

### ⚡ Lab 3: Dự báo Phụ tải điện với Machine Learning (XGBoost)
* **Dữ liệu thực nghiệm**: `AEP_hourly` (American Electric Power - hơn 121,000 bản ghi phụ tải điện theo giờ từ 2004 đến 2018).
* **Nội dung trọng tâm**:
  * **Feature Engineering đa quy mô**:
    * *Đặc trưng trễ (Lag Features)*: Lags ngắn hạn (1h, 2h, 3h), chu kỳ ngày (24h, 48h) và chu kỳ tuần (168h).
    * *Đặc trưng thống kê trượt (Rolling Statistics)*: Rolling Mean, Std, Min, Max với các cửa sổ 24h và 168h (loại trừ rò rỉ dữ liệu).
    * *Đặc trưng lịch và chu kỳ*: Hour, Day of Week, Month, Quarter, Weekend, Hour Sine/Cosine cyclical encoding.
  * **Chống rò rỉ dữ liệu (Data Leakage Prevention)**: Áp dụng TimeSeriesSplit / Chronological Split tuần tự theo trục thời gian, tuyệt đối không xáo trộn (no shuffle).
  * **Benchmark 3 thuật toán**: So sánh định lượng giữa **Linear Regression** (Baseline), **Random Forest** (Non-linear Bagging), và **XGBoost** (Gradient Boosting).
  * **Dự báo đệ quy đa bước (Multi-step Recursive Forecasting)**: Thuật toán tự hồi quy cập nhật động giá trị trễ và rolling stats cho đường chân trời dự báo $H=168$ giờ (1 tuần).
  * **Đo lường & Phân tích lỗi**: MAE, RMSE, MAPE cùng phân tích Feature Importance qua Weight, Gain và Cover.
* 👉 Chi tiết: [lab3/README.md](lab3/README.md)

---

## 📁 Cấu trúc Repository

```text
TimeSeries/
├── .gitignore                           # Git ignore chuẩn hóa (loại trừ weights nặng, logs, cache)
├── requirements.txt                     # Danh mục dependencies tối ưu hóa
├── README.md                            # Cẩm nang tổng quan dự án
│
├── lab1/                                # LAB 1: Khám phá, Phân rã & Tiền xử lý
│   ├── data/                            # Dữ liệu AirPassengers (gốc, trước/sau tiền xử lý)
│   ├── notebooks/                       # Jupyter Notebook thực hành tương tác
│   ├── plots/                           # 9 biểu đồ trực quan hóa độ phân giải cao
│   ├── reports/                         # Báo cáo học thuật & Cẩm nang đọc hiểu biểu đồ
│   └── README.md                        # Hướng dẫn chi tiết Lab 1
│
├── lab2/                                # LAB 2: Kiểm định ADF & ARIMA / SARIMA Pipeline
│   ├── configs/                         # Cấu hình siêu tham số YAML (a10 & AirPassengers)
│   ├── data/                            # Dữ liệu raw và processed
│   ├── src/                             # Mã nguồn module pipeline (DataOps, Trainer,...)
│   ├── artifacts/                       # Kết quả mô hình: figures/, metrics/, models/
│   ├── logs/                            # Runtime execution logs & Audit trail
│   ├── notebooks/                       # 2 Notebooks phân tích chi tiết và đối sánh
│   ├── reports/                         # Báo cáo khoa học & Hướng dẫn diễn giải chỉ số
│   ├── tests/                           # Unit tests tự động cho pipeline
│   ├── main.py                          # CLI entry point thực thi pipeline
│   └── README.md                        # Cẩm nang toàn diện Lab 2
│
└── lab3/                                # LAB 3: Machine Learning & MLOps Pipeline
    ├── configs/                         # Cấu hình XGBoost, Random Forest, Lags, Rolling
    ├── data/                            # Dữ liệu phụ tải điện AEP hourly
    ├── src/                             # Mã nguồn module Feature Engineering & Modeling
    ├── artifacts/                       # Kết quả mô hình: figures/, metrics/, models/
    ├── logs/                            # Execution logs & Audit trail
    ├── notebooks/                       # Notebook thực hành chuyên sâu
    ├── reports/                         # Báo cáo phân tích kỹ thuật & Câu hỏi vấn đáp
    ├── tests/                           # Unit tests kiểm định rò rỉ dữ liệu & đa bước
    ├── main.py                          # CLI entry point thực thi pipeline
    └── README.md                        # Hướng dẫn chi tiết Lab 3
```

---

## 🛠 Cài đặt môi trường

### 1. Khởi tạo Virtual Environment
```bash
# Tạo môi trường ảo Python 3.10+
python3 -m venv .venv

# Kích hoạt trên Linux / macOS:
source .venv/bin/activate

# Kích hoạt trên Fish shell:
source .venv/bin/activate.fish

# Kích hoạt trên Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn thực thi

### 1. Chạy Production Pipeline qua CLI

Toàn bộ quy trình từ tải dữ liệu, tiền xử lý, huấn luyện, sinh biểu đồ và trích xuất chỉ số metric đều được tự động hóa qua CLI:

```bash
# --- LAB 2: Chạy mô hình ARIMA cho dữ liệu a10 ---
python lab2/main.py --config lab2/configs/arima_a10.yaml

# --- LAB 2: Chạy mô hình SARIMA cho dữ liệu AirPassengers ---
python lab2/main.py --config lab2/configs/sarima_airpassengers.yaml

# --- LAB 3: Chạy huấn luyện, Benchmark và Recursive Forecasting AEP Hourly ---
python lab3/main.py --config lab3/configs/xgboost_aep.yaml
```

### 2. Chạy Unit Test kiểm thử chất lượng

Kiểm thử tính toàn vẹn của dữ liệu, kiểm định ADF, logic sai phân chống over-differencing, và xác nhận không có data leakage trong feature engineering:

```bash
# Kiểm thử Unit Test Lab 2:
PYTHONPATH=lab2 python -m unittest discover -s lab2/tests -p "test_*.py"

# Kiểm thử Unit Test Lab 3:
PYTHONPATH=lab3 python -m unittest discover -s lab3/tests -p "test_*.py"
```

### 3. Khám phá với Jupyter Notebook

Khởi động Jupyter Lab để xem trực quan từng cell phân tích, đồ thị tương tác và diễn giải học thuật:

```bash
jupyter lab
```
* **Lab 1**: `lab1/notebooks/thuc_hanh_chuong_2.ipynb`
* **Lab 2**: `lab2/notebooks/lab2.ipynb` hoặc `lab2/notebooks/thuc_hanh_kiem_dinh_adf_arima_sarima.ipynb`
* **Lab 3**: `lab3/notebooks/thuc_hanh_du_doan_chuoi_thoi_gian_xgboost.ipynb`

---

## 📊 Kết quả thực nghiệm tóm tắt (Benchmark Highlights)

### Lab 3: So sánh hiệu năng trên tập Test AEP (24,255 giờ thực tế)

| Mô hình | MAE (MW) | RMSE (MW) | MAPE (%) | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 2,058.46 | 2,572.78 | 13.79% | Tuyến tính đơn giản, không học được tương tác phi tuyến |
| **Random Forest** | 988.61 | 1,489.15 | 6.54% | Nắm bắt tốt đặc tính phi tuyến nhưng mô hình nặng (~52MB) |
| **XGBoost (Champion)** | **883.84** | **1,332.22** | **5.77%** | **Vượt trội toàn diện**: Sai số thấp nhất, tốc độ suy luận nhanh |

---

## 📚 Tài liệu báo cáo chuyên sâu

* **Lab 1**:
  * [Báo cáo thực hành chi tiết Lab 1](lab1/reports/BAO_CAO_CHUONG_2.md)
  * [Cẩm nang đọc hiểu biểu đồ Lab 1](lab1/reports/HUONG_DAN_DOC_HIEU_BIEU_DO.md)
* **Lab 2**:
  * [Báo cáo khoa học đầy đủ ARIMA / SARIMA](lab2/reports/BAO_CAO_LAB2_ARIMA_SARIMA.md)
  * [Cơ sở lý thuyết toán học chuyên sâu](lab2/reports/CO_SO_LY_THUYET_CAC_MO_HINH_CHIEU_SAU.md)
  * [Cẩm nang diễn giải các chỉ số thống kê & P-values](lab2/reports/HUONG_DAN_DIEN_GIAI_CHI_SO.md)
  * [Hướng dẫn đọc hiểu 9 biểu đồ thực nghiệm](lab2/reports/HUONG_DAN_DOC_HIEU_BIEU_DO.md)
* **Lab 3**:
  * [Báo cáo khoa học hoàn chỉnh XGBoost & Machine Learning](lab3/reports/BAO_CAO_LAB3_XGBOOST.md)
  * [Cẩm nang trả lời câu hỏi vấn đáp bảo vệ đồ án](lab3/reports/HUONG_DAN_TRA_LOI_VAN_DAP.md)

---

## 👤 Tác giả
* **GitHub**: [@ThanhLong3125](https://github.com/ThanhLong3125)
