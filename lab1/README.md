# HƯỚNG DẪN THỰC HÀNH LAB 1: TIME SERIES EXPLORATION & PREPROCESSING

Thư mục này chứa đầy đủ code, dữ liệu, đồ thị và báo cáo cho **Lab 1: Khám phá, Phân rã & Tiền xử lý chuỗi thời gian**.

---

### 📁 Cấu trúc thư mục

```
lab1/
├── data/                                            # Các file dữ liệu CSV
│   ├── AirPassengers.csv                           # Chuỗi gốc ban đầu (1949 - 1960)
│   ├── AirPassengers_before_processing.csv          # DỮ LIỆU TRƯỚC XỬ LÝ (chứa NaN và Outliers)
│   └── AirPassengers_after_processing.csv           # DỮ LIỆU SAU XỬ LÝ (đã điền khuyết & lọc sạch Outliers)
├── notebooks/
│   └── thuc_hanh_chuong_2.ipynb                    # Jupyter Notebook đầy đủ code, comment và đồ thị
├── plots/                                          # 9 hình ảnh đồ thị chất lượng cao đã render sẵn
│   ├── bai1_01_raw_time_series.png
│   ├── bai1_02_acf_pacf_40lags.png
│   ├── bai2_compare_white_noise_vs_airpassengers.png
│   ├── bai3_01_decomposition_multiplicative.png
│   ├── bai3_02_residual_acf.png
│   ├── bai4_data_cleaning_missing_outliers.png
│   ├── bai5_01_rolling_statistics.png
│   ├── bai5_02_seasonal_plot_monthly.png
│   └── bai5_03_heatmap_seasonality.png
├── reports/                                        # Các tài liệu báo cáo và hướng dẫn chuyên sâu
│   ├── BAO_CAO_CHUONG_2.md                         # Báo cáo tổng hợp hoàn chỉnh 7 mục theo yêu cầu đề bài
│   └── HUONG_DAN_DOC_HIEU_BIEU_DO.md               # Cẩm nang giải thích lý thuyết và cách đọc từng đồ thị
├── requirements/
│   └── image.png                                   # Đề bài gốc thực hành chương 2
└── README.md                                       # Hướng dẫn sử dụng
```

---

### 🚀 Cách kích hoạt môi trường và chạy bài tập

#### 1. Kích hoạt Virtual Environment (nếu dùng Fish shell hoặc Bash):
- **Nếu bạn dùng Fish shell:**
  ```fish
  source /home/thanhlong/Documents/TimeSeries/.venv/bin/activate.fish
  ```
- **Nếu bạn dùng Bash / Zsh:**
  ```bash
  source /home/thanhlong/Documents/TimeSeries/.venv/bin/activate
  ```

#### 2. Mở Jupyter Notebook:
Mở trực tiếp file `notebooks/thuc_hanh_chuong_2.ipynb` trong VS Code / Antigravity IDE (chọn Kernel là `.venv`), hoặc khởi chạy Jupyter Server:
```bash
jupyter notebook notebooks/thuc_hanh_chuong_2.ipynb
```
