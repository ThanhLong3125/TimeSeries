# LAB 2: KIỂM ĐỊNH ADF & MÔ HÌNH HÓA ARIMA / SARIMA
> **Cẩm nang Toàn diện: Cơ sở Lý thuyết Toán học – Hướng dẫn Đọc hiểu Biểu đồ – Báo cáo Kết quả & Diễn giải Chỉ số Thống kê**

---

## 📑 MỤC LỤC
1. [Tổng quan Đề bài & Cấu trúc Thư mục](#1-tổng-quan-đề-bài--cấu-trúc-thư-mục)
2. [Cơ sở Lý thuyết & Nền tảng Toán học](#2-cơ-sở-lý-thuyết--nền-tảng-toán-học)
   - 2.1. Tính dừng (Stationarity) và Lý do bắt buộc chuỗi phải dừng
   - 2.2. Kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF)
   - 2.3. Ổn định phương sai: Mô hình Nhân tính vs Biến đổi Logarit
   - 2.4. Sai phân thường ($d$) vs Sai phân mùa vụ ($D$) & Nguy cơ Over-differencing
   - 2.5. Quy trình Box-Jenkins & Nhận diện qua hàm ACF / PACF
   - 2.6. Cấu trúc toán học của họ mô hình ARIMA và SARIMA
   - 2.7. Tiêu chuẩn chẩn đoán phần dư chuẩn hóa (White Noise)
   - 2.8. Hệ thống đo lường sai số dự báo (AIC, BIC, MAE, RMSE, MAPE)
3. [Cẩm nang Đọc hiểu & Diễn giải Toàn bộ 9 Biểu đồ Thực nghiệm](#3-cẩm-nang-đọc-hiểu--diễn-giải-toàn-bộ-9-biểu-đồ-thực-nghiệm)
   - [Biểu đồ 1] ds1_01_eda_raw_series.png – Chuỗi gốc Monthly Drug Sales (a10)
   - [Biểu đồ 2] ds1_02_differenced_series.png – Chuỗi sau sai phân bậc 1
   - [Biểu đồ 3] ds1_03_acf_pacf.png – Biểu đồ ACF & PACF xác định (p, d, q) & Giải mã 3 điểm dị biệt
   - [Biểu đồ 4] ds1_04_residual_diagnostics.png – Chẩn đoán 4 tiêu chuẩn phần dư ARIMA
   - [Biểu đồ 5] ds1_05_forecast_arima.png – Đồ thị dự báo ngoài mẫu Train/Test ARIMA
   - [Biểu đồ 6] ds2_01_eda_seasonality.png – Chuỗi AirPassengers nhận diện mùa vụ & Heteroskedasticity
   - [Biểu đồ 7] ds2_02_seasonal_diff.png – So sánh sai phân thường vs Sai phân kép
   - [Biểu đồ 8] ds2_03_acf_pacf_sarima.png – Biểu đồ ACF/PACF xác định (p, q, P, Q)
   - [Biểu đồ 9] ds2_04_sarima_vs_arima.png – Đối chuẩn hiệu năng SARIMA vs ARIMA thuần
4. [Tổng hợp Kết quả & Giải đáp 16 Câu hỏi Cốt lõi](#4-tổng-hợp-kết-quả--giải-đáp-16-câu-hỏi-cốt-lõi)
   - Giải đáp 9 câu hỏi trọng tâm Bài 5 (Dataset 1: a10)
   - Giải đáp 7 bước nâng cao (Dataset 2: AirPassengers)
5. [Hướng dẫn Khởi chạy Notebook](#5-hướng-dẫn-khởi-chạy-notebook)

---

## 1. TỔNG QUAN ĐỀ BÀI & CẤU TRÚC THƯ MỤC

### Đề tài thực hành:
1. **Phần A (Dataset 1 – `a10.csv`)**: Dự báo Doanh số Dược phẩm trị tiểu đường hàng tháng tại Úc (1991 – 2008, 204 quan sát). Nhiệm vụ: EDA, kiểm định ADF chuỗi gốc, lấy sai phân bậc 1, phân tích ACF/PACF, huấn luyện và đánh giá đối đầu $\text{ARIMA}(0, 1, 1)$ vs $\text{ARIMA}(1, 1, 1)$, trả lời 9 câu hỏi chuyên sâu.
2. **Phần B (Dataset 2 – `AirPassengers.csv`)**: Dự báo Lượng hành khách hàng không quốc tế hàng tháng (1949 – 1960, 144 quan sát). Nhiệm vụ: Nhận diện mô hình nhân tính, kiểm định ADF, biến đổi $\ln(Y_t)$, thực hiện sai phân kép ($d=1, D=1, s=12$), phân tích ACF/PACF mùa vụ, huấn luyện $\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$, chẩn đoán phần dư White Noise, dự báo ngoài mẫu và so sánh đối chuẩn hiệu năng với ARIMA phi mùa vụ.

### Cấu trúc thư mục:
```text
lab2/
├── data/                                # Dữ liệu thực hành
│   ├── raw/                             # Dữ liệu gốc
│   │   ├── a10.csv                      # Dataset 1 thô
│   │   └── AirPassengers.csv            # Dataset 2 thô
│   └── processed/                       # Dữ liệu sau tiền xử lý
│       ├── a10_processed.csv            # a10 đã set DatetimeIndex, freq='MS', diff_1
│       └── AirPassengers_processed.csv  # AirPassengers đã log_passengers, diff_1, diff_1_12
│
├── notebooks/                           # Jupyter Notebook thực hành
│   ├── lab2.ipynb                       # Notebook hoàn chỉnh chuyên sâu theo cấu trúc đề bài (Đầy đủ lý thuyết toán, code sạch, biểu đồ và trả lời trọn vẹn 16 câu hỏi)
│   └── thuc_hanh_kiem_dinh_adf_arima_sarima.ipynb   # Notebook quy trình chuẩn hóa Pipeline (Tải raw -> Tiền xử lý -> Huấn luyện -> Đối chuẩn)
│
├── plots/                               # Toàn bộ 9 biểu đồ chất lượng cao trích xuất từ thực nghiệm (300 DPI)
│   ├── ds1_01_eda_raw_series.png        # Biểu đồ 1: Đồ thị chuỗi gốc a10 tích hợp Mean, Min, Max
│   ├── ds1_02_differenced_series.png    # Biểu đồ 2: Đồ thị chuỗi sai phân bậc 1 (d=1) quanh trục 0
│   ├── ds1_03_acf_pacf.png              # Biểu đồ 3: Đồ thị ACF & PACF chuỗi sai phân a10
│   ├── ds1_04_residual_diagnostics.png  # Biểu đồ 4: Chẩn đoán 4 tiêu chuẩn phần dư ARIMA(0,1,1)
│   ├── ds1_05_forecast_arima.png        # Biểu đồ 5: Đồ thị dự báo ngoài mẫu Train/Test ARIMA(0,1,1) & ARIMA(1,1,1)
│   ├── ds2_01_eda_seasonality.png       # Biểu đồ 6: Đồ thị chuỗi AirPassengers nhận diện mùa vụ và Mean, Min, Max
│   ├── ds2_02_seasonal_diff.png         # Biểu đồ 7: So sánh sai phân thường vs Sai phân kép (d=1, D=1, s=12)
│   ├── ds2_03_acf_pacf_sarima.png       # Biểu đồ 8: Đồ thị ACF & PACF xác định tham số SARIMA
│   └── ds2_04_sarima_vs_arima.png       # Biểu đồ 9: Đồ thị so sánh đối chuẩn dự báo SARIMA vs ARIMA thuần
│
├── reports/                             # Tài liệu nghiên cứu và cẩm nang diễn giải
│   ├── BAO_CAO_LAB2_ARIMA_SARIMA.md     # Báo cáo khoa học đầy đủ lý thuyết, phân tích và câu trả lời
│   ├── CO_SO_LY_THUYET_CAC_MO_HINH_CHIEU_SAU.md # Cơ sở lý thuyết toàn diện về AR, MA, ARMA, ARIMA, SARIMA, SARIMAX
│   ├── HUONG_DAN_DOC_HIEU_BIEU_DO.md    # Cẩm nang chuyên sâu giải thích chi tiết 9 biểu đồ thực nghiệm
│   └── HUONG_DAN_DIEN_GIAI_CHI_SO.md    # Bảng tra cứu ý nghĩa các chỉ số thống kê
│
├── excercise_stats_models.pdf           # Đề bài thực hành gốc
├── excercise_ML_models.pdf              # Đề bài thực hành Machine Learning (tham khảo)
└── pseudo_code_stats_model.pdf          # Mã giả hướng dẫn thuật toán
```

---

## 2. CƠ SỞ LÝ THUYẾT & NỀN TẢNG TOÁN HỌC

### 2.1. Bản chất cốt lõi: Phân biệt rõ "Phương sai" (Variance) và "Sai phân" (Differencing)

Hai khái niệm này có tên phát âm gần giống nhau trong tiếng Việt nhưng mang bản chất toán học và vai trò hoàn toàn khác biệt trong phân tích chuỗi thời gian:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHƯƠNG SAI (Variance - Var): Thước đo độ phân tán, vung vẩy quanh trung bình │
│                                                                             │
│ SAI PHÂN (Differencing - Δ): Phép trừ triệt tiêu xu hướng (y(t) - y(t-1))    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### A. PHƯƠNG SAI (Variance – Ký hiệu: $\sigma^2$ hoặc $\text{Var}(Y)$)
- **Định nghĩa toán học**:
  $$\text{Var}(Y) = \sigma^2 = \mathbb{E}[(Y - \mu)^2] = \frac{1}{N}\sum_{t=1}^N (y_t - \mu)^2$$
- **Bản chất trực quan**:
  - Đo lường **mức độ phân tán, độ rải rác và biên độ dao động** của các điểm dữ liệu xung quanh giá trị trung bình ($\mu$).
  - *Phương sai nhỏ*: Các quan sát nằm co cụm, bám sát rạt quanh đường trung bình $\implies$ Dữ liệu ổn định, độ biến động thấp, độ rủi ro thấp.
  - *Phương sai lớn*: Các quan sát vung vẩy dữ dội, liên tục tạo các đỉnh nhọn vọt lên cao và đáy cắm sâu xuống dưới $\implies$ Dữ liệu biến động mạnh, bất ổn định cao.
  - *Độ lệch chuẩn ($\sigma$ - Standard Deviation)*: Là căn bậc hai của phương sai $\sigma = \sqrt{\text{Var}(Y)}$, có ưu điểm là mang cùng đơn vị đo với dữ liệu gốc (triệu USD, nghìn hành khách).
- **Phương sai trong chuỗi thời gian (Tính dừng về phương sai)**:
  - **Phương sai đồng nhất (Homoskedasticity)**: Chuỗi đạt tính dừng phương sai khi biên độ dao động duy trì ổn định qua mọi thời kỳ: $\text{Var}(y_t) = \sigma^2 = \text{const}$.
  - **Phương sai thay đổi (Heteroskedasticity)**: Chuỗi có biên độ dao động phình to dần theo thời gian (như đồ thị AirPassengers từ năm 1949 đến 1960). Hiện tượng này vi phạm tính dừng và **sai phân không thể tự xử lý được**, bắt buộc phải áp dụng **biến đổi Logarit $\ln(Y_t)$** để nén phẳng phương sai.

---

#### B. SAI PHÂN (Differencing – Ký hiệu: $\Delta$ hoặc $\nabla$)
- **Định nghĩa toán học**:
  Là phép toán **lấy giá trị của thời điểm hiện tại trừ đi giá trị của một thời điểm trong quá khứ**:
  - *Sai phân thường bậc 1 ($d=1$)*: $\Delta y_t = y_t - y_{t-1} = (1 - B)y_t$.
  - *Sai phân thường bậc 2 ($d=2$)*: $\Delta^2 y_t = \Delta(\Delta y_t) = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) = y_t - 2y_{t-1} + y_{t-2}$.
  - *Sai phân mùa vụ bậc 1 ($D=1, s=12$)*: $\Delta_{12} y_t = y_t - y_{t-12} = (1 - B^{12})y_t$.
- **Bản chất trực quan & Ẩn dụ vật lý**:
  - Sai phân chuyển bài toán từ việc khảo sát **mức độ tuyệt đối** (vốn đang leo dốc không ngừng) sang khảo sát **tốc độ thay đổi / mức tăng giảm** giữa các thời điểm.
  - *Ẩn dụ chiếc xe hơi*:
    - Chuỗi gốc $y_t$ giống như **Quãng đường** chiếc xe đã đi (luôn dốc lên tăng dần theo thời gian $\implies$ không dừng).
    - Sai phân bậc 1 $\Delta y_t$ giống như **Vận tốc** của xe (quãng đường đi được mỗi giờ). Nếu xe chạy đều, vận tốc sẽ dao động ổn định quanh 60 km/h $\implies$ **Chuỗi đã đạt tính dừng!**
    - Sai phân bậc 2 $\Delta^2 y_t$ giống như **Gia tốc** của xe (sự thay đổi của vận tốc).

---

#### C. MỐI QUAN HỆ & PHÂN BIỆT GIỮA SAI PHÂN VÀ PHƯƠNG SAI
1. **Sai phân xử lý cái gì?** $\rightarrow$ Sai phân xử lý **Xu hướng (Trend)**, đưa giá trị trung bình $\mathbb{E}[y_t]$ về mức ổn định quanh 0.
2. **Biến đổi Logarit xử lý cái gì?** $\rightarrow$ Logarit xử lý **Phương sai (Variance)**, nén biên độ dao động mùa vụ hình phễu về mức đồng đều.
3. **Quy tắc phối hợp chuẩn mực Box-Jenkins**:
   $$\text{Chuỗi gốc mang mô hình nhân tính } Y_t \xrightarrow[\text{Ổn định PHƯƠNG SAI}]{\text{Lấy Logarit } \ln(Y_t)} z_t \xrightarrow[\text{Ổn định KỲ VỌNG / Khử Trend}]{\text{Lấy SAI PHÂN } \Delta \Delta_{12}} \text{Chuỗi DỪNG HOÀN TOÀN}$$
4. **Cảnh báo Over-differencing (Sai phân quá mức làm bùng nổ phương sai)**:
   - Khi chưa đủ sai phân ($d=0$), chuỗi phi dừng có phương sai rất lớn.
   - Khi lấy đúng bậc sai phân ($d=1$), phương sai giảm xuống mức tối thiểu (chuỗi dừng tốt nhất).
   - Nếu lạm dụng lấy thừa sai phân ($d=2$), phương sai sẽ **bị khuếch đại tăng vọt trở lại** (ở chuỗi a10, phương sai nhảy từ $2.78$ lên $7.03$). Đây là dấu hiệu nhận biết mô hình đã bị quá sai phân!

---

### 2.2. Tính dừng (Stationarity) và Lý do bắt buộc chuỗi phải dừng
- **Định nghĩa tính dừng yếu (Weak / Covariance Stationarity)**: Chuỗi $\{y_t\}$ được gọi là dừng yếu khi thỏa mãn đồng thời 3 điều kiện:
  1. $\mathbb{E}[y_t] = \mu < \infty$ (Kỳ vọng trung bình không đổi theo thời gian).
  2. $\text{Var}(y_t) = \sigma^2 < \infty$ (Phương sai hữu hạn và không đổi).
  3. $\text{Cov}(y_t, y_{t-k}) = \gamma_k$ (Hiệp phương sai giữa 2 mốc thời gian chỉ phụ thuộc vào khoảng cách độ trễ $k$, hoàn toàn độc lập với vị trí $t$).
- **Vì sao bắt buộc chuỗi phải dừng?**
  - **Tránh hồi quy giả mạo (Spurious Regression)**: Hai chuỗi phi dừng có cùng xu hướng thời gian khi hồi quy với nhau sẽ cho hệ số $R^2$ rất cao và $t$-statistic có ý nghĩa giả tạo dù về mặt thực tế chúng hoàn toàn độc lập.
  - **Khái quát hóa tham số (Parameter Invariance)**: Các mô hình thống kê học trọng số từ lịch sử quá khứ để dự báo tương lai. Nếu phân phối xác suất trôi dạt vô định (kỳ vọng tăng hoặc phương sai mở rộng), các tham số ước lượng ở quá khứ sẽ trở nên vô giá trị đối với tương lai.
  - **Điều kiện khả nghịch và ổn định**: Đa thức đặc trưng của quá trình tự hồi quy chỉ hội tụ khi các nghiệm nằm ngoài vòng tròn đơn vị (Unit Circle).

---

### 2.2. Kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF Test)
Kiểm định ADF kiểm tra sự hiện diện của nghiệm đơn vị (Unit Root) thông qua phương trình hồi quy dạng sai phân:
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{k} \delta_i \Delta y_{t-i} + \epsilon_t$$
Trong đó $\gamma = \rho - 1$ ($\rho$ là hệ số tự hồi quy bậc 1).
- **Giả thuyết vô hiệu ($H_0$)**: $\gamma = 0$ (tương đương $\rho = 1$) $\implies$ **Chuỗi chứa nghiệm đơn vị $\implies$ Chuỗi KHÔNG DỪNG (Non-stationary)**.
- **Đối thuyết ($H_1$)**: $\gamma < 0$ (tương đương $|\rho| < 1$) $\implies$ **Chuỗi không có nghiệm đơn vị $\implies$ Chuỗi DỪNG (Stationary)**.
- **Quy tắc quyết định**:
  - Dưới giả thuyết $H_0$, thống kê kiểm định $\text{ADF} = \frac{\hat{\gamma}}{\text{SE}(\hat{\gamma})}$ không tuân theo phân phối Student $t$ chuẩn mà tuân theo **phân phối Dickey-Fuller phi chuẩn** (lệch hẳn về phía âm).
  - So sánh giá trị thống kê ADF với các giá trị tới hạn (Critical Values ở mức $1\%, 5\%, 10\%$):
    $$\text{Nếu } \text{ADF Statistic} < \text{Critical Value}_{5\%} \text{ (và } p\text{-value} < 0.05\text{)} \implies \text{Bác bỏ } H_0 \implies \text{Chuỗi DỪNG}$$
    $$\text{Nếu } \text{ADF Statistic} \ge \text{Critical Value}_{5\%} \text{ (và } p\text{-value} \ge 0.05\text{)} \implies \text{Chưa thể bác bỏ } H_0 \implies \text{Chuỗi PHI DỪNG}$$

---

### 2.3. Ổn định phương sai: Mô hình Nhân tính vs Biến đổi Logarit
- **Mô hình cộng tính (Additive Seasonality)**: $Y_t = T_t + S_t + I_t$. Biên độ dao động mùa vụ $S_t$ giữ nguyên không đổi qua mọi năm bất kể mức độ xu hướng $T_t$ tăng hay giảm.
- **Mô hình nhân tính (Multiplicative Seasonality)**: $Y_t = T_t \times S_t \times I_t$. Khi mức trung bình chuỗi tăng, biên độ dao động mùa vụ cũng tăng theo cấp số nhân (hiện tượng phương sai thay đổi - Heteroskedasticity).
- **Cơ sở toán học của phép biến đổi Logarit**:
  $$\ln(Y_t) = \ln(T_t \times S_t \times I_t) = \ln(T_t) + \ln(S_t) + \ln(I_t)$$
  Phép lấy logarit tự nhiên biến đổi mô hình nhân tính thành mô hình cộng tính tuyến tính, nén chặt sự bùng nổ phương sai, đưa biên độ dao động mùa vụ về mức đồng đều (Homoskedasticity) qua mọi năm. Đây là điều kiện tiên quyết trước khi áp dụng các phép sai phân tuyến tính.

---

### 2.4. Sai phân thường ($d$) vs Sai phân mùa vụ ($D$) & Nguy cơ Over-differencing
- **Sai phân thường bậc 1 ($d = 1$)**:
  $$\Delta y_t = (1 - B) y_t = y_t - y_{t-1}$$
  *Tác dụng*: Triệt tiêu xu hướng tăng trưởng tuyến tính dài hạn (loại bỏ nghiệm đơn vị tại tần số 0).
- **Sai phân mùa vụ bậc 1 ($D = 1, s = 12$)**:
  $$\Delta_{12} y_t = (1 - B^{12}) y_t = y_t - y_{t-12}$$
  *Tác dụng*: Triệt tiêu tính tuần hoàn mùa vụ lặp lại hàng năm (loại bỏ nghiệm đơn vị mùa vụ tại tần số $2\pi/12$).
- **Sai phân kép ($d = 1, D = 1, s = 12$)**:
  $$\Delta \Delta_{12} y_t = (1 - B)(1 - B^{12}) y_t = (y_t - y_{t-1}) - (y_{t-12} - y_{t-13})$$
  *Tác dụng*: Đưa chuỗi chứa cả xu hướng và mùa vụ về trạng thái dừng hoàn toàn.
- **Nguy cơ của Sai phân quá mức (Over-differencing)**:
  - Nếu lấy sai phân vượt quá bậc cần thiết (ví dụ nâng lên $d=2$ khi $d=1$ đã đủ khử xu hướng), phương sai của chuỗi không giảm mà bị khuếch đại tăng vọt (ở Dataset 1, $\text{Var}$ tăng từ $2.780$ lên $7.026$, tăng gấp 2.5 lần).
  - Đồng thời, over-differencing ép hệ số tự tương quan tại lag 1 xuống mức âm cực mạnh ($\rho_1 \approx -0.55$), đưa một nghiệm đơn vị trung bình trượt nhân tạo vào mô hình ($MA(1) \approx -1$), làm mô hình mất đi tính khả nghịch (Invertibility).

---

### 2.5. Quy trình Box-Jenkins & Nhận diện qua hàm ACF / PACF
Quy trình Box-Jenkins (1970) gồm 4 bước chuẩn mực:
1. **Nhận dạng mô hình (Identification)**: Đánh giá tính dừng bằng ADF; chọn bậc sai phân $d, D$; dùng đồ thị ACF và PACF để nhận diện bậc tự hồi quy ($p, P$) và trung bình trượt ($q, Q$).
2. **Ước lượng tham số (Estimation)**: Sử dụng phương pháp Hợp lý cực đại (Maximum Likelihood Estimation - MLE) hoặc Bình phương tối thiểu phi tuyến (NLS).
3. **Chẩn đoán mô hình (Diagnostic Checking)**: Kiểm tra xem phần dư có phải là Nhiễu trắng (White Noise) hay không (Kỳ vọng $\approx 0$, không còn tự tương quan, phân phối chuẩn).
4. **Dự báo (Forecasting)**: Dự báo ngoài mẫu (Out-of-sample) và tính khoảng tin cậy.

#### Bảng tra cứu Box-Jenkins kinh điển:
| Thành phần | Miền độ trễ | Biểu hiện ACF | Biểu hiện PACF | Nhận diện bậc |
| :--- | :--- | :--- | :--- | :--- |
| **Phi mùa vụ** | Ngắn hạn ($k = 1, 2, 3$) | Cắt cụt sau lag $q$ | Giảm dần (Tails off) | $\text{MA}(q)$ ($p=0$) |
| | | Giảm dần (Tails off) | Cắt cụt sau lag $p$ | $\text{AR}(p)$ ($q=0$) |
| | | Giảm dần | Giảm dần | Hỗn hợp $\text{ARMA}(p, q)$ |
| **Mùa vụ** | Bội số chu kỳ ($k = s, 2s, 3s$) | Cắt cụt sau lag $Qs$ | Giảm dần tại $s, 2s$ | $\text{SMA}(Q)$ ($P=0$) |
| | | Giảm dần tại $s, 2s$ | Cắt cụt sau lag $Ps$ | $\text{SAR}(P)$ ($Q=0$) |
| | | Giảm dần | Giảm dần | Hỗn hợp $\text{SARMA}(P, Q)_s$ |

---

### 2.6. Cấu trúc toán học & Giải thích chi tiết ý nghĩa các tham số ARIMA(p,d,q) và SARIMA(p,d,q)(P,D,Q)s

Mô hình Box-Jenkins phân rã mọi chuỗi thời gian thành các cơ chế toán học xử lý xu hướng, quán tính quá khứ và các cú sốc ngẫu nhiên.

---

#### A. GIẢI THÍCH CHI TIẾT CÁC THAM SỐ $(p, d, q)$ TRONG MÔ HÌNH ARIMA PHI MÙA VỤ

Phương trình toán học tổng quát:
$$\phi_p(B)(1 - B)^d y_t = c + \theta_q(B)\epsilon_t$$
$$(1 - \phi_1 B - \phi_2 B^2 - \dots - \phi_p B^p)(1 - B)^d y_t = c + (1 + \theta_1 B + \theta_2 B^2 + \dots + \theta_q B^q)\epsilon_t$$

| Tham số | Tên gọi | Ký hiệu hệ số | Ý nghĩa toán học & Cơ chế hoạt động | Ẩn dụ đời thường dễ nhớ |
| :---: | :--- | :---: | :--- | :--- |
| **$p$** | **Bậc tự hồi quy** *(Autoregressive)* | $\phi_1, \dots, \phi_p$ | Mô hình hóa giá trị hiện tại $y_t$ theo tổ hợp tuyến tính của **$p$ giá trị quá khứ liền kề** ($y_{t-1}, \dots, y_{t-p}$). Thể hiện mối liên hệ nhân quả nối tiếp theo thời gian. | **"Quán tính / Trí nhớ ngắn hạn"**: Giá trị ngày hôm nay thừa hưởng bao nhiêu phần từ ngày hôm qua và ngày hôm kia. Ví dụ $p=1$: Hôm nay bán được nhiều thì ngày mai thường vẫn bán được nhiều. |
| **$d$** | **Bậc sai phân tích hợp** *(Integrated)* | $\nabla^d = (1-B)^d$ | Số lần lấy sai phân liên tiếp để **loại bỏ xu hướng (Trend)** và nghiệm đơn vị, đưa chuỗi phi dừng về chuỗi dừng. $d=0$ là chuỗi dừng sẵn; $d=1$ khử xu hướng tuyến tính; $d=2$ khử xu hướng phi tuyến bậc 2. | **"Bộ lọc khử xu hướng"**: Thay vì dự đoán giá trị tuyệt đối $y_t$ (vốn đang leo dốc liên tục), ta chuyển sang dự đoán **mức độ tăng giảm (sai phân $\Delta y_t = y_t - y_{t-1}$)** quanh mức 0. |
| **$q$** | **Bậc trung bình trượt** *(Moving Average)* | $\theta_1, \dots, \theta_q$ | Mô hình hóa giá trị hiện tại theo tổ hợp tuyến tính của **$q$ cú sốc ngẫu nhiên / sai số dự báo quá khứ** ($\epsilon_{t-1}, \dots, \epsilon_{t-q}$). | **"Cơ chế tự sửa sai sau cú sốc"**: Nếu hôm qua xảy ra một cú sốc bất ngờ (tin tức giật gân, thời tiết cực đoan) khiến dự báo bị lệch một lượng $\epsilon_{t-1}$, thì hôm nay mô hình sẽ tự động hiệu chỉnh một tỷ lệ $\theta_1 \epsilon_{t-1}$ để bù đắp sai số đó. |
| **$c$** | **Hệ số tự do** *(Constant / Drift)* | $c$ | • Khi $d=0$: $c = \mu(1 - \sum \phi_i)$, đại diện cho mức trung bình dài hạn $\mu$.<br>• Khi $d=1$: $c$ là **tốc độ tăng trưởng trôi dạt (Drift)**. Nếu $c > 0$, chuỗi sau sai phân có xu hướng dốc lên tự nhiên theo thời gian. | **"Mức tăng trưởng nền tảng"**: Mức tăng trưởng tự nhiên mặc định mỗi tháng ngay cả khi không có cú sốc nào xảy ra. |

##### Điều kiện hoạt động bắt buộc:
1. **Tính dừng của AR ($p$)**: Mọi nghiệm của đa thức $1 - \phi_1 z - \dots - \phi_p z^p = 0$ phải nằm ngoài vòng tròn đơn vị ($|z| > 1$). Với $p=1$, điều kiện là $|\phi_1| < 1$.
2. **Tính khả nghịch của MA ($q$) (Invertibility)**: Mọi nghiệm của đa thức $1 + \theta_1 z + \dots + \theta_q z^q = 0$ phải nằm ngoài vòng tròn đơn vị ($|z| > 1$). Với $q=1$, điều kiện là $|\theta_1| < 1$, đảm bảo các cú sốc trong quá khứ xa xôi giảm dần ảnh hưởng chứ không bị khuếch đại vô hạn.

---

#### B. GIẢI THÍCH CHI TIẾT CÁC THAM SỐ MÙA VỤ $(P, D, Q)_s$ TRONG MÔ HÌNH SARIMA

Phương trình toán học tổng quát:
$$\Phi_P(B^s) \phi_p(B) (1 - B)^d (1 - B^s)^D y_t = \Theta_Q(B^s) \theta_q(B) \epsilon_t$$

Trong đó:
$$(1 - \Phi_1 B^s - \dots - \Phi_P B^{Ps}) \phi_p(B) (1 - B)^d (1 - B^s)^D y_t = (1 + \Theta_1 B^s + \dots + \Theta_Q B^{Qs}) \theta_q(B) \epsilon_t$$

| Tham số | Tên gọi | Ký hiệu hệ số | Ý nghĩa toán học & Cơ chế hoạt động | Ẩn dụ đời thường dễ nhớ |
| :---: | :--- | :---: | :--- | :--- |
| **$s$** | **Chu kỳ mùa vụ** *(Seasonal Period)* | $s$ | Độ dài bước trễ của một chu kỳ lặp lại tuần hoàn. Với dữ liệu tháng lặp lại theo năm thì $s = 12$; dữ liệu quý thì $s = 4$; dữ liệu ngày theo tuần thì $s = 7$. | **"Độ dài một mùa / một năm"**: Khoảng thời gian để một chu kỳ kinh doanh quay trở lại điểm xuất phát. |
| **$P$** | **Bậc tự hồi quy mùa vụ** *(Seasonal AR)* | $\Phi_1, \dots, \Phi_P$ | Liên kết giá trị thời điểm hiện tại $y_t$ với **chính thời điểm này của các chu kỳ trước** ($y_{t-s}, y_{t-2s}, \dots$). | **"Trí nhớ cùng kỳ năm ngoái"**: Ví dụ $P=1, s=12$: Lượng khách đi máy bay du lịch tháng 7 năm nay chịu ảnh hưởng trực tiếp từ lượng khách tháng 7 năm ngoái. |
| **$D$** | **Bậc sai phân mùa vụ** *(Seasonal Integrated)* | $\nabla_s^D = (1 - B^s)^D$ | Số lần lấy **sai phân bước nhảy $s$**: $\Delta_s y_t = y_t - y_{t-s}$. Triệt tiêu hoàn toàn dạng sóng dao động tuần hoàn hàng năm (khử nghiệm đơn vị mùa vụ). | **"Bộ lọc khử sóng mùa vụ"**: So sánh tháng 7 năm nay trực tiếp với tháng 7 năm ngoái để loại bỏ yếu tố mùa hè định kỳ. |
| **$Q$** | **Bậc trung bình trượt mùa vụ** *(Seasonal MA)* | $\Theta_1, \dots, \Theta_Q$ | Điều chỉnh giá trị dự báo hiện tại dựa trên **cú sốc sai số của cùng kỳ các chu kỳ trước** ($\epsilon_{t-s}, \epsilon_{t-2s}, \dots$). | **"Sửa sai theo cú sốc cùng kỳ"**: Nếu tháng 7 năm ngoái có sự kiện đặc biệt (dịch bệnh, lễ hội lớn) gây sai số $\epsilon_{t-12}$, tham số $\Theta_1$ sẽ tự động điều chỉnh dự báo cho tháng 7 năm nay. |

##### Tóm tắt tương quan giữa các tham số:
- $(p, d, q)$: Nắm bắt **quan hệ ngắn hạn giữa các tháng liên tiếp** ($t$ so với $t-1, t-2$).
- $(P, D, Q)_s$: Nắm bắt **quan hệ dài hạn giữa các năm cùng kỳ** ($t$ so với $t-12, t-24$).
- Khi kết hợp nhân chéo $[\phi_p(B) \Phi_P(B^s)]$, mô hình tự động tạo ra các tương quan chéo phức tạp (như độ trễ $s-1$ và $s+1$, giải thích triệt để các gai dị biệt ở lag 11 và lag 13).

---

### 2.7. Tiêu chuẩn chẩn đoán phần dư chuẩn hóa (White Noise)
Phần dư là phần sai lệch giữa dữ liệu thực tế và giá trị khớp mẫu: $e_t = y_t - \hat{y}_t$. Một mô hình hoàn hảo phải trích xuất hết mọi thông tin có quy luật, biến phần dư thành **Nhiễu trắng (White Noise)** thỏa mãn 4 tiêu chí:
1. **Trung bình kỳ vọng $\approx 0$**: Không có sai số chệch hệ thống.
2. **Phương sai không đổi (Homoskedasticity)**: Biên độ dao động của phần dư ngẫu nhiên và ổn định theo thời gian.
3. **Phân phối Chuẩn (Normality)**: Histogram và đường KDE ôm sát phân phối chuẩn $N(0, 1)$, biểu đồ Normal Q-Q Plot bám sát đường thẳng 45 độ.
4. **Không có tự tương quan (Uncorrelated)**: Toàn bộ các hệ số tự tương quan ACF của phần dư phải nằm trọn bên trong dải tin cậy 95%. Kiểm định **Ljung-Box** trên nhiều độ trễ ($k = 6, 12, 24$) phải có:
   $$p\text{-value} > 0.05 \implies \text{Không thể bác bỏ } H_0 \implies \text{Phần dư ĐỘC LẬP (White Noise)}$$

---

### 2.8. Hệ thống đo lường sai số dự báo (AIC, BIC, MAE, RMSE, MAPE)
- **Akaike Information Criterion (AIC)** & **Bayesian Information Criterion (BIC)**:
  $$\text{AIC} = 2k - 2\ln(\hat{L})$$
  $$\text{BIC} = k\ln(n) - 2\ln(\hat{L})$$
  - Thưởng cho mô hình có hàm hợp lý $\hat{L}$ cao và phạt số lượng tham số $k$.
  - *Lưu ý quan trọng*: Không được chọn mô hình chỉ dựa vào AIC/BIC huấn luyện nội mẫu. Mô hình ít tham số hơn (Parsimonious) thường tổng quát hóa và dự báo ngoài mẫu tốt hơn mô hình phức tạp.
- **Độ đo sai số ngoài mẫu trên tập Test**:
  $$\text{MAE} = \frac{1}{H}\sum_{h=1}^H |y_{t+h} - \hat{y}_{t+h}|$$
  $$\text{RMSE} = \sqrt{\frac{1}{H}\sum_{h=1}^H (y_{t+h} - \hat{y}_{t+h})^2}$$
  $$\text{MAPE} = \frac{1}{H}\sum_{h=1}^H \left|\frac{y_{t+h} - \hat{y}_{t+h}}{y_{t+h}}\right| \times 100\%$$
  - **MAE**: Sai số tuyệt đối trung bình, trực quan, có cùng đơn vị với chuỗi gốc.
  - **RMSE**: Nhạy cảm nhất với các điểm dị biệt (Outliers) do sai số bị bình phương trước khi lấy căn; phạt rất nặng các sai số dự báo lệch lớn tại đỉnh mùa vụ.
  - **MAPE**: Đo lường phần trăm sai số tương đối, rất trực quan cho các nhà quản lý doanh nghiệp. Chỉ hợp lệ khi toàn bộ các điểm thực tế đều khác 0 và cách xa 0.

---

## 3. CẨM NANG ĐỌC HIỂU & DIỄN GIẢI TOÀN BỘ 9 BIỂU ĐỒ THỰC NGHIỆM

Toàn bộ 9 biểu đồ chất lượng cao được lưu trữ tại thư mục [lab2/plots/](file:///home/thanhlong/Documents/TimeSeries/lab2/plots).

### [Biểu đồ 1] `ds1_01_eda_raw_series.png` – Chuỗi gốc Monthly Drug Sales (a10)
- **Mục đích**: Nhận diện hình thái tổng thể của chuỗi doanh số dược phẩm trị tiểu đường Úc (07/1991 – 06/2008).
- **Cách quan sát**:
  - **Xu hướng (Trend)**: Đường chuỗi màu xanh dương dốc lên rất mạnh và đều đặn theo thời gian từ mức $3.53$ triệu USD (07/1991) lên mức đỉnh kỷ lục $29.67$ triệu USD (01/2008).
  - **Mùa vụ (Seasonality)**: Các sóng dao động hình răng cưa lặp lại chính xác mỗi 12 tháng. Đỉnh cao nhất rơi vào tháng 12 – tháng 1, đáy sâu rơi vào tháng 2.
  - **Các mốc thống kê định lượng**:
    - Đường nét đứt màu đỏ: $\text{Mean} = 10.69$ triệu USD.
    - Đường chấm xanh lá: $\text{Max} = 29.67$ triệu USD (tháng 01/2008).
    - Đường chấm tím: $\text{Min} = 3.53$ triệu USD (tháng 07/1991).
- **Diễn giải học thuật**: Chuỗi vi phạm hoàn toàn tính dừng yếu vì kỳ vọng $\mathbb{E}[y_t]$ tăng theo thời gian và phương sai mở rộng dần.

---

### [Biểu đồ 2] `ds1_02_differenced_series.png` – Chuỗi sau sai phân bậc 1 ($d=1$)
- **Mục đích**: Kiểm tra tác động của phép lấy sai phân thường bậc 1: $y'_t = y_t - y_{t-1}$.
- **Cách quan sát**:
  - Trục trung bình đã bị kéo về dao động quanh đường thẳng nằm ngang $0$ (giá trị trung bình thực tế sau sai phân $\approx 0.11$ triệu USD).
  - Xu hướng tăng trưởng dài hạn đã biến mất hoàn toàn.
  - Tuy nhiên, hình thái sóng dao động mùa vụ hình sin vẫn còn nguyên vẹn, các đợt tăng vọt tháng 1 và sụt giảm tháng 2 vẫn lặp lại đều đặn mỗi năm.
- **Diễn giải học thuật**: Sai phân bậc 1 thành công đưa chuỗi về trạng thái dừng về trung bình (Mean-Stationary), loại bỏ nghiệm đơn vị xu hướng.

---

### [Biểu đồ 3] `ds1_03_acf_pacf.png` – Biểu đồ ACF & PACF xác định $(p, d, q)$ & Giải mã 3 điểm dị biệt
- **Mục đích**: Nhận diện bậc $(p, d, q)$ cho mô hình ARIMA theo Box-Jenkins.
- **Cách quan sát**:
  - Dải bóng màu xanh nhạt đại diện cho khoảng tin cậy $95\%$ ($\pm 0.1376$). Bất kỳ thanh nào nằm lọt bên trong dải xanh được coi là bằng 0 về mặt thống kê.
- **🔍 GIẢI MÃ 3 NHÓM ĐIỂM DỊ BIỆT TRỌNG YẾU**:
  1. **Dị biệt 1 (Lag 1 âm sâu trên cả ACF và PACF: $-0.3651$)**:
     - *Căn nguyên*: Tự tương quan âm nhân tạo do sai phân (Differencing-induced Negative Autocorrelation). Khi lấy $y_t - y_{t-1}$, hai số hạng liên tiếp dùng chung nhiễu $\epsilon_{t-1}$ nhưng trái dấu, tạo hiệp phương sai âm $-\sigma^2$ ($\rho_1 \approx -0.5$).
     - *Ứng dụng*: ACF cắt cụt ngay sau lag 1 $\implies$ Chọn bậc trung bình trượt $q = 1$ để bù trừ tương quan âm này.
  2. **Dị biệt 2 (Các gai dương khổng lồ tại Lag 12, 24, 36 trên ACF: $+0.7569, +0.6152, +0.5395$)**:
     - *Căn nguyên*: Do sai phân bậc 1 chỉ khử xu hướng chứ không khử mùa vụ. Doanh số tháng này năm nay tương quan thuận cực mạnh với cùng kỳ năm trước.
     - *Ứng dụng*: Minh chứng khẳng định chuỗi còn tính mùa vụ mạnh mẽ.
  3. **Dị biệt 3 (Gai âm kỷ lục tại Lag 11 trên PACF: $-0.6340$ và các gai âm tại Lag 11, 13 trên ACF)**:
     - *Căn nguyên (Đối nghịch Đỉnh – Đáy)*: Trong chuỗi `a10`, **tháng 1 là đỉnh cao nhất năm** và **tháng 2 liền kề là đáy sâu nhất năm**. Độ lệch 11 tháng và 13 tháng đối chiếu một Đỉnh của năm nay với một Đáy của năm ngoái, tạo ra tương quan âm rất lớn. Trên PACF, sau khi loại bỏ ảnh hưởng trung gian, mối đối nghịch đỉnh-đáy này bộc lộ mạnh nhất tại lag 11 với giá trị kỷ lục $-0.6340$.
- **Đề xuất mô hình**: $\text{ARIMA}(0, 1, 1)$ (ưu tiên theo Parsimony) và $\text{ARIMA}(1, 1, 1)$ (hỗn hợp dự phòng).

---

### [Biểu đồ 4] `ds1_04_residual_diagnostics.png` – Chẩn đoán 4 tiêu chuẩn phần dư ARIMA
- **Mục đích**: Kiểm tra xem phần dư của $\text{ARIMA}(0, 1, 1)$ có đạt chuẩn Nhiễu trắng (White Noise) hay không.
- **Cách quan sát 4 góc đồ thị**:
  1. *Góc trên trái (Residuals theo thời gian)*: Dao động ngẫu nhiên quanh mức 0, biên độ tương đối ổn định.
  2. *Góc trên phải (Phân phối phần dư & KDE)*: Đường cong ước lượng mật độ KDE có dạng hình chuông tương đối đối xứng quanh 0.
  3. *Góc dưới trái (Normal Q-Q Plot)*: Các điểm phần dư thực nghiệm bám sát đường thẳng 45 độ, chỉ chệch nhẹ ở 2 đuôi.
  4. *Góc dưới phải (ACF của phần dư)*: Các lag ngắn 1 đến 10 nằm trọn trong dải tin cậy 95%, nhưng tại lag 12 vẫn còn một gai mùa vụ vượt biên.
- **Diễn giải học thuật**: Phần dư không có tự tương quan ngắn hạn (Ljung-Box lag 10 có $p=0.385$), nhưng chưa đạt White Noise thuần túy tuyệt đối do ARIMA phi mùa vụ không thể hấp thu chu kỳ 12 tháng.

---

### [Biểu đồ 5] `ds1_05_forecast_arima.png` – Đồ thị dự báo ngoài mẫu Train/Test ARIMA
- **Mục đích**: Đánh giá khả năng dự báo ngoài mẫu trên 41 tháng tập Test (02/2005 – 06/2008).
- **Cách quan sát**:
  - Đường màu đen: Dữ liệu Test thực tế dao động hình sin dữ dội từ $12.16$ đến $29.67$ triệu USD.
  - Đường màu đỏ và màu cam: Dự báo của $\text{ARIMA}(0, 1, 1)$ và $\text{ARIMA}(1, 1, 1)$ nhanh chóng hội tụ về một **đường thẳng nằm ngang tiệm cận mức $17.5 - 19.2$ triệu USD**.
- **Diễn giải học thuật**: Mô hình ARIMA phi mùa vụ chỉ nắm bắt được mức trung bình dài hạn, hoàn toàn bất lực trước quy luật mùa vụ. Sai lệch lớn nhất xảy ra tại **tháng 01/2008** (đỉnh thực tế $29.67$ vs dự báo $\approx 19.17 \implies$ lệch hơn $10.5$ triệu USD). $\text{ARIMA}(0, 1, 1)$ ít tham số hơn cho sai số $\text{RMSE} = 4.15$ tốt hơn $\text{ARIMA}(1, 1, 1)$ ($\text{RMSE} = 4.62$).

---

### [Biểu đồ 6] `ds2_01_eda_seasonality.png` – Chuỗi AirPassengers nhận diện mùa vụ & Heteroskedasticity
- **Mục đích**: Nhận diện đặc trưng chuỗi số lượng hành khách hàng không quốc tế (1949 – 1960).
- **Cách quan sát**:
  - Chuỗi có xu hướng tăng trưởng tuyến tính rõ rệt qua 12 năm (từ 104k lên 622k hành khách).
  - Chu kỳ mùa vụ lặp lại đều đặn mỗi $s=12$ tháng: Đỉnh hè bùng nổ vào tháng 7 và tháng 8, đáy đông rơi vào tháng 11.
  - **Hiện tượng phương sai thay đổi (Heteroskedasticity)**: Biên độ sóng mùa hè mở rộng theo cấp số nhân từ $\sim 20$k khách (năm 1949) lên hơn $\sim 150$k khách (năm 1960).
- **Diễn giải học thuật**: Khẳng định bản chất **Mô hình nhân tính ($Y_t = T_t \times S_t \times I_t$)**. Bắt buộc phải áp dụng biến đổi $\ln(Y_t)$ để nén ổn định phương sai trước khi lấy sai phân.

---

### [Biểu đồ 7] `ds2_02_seasonal_diff.png` – So sánh sai phân thường vs Sai phân kép
- **Mục đích**: Kiểm chứng tại sao sai phân thường $d=1$ là chưa đủ và tại sao bắt buộc phải dùng sai phân kép ($d=1, D=1, s=12$).
- **Cách quan sát 2 tầng biểu đồ**:
  - *Tầng trên (Sai phân thường $d=1$)*: Đã khử xu hướng, nhưng dạng sóng sin mùa vụ 12 tháng vẫn còn nguyên vẹn. Kiểm định ADF cho $p\text{-value} = 0.0711 > 0.05 \implies$ **Chưa dừng ở mức 5%**.
  - *Tầng dưới (Sai phân kép $d=1, D=1, s=12$)*: Dao động hoàn toàn ngẫu nhiên quanh mức 0, sóng mùa vụ bị triệt tiêu phẳng lì. Kiểm định ADF cho $\text{ADF} = -4.4433, p\text{-value} = 0.00025 \ll 0.01 \implies$ **Dừng tuyệt đối**.
- **Diễn giải học thuật**: Sai phân thường chỉ khử được nghiệm đơn vị tại tần số 0, chỉ có sai phân mùa vụ mới khử được nghiệm đơn vị tại tần số $2\pi/12$.

---

### [Biểu đồ 8] `ds2_03_acf_pacf_sarima.png` – Biểu đồ ACF/PACF xác định $(p, q, P, Q)$
- **Mục đích**: Nhận diện các tham số của mô hình $\text{SARIMA}(p, d, q)(P, D, Q)_{12}$ trên chuỗi sai phân kép.
- **Cách quan sát**:
  - *Ở các trễ ngắn ($1 \le lag \le 3$)*: ACF có một gai âm đơn độc tại lag 1 ($r_1 \approx -0.34$, vượt ra ngoài dải xanh) và cắt cụt dứt khoát tại lag 2, 3 $\implies q = 1$. PACF có gai âm tại lag 1 và giảm dần $\implies p \in \{0, 1\}$.
  - *Ở các trễ mùa vụ ($lag = 12, 24, 36$)*: ACF tại lag 12 có một gai âm rất nổi bật ($r_{12} \approx -0.38$) và cắt cụt hoàn toàn tại lag 24, 36 $\implies Q = 1$. PACF có phản ứng tại lag 12 và giảm dần $\implies P \in \{0, 1\}$.
- **Diễn giải học thuật**: Xác định cấu trúc tối ưu là Box-Jenkins Airline Model $\text{SARIMA}(0, 1, 1)(0, 1, 1)_{12}$ và mô hình tổng quát $\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$.

---

### [Biểu đồ 9] `ds2_04_sarima_vs_arima.png` – Đối chuẩn hiệu năng SARIMA vs ARIMA thuần
- **Mục đích**: So sánh trực tiếp năng lực dự báo ngoài mẫu trên 29 tháng tập Test (08/1958 – 12/1960).
- **Cách quan sát**:
  - Đường màu xanh lá nét đứt (SARIMA): Tái hiện chính xác từng đỉnh cao điểm mùa hè tháng 7-8 và từng đáy mùa đông tháng 11, dải bóng mờ tin cậy 95% bao trọn mọi quan sát thực tế màu đen.
  - Đường màu đỏ chấm chấm (ARIMA thuần): Nhanh chóng suy biến thành một đường dốc nhẹ tiệm cận, hoàn toàn phẳng lì, bỏ lỡ toàn bộ các đợt cao điểm du lịch hè (lệch gần 150 nghìn khách tại các đỉnh).
- **Diễn giải học thuật**:
  - $\text{ARIMA}(1, 1, 1)$ thuần: $\text{MAE} = 84.25$, $\text{RMSE} = 96.33$, $\text{MAPE} = 21.01\%$.
  - $\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$: $\text{MAE} = 12.46$, $\text{RMSE} = 15.25$, $\text{MAPE} = 2.82\%$.
  - **SARIMA giảm hơn 84.2% sai số RMSE**, khẳng định việc tích hợp toán tử mùa vụ $(1 - B^{12})$ là bắt buộc đối với dữ liệu chu kỳ.

---

## 4. TỔNG HỢP KẾT QUẢ & GIẢI ĐÁP 16 CÂU HỎI CỐT LÕI

### Giải đáp 9 câu hỏi trọng tâm Bài 5 (Dataset 1: a10):
1. **Vì sao không được shuffle? Test set dùng để làm gì?**  
   Chuỗi thời gian có cấu trúc phụ thuộc nhân quả thời gian ($y_t$ phụ thuộc $y_{t-1}$). Shuffle sẽ phá vỡ liên kết này và gây **Data Leakage (Rò rỉ dữ liệu)** khi lấy tương lai học ngược quá khứ. Test set là dữ liệu tương lai chưa thấy dùng để kiểm thử khách quan năng lực dự báo ngoài mẫu.
2. **Ý nghĩa $p, d, q$ đã chọn ($1, 1, 1$ hoặc $0, 1, 1$)?**  
   $d=1$ khử xu hướng tăng; $p=1$ biểu thị quán tính tăng trưởng phụ thuộc 1 tháng liền trước; $q=1$ cho phép hệ thống tự điều chỉnh theo cú sốc ngẫu nhiên của 1 tháng trước đó.
3. **Mô hình có hội tụ không?**  
   Có hội tụ (`converged: True`), thuật toán MLE tối ưu hóa thành công hàm khả năng hợp lý cực đại mà không gặp cảnh báo số học.
4. **Residuals có phải White Noise không?**  
   Kỳ vọng trung bình $\approx 0$ ($0.038$), không có tự tương quan ngắn hạn (Ljung-Box lag 10 có $p=0.385$), nhưng chưa phải White Noise hoàn hảo vì còn sót lại tương quan tại trễ mùa vụ 12 tháng do giới hạn của mô hình phi mùa vụ.
5. **Nếu residuals còn tự tương quan $\rightarrow$ Điều chỉnh gì?**  
   Chuyển sang mô hình mùa vụ **$\text{SARIMA}(p, d, q)(P, D, Q)_{12}$** với $D=1, s=12$ để triệt tiêu dứt điểm tương quan chu kỳ.
6. **Dự đoán có bám sát xu hướng không?**  
   Bám sát xu hướng trung bình dài hạn quanh mức $17.5 - 19.2$ triệu USD, nhưng là đường thẳng đơn điệu, không mô phỏng được các sóng mùa vụ.
7. **Sai lệch lớn nhất xảy ra ở đâu?**  
   Tại **tháng 01/2008** (đỉnh thực tế $29.67$ triệu USD vs dự báo $\approx 19.17$ triệu USD $\implies$ Max Error $> 10.5$ triệu USD) do khoảng cách giữa đỉnh nhọn thực tế và đường dự báo phẳng.
8. **Chỉ số nào nhạy với outlier nhất?**  
   **RMSE** là chỉ số nhạy nhất do phép toán bình phương sai số phạt rất nặng các độ lệch lớn tại các đỉnh mùa vụ.
9. **RMSE lớn hay nhỏ nói lên điều gì?**  
   RMSE đo độ lệch chuẩn của sai số dự báo; RMSE càng nhỏ thì khoảng cách giữa dự báo và thực tế càng hẹp, độ tin cậy càng cao.

---

### Giải đáp 7 bước nâng cao (Dataset 2: AirPassengers):
1. **Nhận diện mùa vụ**: Chu kỳ $s=12$, đỉnh hè tháng 7-8, đáy đông tháng 11. Biên độ mở rộng theo thời gian chứng minh mô hình nhân tính $\implies$ Cần biến đổi Logarit $\ln(Y_t)$ để ổn định phương sai.
2. **ADF chuỗi gốc**: $\text{ADF} = +0.8154, p\text{-value} = 0.9919 \implies$ Chuỗi gốc phi dừng mạnh mẽ.
3. **Sai phân thường + Mùa vụ**: Sai phân thường $d=1$ chỉ khử trend ($p=0.0711 > 0.05$ chưa dừng); kết hợp thêm sai phân mùa vụ $D=1, s=12$ đưa $\text{ADF} = -4.4433, p\text{-value} = 0.00025 \ll 0.01 \implies$ Chuỗi dừng tuyệt đối.
4. **ACF/PACF xác định tham số**: Trễ ngắn cắt cụt sau lag 1 trên ACF $\implies q=1, p \in \{0, 1\}$; trễ mùa vụ cắt cụt sau lag 12 trên ACF $\implies Q=1, P \in \{0, 1\}$.
5. **Fit SARIMA**: Mô hình hội tụ hoàn hảo, AIC tối ưu ($\approx -368.08$), phần dư đạt chuẩn White Noise (Ljung-Box lag 12 có $p=0.548$, lag 24 có $p=0.695 > 0.05$).
6. **Forecast & Đánh giá**: $\text{MAE} = 12.46$ nghìn khách, $\text{RMSE} = 15.25$ nghìn khách, $\text{MAPE} = 2.82\%$.
7. **So sánh với ARIMA**: ARIMA thuần có sai số khổng lồ ($\text{RMSE} = 96.33$, $\text{MAPE} = 21.01\%$) vì suy biến thành đường tiệm cận phẳng và hoàn toàn mù tịt trước mùa du lịch hè; SARIMA giảm hơn **$84.2\%$ sai số RMSE** nhờ toán tử mùa vụ $(1 - B^{12})$.

---

## 5. HƯỚNG DẪN KHỞI CHẠY NOTEBOOK

Mở notebook bằng Jupyter Lab / Jupyter Notebook hoặc trực tiếp trên VSCode / Antigravity IDE:
```bash
# Kích hoạt môi trường ảo
source /home/thanhlong/Documents/TimeSeries/.venv/bin/activate

# Mở Jupyter Lab
jupyter lab lab2/notebooks/lab2.ipynb
```

*Ghi chú*: Cả hai file notebook [lab2.ipynb](file:///home/thanhlong/Documents/TimeSeries/lab2/notebooks/lab2.ipynb) và [thuc_hanh_kiem_dinh_adf_arima_sarima.ipynb](file:///home/thanhlong/Documents/TimeSeries/lab2/notebooks/thuc_hanh_kiem_dinh_adf_arima_sarima.ipynb) đã được thực thi kiểm thử 100% không có lỗi, toàn bộ biểu đồ, số liệu thống kê và bảng đánh giá đã được render sẵn nội dòng.
