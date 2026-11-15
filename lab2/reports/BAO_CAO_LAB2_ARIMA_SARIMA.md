# BÁO CÁO THỰC HÀNH LAB 2: KIỂM ĐỊNH ADF & MÔ HÌNH HÓA ARIMA / SARIMA

**Khóa học:** Phân tích Chuỗi thời gian (Time Series Analysis)  
**Tập dữ liệu sử dụng:** 
1. `Dataset 1 (a10.csv)`: Doanh số bán lẻ dược phẩm trị tiểu đường hàng tháng tại Úc (Monthly Anti-diabetic Drug Sales).
   - Dữ liệu gốc: `lab2/data/raw/a10.csv`
   - Dữ liệu sau xử lý: `lab2/data/processed/a10_processed.csv`
2. `Dataset 2 (AirPassengers.csv)`: Số lượng hành khách hàng không quốc tế hàng tháng (1949 – 1960).
   - Dữ liệu gốc: `lab2/data/raw/AirPassengers.csv`
   - Dữ liệu sau xử lý: `lab2/data/processed/AirPassengers_processed.csv`

---

## MỤC LỤC
1. [CƠ SỞ LÝ THUYẾT & NỀN TẢNG TOÁN HỌC](#1-cơ-sở-lý-thuyết--nền-tảng-toán-học)
   - 1.1. Bản chất của tính dừng (Stationarity) và lý do bắt buộc chuỗi phải dừng
   - 1.2. Bản chất toán học của kiểm định ADF (Augmented Dickey-Fuller)
   - 1.3. Kỹ thuật sai phân thường ($d$) và sai phân mùa vụ ($D$)
   - 1.4. Phân tích hàm ACF và PACF theo phương pháp Box-Jenkins
   - 1.5. Cấu trúc mô hình ARIMA$(p, d, q)$ và SARIMA$(p, d, q)(P, D, Q)_s$
   - 1.6. Chẩn đoán phần dư (Residual Diagnostics) và tính chất White Noise
   - 1.7. Các tiêu chuẩn so sánh và chỉ số đánh giá sai số (AIC, BIC, MAE, RMSE, MAPE)
2. [QUY TRÌNH EDA DỮ LIỆU THÔ (RAW DATA) & TIỀN XỬ LÝ](#2-quy-trình-eda-dữ-liệu-thô-raw-data--tiền-xử-lý)
   - Khám phá hiện trạng dữ liệu thô (Kiểu dữ liệu, Missing, Duplicates)
   - Lý do và các bước tiền xử lý bắt buộc (DatetimeIndex, Tần suất 'MS', Log-transform)
3. [KẾT QUẢ THỰC HÀNH DATASET 1 (a10.csv) - BÀI 1 ĐẾN BÀI 5](#3-kết-quả-thực-hành-dataset-1-a10csv---bài-1-đến-bài-5)
   - Bài 1: Khám phá dữ liệu (EDA) & Trả lời câu hỏi
   - Bài 2: Kiểm định ADF chuỗi gốc & Trả lời câu hỏi
   - Bài 3: Sai phân bậc 1 ($d=1$) & Kiểm định lại
   - Bài 4: Phân tích ACF & PACF xác định $(p, d, q)$
   - Bài 5: Thiết lập ARIMA, Huấn luyện, Dự đoán & Trả lời trọn vẹn 9 câu hỏi
4. [KẾT QUẢ BÀI TẬP NÂNG CAO SARIMA VỚI DATASET 2 (AirPassengers.csv)](#4-kết-quả-bài-tập-nâng-cao-sarima-với-dataset-2-airpassengerscsv)
   - Nhận diện mùa vụ & Kiểm định ADF chuỗi gốc
   - Sai phân kép ($d=1, D=1, s=12$) kết hợp biến đổi Logarit
   - Nhận diện tham số $(p, d, q) \times (P, D, Q)_{12}$
   - Huấn luyện mô hình SARIMA và so sánh đối chuẩn hiệu năng với ARIMA thuần
5. [TỔNG KẾT & BÀI HỌC KINH NGHIỆM](#5-tổng-kết--bài-học-kinh-nghiệm)

---

# 1. CƠ SỞ LÝ THUYẾT & NỀN TẢNG TOÁN HỌC

### 1.1. Bản chất của tính dừng (Stationarity)
- **Định nghĩa tính dừng yếu (Weak / Covariance Stationarity):**  
  Chuỗi thời gian $\{y_t\}$ là dừng yếu nếu:
  1. $\mathbb{E}[y_t] = \mu$ (Kỳ vọng không đổi theo thời gian).
  2. $\text{Var}(y_t) = \sigma^2 < \infty$ (Phương sai hữu hạn và không đổi).
  3. $\text{Cov}(y_t, y_{t-k}) = \gamma_k$ (Hiệp phương sai chỉ phụ thuộc vào độ trễ $k$, không phụ thuộc vào mốc $t$).

- **Vì sao bắt buộc chuỗi phải dừng trước khi mô hình hóa?**
  1. **Tính khái quát hóa của tham số:** Các mô hình thống kê học các trọng số từ quá khứ để áp dụng cho tương lai. Nếu giá trị trung bình hoặc phương sai trôi dạt vô định (Non-stationary), các tham số ước lượng ở quá khứ sẽ trở nên vô nghĩa trong tương lai.
  2. **Tránh hồi quy giả mạo (Spurious Regression):** Khi hai biến không dừng cùng có xu hướng tăng theo thời gian, phép hồi quy giữa chúng sẽ cho ra hệ số xác định $R^2$ rất cao và $t$-statistic có ý nghĩa giả tạo, dù hai chuỗi hoàn toàn độc lập về mặt bản chất.
  3. **Đảm bảo tính ổn định và khả nghịch:** Phương trình tự hồi quy AR chỉ có nghiệm hội tụ khi các nghiệm của đa thức đặc trưng nằm ngoài vòng tròn đơn vị (Unit Circle).

---

### 1.2. Bản chất toán học của kiểm định ADF (Augmented Dickey-Fuller Test)
Mô hình hồi quy của kiểm định ADF có dạng tổng quát:
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{k} \delta_i \Delta y_{t-i} + \epsilon_t$$
- **Hệ số $\gamma$:** Bằng $\rho - 1$, trong đó $\rho$ là hệ số của quá trình $AR(1)$: $y_t = \rho y_{t-1} + \epsilon_t$.
  - Nếu $\rho = 1 \implies \gamma = 0$: Chuỗi có nghiệm đơn vị (Random walk - không dừng).
  - Nếu $|\rho| < 1 \implies \gamma < 0$: Chuỗi quay về mức cân bằng (Stationary).
- **Cặp giả thuyết kiểm định:**
  - $H_0: \gamma = 0$ (Chuỗi chứa nghiệm đơn vị $\rightarrow$ **KHÔNG DỪNG**).
  - $H_1: \gamma < 0$ (Chuỗi không có nghiệm đơn vị $\rightarrow$ **DỪNG**).
- **Quy tắc quyết định:**
  - So sánh $ADF\ Statistic$ với giá trị tới hạn ($Critical\ Values$) của phân phối Dickey-Fuller (phân phối lệch trái đặc thù, không phải Student-$t$).
  - Nếu $ADF\ Statistic < \text{Critical Value (5\%)}$ và $p\text{-value} < 0.05 \implies$ Bác bỏ $H_0$, kết luận chuỗi đã dừng.

---

### 1.3. Kỹ thuật Sai phân (Differencing)
1. **Sai phân thường bậc 1 ($d=1$):**
   $$\Delta y_t = y_t - y_{t-1} = (1 - B)y_t$$
   Triệt tiêu xu hướng tuyến tính (Linear Trend) trong chuỗi.
2. **Sai phân mùa vụ ($D=1$ với chu kỳ $s$):**
   $$\Delta_s y_t = y_t - y_{t-s} = (1 - B^s)y_t$$
   Triệt tiêu dao động lặp lại theo mùa.
3. **Sai phân kép ($d=1, D=1$):**
   $$w_t = (1 - B)(1 - B^s)y_t = (y_t - y_{t-1}) - (y_{t-s} - y_{t-s-1})$$
   Đồng thời loại bỏ cả xu hướng và tính chu kỳ mùa vụ.

---

### 1.4. Nhận diện bậc qua ACF và PACF theo Box-Jenkins
- **Hàm ACF (Autocorrelation Function):** Đo lường tương quan tổng thể giữa $y_t$ và $y_{t-k}$.
- **Hàm PACF (Partial Autocorrelation Function):** Đo lường tương quan thuần túy giữa $y_t$ và $y_{t-k}$ sau khi đã trừ đi ảnh hưởng tuyến tính của tất cả các biến trung gian $y_{t-1}, \dots, y_{t-k+1}$.
- **Quy tắc vàng:**
  - Nếu PACF **cắt cụt** sau lag $p$ và ACF **giảm dần** $\implies$ Mô hình $\text{AR}(p)$.
  - Nếu ACF **cắt cụt** sau lag $q$ và PACF **giảm dần** $\implies$ Mô hình $\text{MA}(q)$.
  - Nếu cả ACF và PACF cùng suy giảm dần $\implies$ Mô hình $\text{ARMA}(p, q)$.
  - Đối với thành phần mùa vụ: Quan sát các gai tại các trễ bội số của chu kỳ mùa vụ $s$ ($s, 2s, 3s,...$).

---

# 2. QUY TRÌNH EDA DỮ LIỆU THÔ (RAW DATA) & TIỀN XỬ LÝ

Trước khi đưa vào mô hình hóa, dữ liệu thô trong `lab2/data/raw/` được khám phá toàn diện để phát hiện các rào cản kỹ thuật:

### 2.1. Kết quả kiểm tra chất lượng dữ liệu thô (EDA Raw Data):
- **Missing values**: Cả 2 dataset đều có $0$ giá trị khuyết thiếu.
- **Duplicates**: $0$ dòng trùng lặp.
- **Định dạng cột thời gian**: Cột `date` (a10) và `Month` (AirPassengers) đều đang ở dạng **chuỗi ký tự văn bản (`object / string`)**, chưa phải là trục thời gian thực sự.
- **Tần suất (Frequency)**: Chưa được gán tần suất chuỗi, khiến thư viện thống kê không tự nhận diện được độ trễ chu kỳ.

### 2.2. Các bước Tiền xử lý bắt buộc:
1. **Ép kiểu thời gian**: Chuyển đổi cột ngày tháng sang kiểu `pd.DatetimeIndex` và gán làm Index chính.
2. **Khai báo tần suất**: Gán `freq='MS'` (*Month Start*) để cố định bước nhảy đều đặn 1 tháng.
3. **Biến đổi Logarit $\ln(y_t)$ (với AirPassengers)**: Ép phẳng biên độ dao động mùa vụ ngày càng phình to, đưa mô hình nhân tính về mô hình cộng tính nhằm thỏa mãn giả định phương sai không đổi (*Homoscedasticity*).
4. **Tính toán sai phân ($d=1, D=1$)** và xuất dữ liệu sạch sang `lab2/data/processed/`.

---

# 3. KẾT QUẢ THỰC HÀNH DATASET 1 (a10.csv) - BÀI 1 ĐẾN BÀI 5

### Bài 1: Khám phá Dữ liệu (EDA)
- **Tập dữ liệu:** 204 quan sát từ tháng 07/1991 đến tháng 06/2008.
- **Trực quan hóa:**
  - Đồ thị cho thấy chuỗi có **xu hướng tăng trưởng rất mạnh (Upward Trend)**: Doanh số tăng từ 3.5 triệu USD lên tới gần 30 triệu USD.
  - **Tính mùa vụ hàng năm rõ rệt (Seasonality với $s=12$)**: Hàng năm doanh số tăng vọt đạt đỉnh cao nhất vào tháng 12 và tháng 1, sau đó giảm sâu chạm đáy vào tháng 2.
  - **Kết luận:** Chuỗi vi phạm cả điều kiện kỳ vọng không đổi lẫn phương sai không đổi $\rightarrow$ **Chuỗi KHÔNG DỪNG**.

---

### Bài 2: Kiểm định ADF trên chuỗi gốc
- **Kết quả tính toán:**
  - $ADF\ Statistic = 3.1452$
  - $p\text{-value} = 1.0000$
  - Ngưỡng tới hạn (Critical Values): $1\%: -3.4656; \quad 5\%: -2.8770; \quad 10\%: -2.5750$
- **Trả lời câu hỏi:**
  - $H_0$: Chuỗi chứa nghiệm đơn vị (Unit Root), tức chuỗi **không dừng**.
  - $p\text{-value} = 1.0 > 0.05$: Không đủ cơ sở bác bỏ $H_0$.
  - **Kết luận:** Chuỗi gốc là chuỗi **KHÔNG DỪNG**.

---

### Bài 3: Sai phân bậc 1 & Kiểm định lại
- Áp dụng sai phân bậc 1: $\Delta y_t = y_t - y_{t-1}$.
- **Kết quả tính toán ADF sau sai phân bậc 1:**
  - $ADF\ Statistic = -2.4952$
  - $p\text{-value} = 0.1166$
  - Ngưỡng tới hạn 10%: $-2.5751$.
- **Trả lời câu hỏi:**
  - **Sau sai phân chuỗi đã dừng chưa?** Xu hướng tuyến tính đã bị triệt tiêu hoàn toàn, trung bình đã dao động quanh 0. $p\text{-value}$ ở mức $0.1166$ (gần đạt mức ý nghĩa 10%) do chuỗi vẫn còn thành phần mùa vụ 12 tháng mạnh mẽ.
  - **Giá trị $d$ trong ARIMA là bao nhiêu?** Chọn **$d = 1$**. Việc giữ $d=1$ là chuẩn mực để tránh hiện tượng quá sai phân (over-differencing).

---

### Bài 4: Phân tích ACF & PACF
- **Quan sát đồ thị:**
  - Đồ thị PACF: Có gai nhọn có ý nghĩa thống kê vượt dải tin cậy tại **lag 1** và **lag 2**, sau đó giảm dần. $\implies p \in \{1, 2\}$.
  - Đồ thị ACF: Cắt cụt rõ nét sau **lag 1** với hệ số tương quan âm, tiếp sau đó xuất hiện sóng chu kỳ mùa vụ tại các lag 12, 24, 36. $\implies q = 1$.
- **Đề xuất mô hình:**
  - Mô hình 1: $\text{ARIMA}(1, 1, 1)$
  - Mô hình 2: $\text{ARIMA}(2, 1, 1)$

---

### Bài 5: Thiết lập ARIMA – Huấn luyện – Chẩn đoán – Dự đoán – Đánh giá
- **Chia tập dữ liệu:** 
  - Train (80%): 163 tháng (1991-07 đến 2005-01).
  - Test (20%): 41 tháng (2005-02 đến 2008-06).
- **So sánh tiêu chuẩn thông tin:**
  - $\text{ARIMA}(1, 1, 1)$: $AIC = 582.47, \quad BIC = 591.73$
  - $\text{ARIMA}(2, 1, 1)$: $AIC = 584.46, \quad BIC = 596.81$
  - $\implies$ Chọn **$\text{ARIMA}(1, 1, 1)$** vì cả AIC và BIC đều thấp hơn.
- **Kết quả đánh giá trên tập Test:**
  - **MAE**: $3.6667$ triệu USD
  - **RMSE**: $4.6243$ triệu USD
  - **MAPE**: $18.04\%$

#### **Bảng trả lời chi tiết 9 câu hỏi của Bài 5:**
1. **Vì sao không được shuffle dữ liệu? Test set dùng để làm gì?**
   - Không được shuffle vì dữ liệu chuỗi thời gian có **tính phụ thuộc thời gian (Autocorrelation)**. Nếu tráo đổi, quy luật nhân quả thời gian bị phá hủy và gây rò rỉ dữ liệu (Data Leakage / Lookahead Bias). Test set đóng vai trò dữ liệu tương lai chưa từng thấy để kiểm định độc lập khả năng dự báo.
2. **Ý nghĩa của $p, d, q$ đã chọn ($1, 1, 1$)?**
   - $d=1$: Số lần sai phân để khử xu hướng.
   - $p=1$: Biến động kỳ này phụ thuộc vào biến động sai phân của 1 kỳ quá khứ gần nhất.
   - $q=1$: Dự báo được bù trừ thêm bởi sai số ngẫu nhiên của 1 kỳ trước đó.
3. **Mô hình có hội tụ không?**
   - **Có hội tụ**. Thuật toán tối ưu hóa ước lượng hợp lý cực đại (MLE) hội tụ hoàn toàn không có cảnh báo.
4. **Residuals có phải White Noise không?**
   - Chưa phải White Noise hoàn hảo. Kỳ vọng trung bình xấp xỉ 0 (0.0381) và không có tự tương quan ngắn hạn (Ljung-Box lag 10 có $p=0.385$), nhưng đồ thị ACF phần dư vẫn còn các đỉnh nhọn vượt ngưỡng tại lag 12 và 24 do mô hình ARIMA phi mùa vụ chưa xử lý được mùa vụ.
5. **Nếu residuals còn tự tương quan $\rightarrow$ điều chỉnh gì?**
   - Nâng cấp mô hình lên **$\text{SARIMA}(p, d, q)(P, D, Q)_{12}$** để đưa thêm các tham số mùa vụ vào triệt tiêu hoàn toàn tự tương quan còn sót lại.
6. **Dự đoán có bám sát xu hướng không?**
   - Có bám sát đường xu hướng tăng trung bình (Mean Trend), nhưng là đường phẳng tiệm cận, không thể tái tạo lại các đỉnh đáy mùa vụ.
7. **Sai lệch lớn nhất xảy ra ở đâu?**
   - Xảy ra tại các tháng **đỉnh cao điểm (tháng 12 - tháng 1)** và tháng **đáy thấp điểm (tháng 2)** của từng năm.
8. **Chỉ số nào nhạy với outlier nhất?**
   - **RMSE** là chỉ số nhạy nhất vì có phép toán bình phương sai số, khuếch đại cực đại các sai số lớn.
9. **RMSE lớn hay nhỏ nói lên điều gì?**
   - RMSE đo lường độ phân tán chuẩn của sai số dự báo. RMSE càng nhỏ chứng tỏ dự báo càng chính xác và độ tin cậy càng cao.

---

# 4. KẾT QUẢ BÀI TẬP NÂNG CAO SARIMA VỚI DATASET 2 (AirPassengers.csv)

### 3.1. Nhận diện đặc tính chuỗi gốc
- Chuỗi gồm 144 tháng (1949 – 1960).
- Biểu hiện xu hướng tăng trưởng theo hàm mũ cùng với biên độ mùa vụ $s=12$ mở rộng dần $\rightarrow$ **Mô hình mùa vụ nhân tính (Multiplicative Seasonality)**.
- Kiểm định ADF chuỗi gốc: $ADF = 0.8154, \quad p\text{-value} = 0.9919 > 0.05 \implies$ **KHÔNG DỪNG**.

### 3.2. Sai phân kép ($d=1, D=1, s=12$) trên Logarit
- Áp dụng biến đổi logarit: $z_t = \ln(y_t)$ để ổn định phương sai biến thiên.
- Áp dụng sai phân kép: $w_t = (1 - B)(1 - B^{12})z_t$.
- **Kiểm định ADF sau sai phân kép:**
  - $ADF\ Statistic = -4.4433$
  - $p\text{-value} = 0.000248 \ll 0.01$ (vượt ngưỡng tới hạn 1%: $-3.4870$).
  - $\implies$ **Chuỗi DỪNG HOÀN TOÀN**.

### 3.3. Nhận diện bậc và huấn luyện SARIMA$(1,1,1)(1,1,1)_{12}$
- Quan sát ACF và PACF tại lag 1 (non-seasonal) và lag 12 (seasonal): Đề xuất mô hình chuẩn $\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$.
- Chia tập Train (80% = 115 tháng) và Test (20% = 29 tháng: 1958-08 đến 1960-12).

### 3.4. Bảng So sánh Đối chuẩn Hiệu năng Dự báo (SARIMA vs ARIMA thuần)

| Tiêu chí đánh giá | ARIMA(1, 1, 1) thuần | SARIMA(1, 1, 1)(1, 1, 1)[12] | Mức độ cải thiện |
| :--- | :---: | :---: | :---: |
| **MAE (Khách hàng)** | $84.25$ | **$12.46$** | Giảm **$85.2\%$** |
| **RMSE (Khách hàng)** | $96.33$ | **$15.25$** | Giảm **$84.2\%$** (hơn 6 lần) |
| **MAPE (%)** | $21.01\%$ | **$2.82\%$** | Giảm từ $21\%$ xuống chỉ còn **$2.8\%$** |

**Nhận xét:**
- Biểu đồ dự báo cho thấy mô hình SARIMA bám sát từng đỉnh sóng tháng 7-8 và đáy sóng mùa đông trong 29 tháng của tập Test một cách xuất sắc.
- Mô hình ARIMA thuần không có thành phần mùa vụ chỉ vẽ ra một đường thẳng dự báo xu hướng trung bình, dẫn đến sai số rất lớn và không thể áp dụng vào thực tiễn kinh doanh.

---

# 5. TỔNG KẾT & BÀI HỌC KINH NGHIỆM

1. **Kiểm định ADF là công cụ bắt buộc** trước khi tiến hành mô hình hóa theo trường phái Box-Jenkins để định lượng mức độ dừng và lựa chọn số lần sai phân $d$.
2. **Sai phân quá mức (Over-differencing) nguy hiểm không kém việc chưa đủ sai phân**, do làm phức tạp hóa cấu trúc phần dư và làm tăng phương sai vô ích.
3. **Phần dư phải được kiểm tra tính White Noise**: Nếu phần dư vẫn còn tự tương quan (đặc biệt tại các bội số của chu kỳ mùa vụ), mô hình vẫn chưa khai thác hết thông tin trong dữ liệu và bắt buộc phải chuyển sang **SARIMA**.
4. **Quy trình Box-Jenkins tiêu chuẩn**:
   $$\text{Identification (ADF/ACF/PACF)} \longrightarrow \text{Estimation (MLE/AIC/BIC)} \longrightarrow \text{Diagnostic Checking (Residuals)} \longrightarrow \text{Forecasting}$$
