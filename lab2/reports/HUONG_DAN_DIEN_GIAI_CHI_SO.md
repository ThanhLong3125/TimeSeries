# HƯỚNG DẪN DIỄN GIẢI CÁC CHỈ SỐ & BIỂU ĐỒ TRONG PHÂN TÍCH CHUỖI THỜI GIAN (LAB 2)

Tài liệu này tóm lược quy tắc đọc nhanh và diễn giải ý nghĩa của toàn bộ các chỉ số thống kê, đồ thị trong quy trình xây dựng mô hình ARIMA / SARIMA.

---

## 1. Kiểm định Augmented Dickey-Fuller (ADF Test)

| Chỉ số / Đại lượng | Ý nghĩa | Cách diễn giải |
| :--- | :--- | :--- |
| **$H_0$ (Null Hypothesis)** | Chuỗi có nghiệm đơn vị (Unit root). | Mặc định coi chuỗi là **Không dừng (Non-stationary)**. |
| **$H_1$ (Alternative)** | Chuỗi không có nghiệm đơn vị. | Chuỗi **Dừng (Stationary)**. |
| **ADF Statistic** | Giá trị thống kê kiểm định Dickey-Fuller. | Giá trị càng âm (càng nhỏ hơn các Critical Values âm) thì bằng chứng bác bỏ $H_0$ càng mạnh. |
| **Critical Values (1%, 5%, 10%)** | Các ngưỡng tới hạn của phân phối DF. | Để kết luận chuỗi dừng ở mức 5%, bắt buộc **ADF Statistic < Critical Value 5%**. |
| **$p\text{-value}$** | Xác suất xảy ra sai lầm loại 1 khi bác bỏ $H_0$. | • $p < 0.05 \implies$ **Bác bỏ $H_0$, Chuỗi DỪNG**.<br>• $p \ge 0.05 \implies$ **Không thể bác bỏ $H_0$, Chuỗi KHÔNG DỪNG**. |

---

## 2. Quy tắc đọc đồ thị ACF và PACF để chọn bậc $(p, d, q)$

- **Dải tin cậy màu xanh (Confidence Band, thường là 95%)**: Bất kỳ thanh đứng nào nằm lọt bên trong dải tin cậy thì hệ số tương quan đó được coi là bằng 0 về mặt thống kê.
- **Cắt cụt (Cuts off / Sharp Drop)**: Đột ngột rớt vào trong dải tin cậy ngay sau một số bước trễ cụ thể.
- **Giảm dần (Tails off)**: Giảm dần đều theo hàm mũ hoặc dao động sóng hình sin tắt dần.

### Bảng tra cứu mô hình:
| Biểu hiện PACF | Biểu hiện ACF | Kết luận mô hình |
| :--- | :--- | :--- |
| **Cắt cụt sau lag $p$** | Giảm dần (Tails off) | Mô hình **$\text{AR}(p)$** |
| Giảm dần (Tails off) | **Cắt cụt sau lag $q$** | Mô hình **$\text{MA}(q)$** |
| Giảm dần | Giảm dần | Mô hình hỗn hợp **$\text{ARMA}(p, q)$** |
| Cắt cụt tại lag $s, 2s$ | Giảm dần tại các lag $s, 2s$ | Thành phần mùa vụ **$P$** |
| Giảm dần tại các lag $s, 2s$ | Cắt cụt tại lag $s, 2s$ | Thành phần mùa vụ **$Q$** |

---

## 3. Tiêu chuẩn lựa chọn mô hình: AIC và BIC

$$AIC = 2k - 2\ln(\hat{L})$$
$$BIC = k\ln(n) - 2\ln(\hat{L})$$
- $k$: Số lượng tham số của mô hình.
- $n$: Số lượng quan sát dữ liệu.
- $\hat{L}$: Giá trị cực đại của hàm hợp lý (Likelihood).
- **Quy tắc:**
  - Mô hình nào có **AIC và BIC càng nhỏ** thì mô hình đó càng tối ưu.
  - Cả hai chỉ số đều thưởng cho mô hình khớp dữ liệu tốt (tăng $\hat{L}$) và phạt mô hình quá phức tạp (tăng $k$). Trong đó, **BIC phạt nặng hơn đối với số lượng tham số** khi $n$ lớn.

---

## 4. Chẩn đoán phần dư (Residual Diagnostics)

Phần dư được định nghĩa là phần sai số thực tế trừ đi dự báo: $e_t = y_t - \hat{y}_t$.  
Một mô hình đạt chuẩn bắt buộc phần dư phải là **Nhiễu trắng (White Noise)**:
1. **Đồ thị thời gian của phần dư:** Phải dao động ngẫu nhiên xung quanh đường $0$, không có xu hướng tăng/giảm và biên độ dao động phải đồng đều qua thời gian (phương sai không đổi).
2. **Đồ thị phân phối (Histogram & KDE):** Phải có dạng hình chuông đối xứng (phân phối chuẩn $\mathcal{N}(0, \sigma^2)$).
3. **Normal Q-Q Plot:** Các điểm sai số phải bám sát đường thẳng 45 độ.
4. **Đồ thị ACF của phần dư:** Toàn bộ các thanh tương quan tại tất cả các độ trễ $k \ge 1$ phải nằm hoàn toàn bên trong dải tin cậy 95%.
5. **Kiểm định Ljung-Box:**
   - $H_0$: Các phần dư hoàn toàn độc lập, không tự tương quan.
   - Để đạt chuẩn White Noise, ta cần **$p\text{-value} > 0.05$** (không bác bỏ $H_0$).

---

## 5. Các chỉ số đo lường sai số dự báo (Forecast Accuracy)

| Chỉ số | Công thức toán | Đơn vị | Ưu điểm | Nhược điểm |
| :--- | :---: | :---: | :--- | :--- |
| **MAE** *(Mean Absolute Error)* | $\frac{1}{n}\sum_{t=1}^n \|y_t - \hat{y}_t\|$ | Cùng đơn vị với dữ liệu gốc | Trực quan, dễ hiểu, không phóng đại outlier | Không phạt nặng các sai số đột biến nguy hiểm |
| **RMSE** *(Root Mean Squared Error)* | $\sqrt{\frac{1}{n}\sum_{t=1}^n (y_t - \hat{y}_t)^2}$ | Cùng đơn vị với dữ liệu gốc | Phản ánh độ lệch chuẩn sai số; phạt rất nặng các sai số lớn | **Rất nhạy cảm với outlier** (chỉ cần 1 điểm lệch lớn sẽ kéo RMSE tăng vọt) |
| **MAPE** *(Mean Absolute Percentage Error)* | $\frac{100\%}{n}\sum_{t=1}^n \left\|\frac{y_t - \hat{y}_t}{y_t}\right\|$ | Tỷ lệ phần trăm ($\%$) | Không phụ thuộc đơn vị đo, so sánh được giữa các bài toán khác nhau | Không xác định nếu $y_t = 0$; bị méo mó nếu $y_t$ rất nhỏ gần 0 |

---

## 6. Sơ đồ tư duy kiểm tra tính hợp lệ của mô hình

```
[Mô hình chuỗi thời gian]
         │
         ├── 1. Khám phá (EDA)
         │       └── Có Trend? Có Seasonality?
         │
         ├── 2. Kiểm định tính dừng (ADF)
         │       ├── p >= 0.05 ──> Cần sai phân (d hoặc D)
         │       └── p < 0.05  ──> Đã dừng, sẵn sàng mô hình hóa
         │
         ├── 3. Định dạng bậc (ACF/PACF)
         │       └── Tìm (p, q) và (P, Q) qua vị trí cắt cụt
         │
         ├── 4. Huấn luyện & So sánh AIC/BIC
         │       └── Chọn mô hình có AIC/BIC thấp nhất
         │
         ├── 5. Chẩn đoán phần dư (Residuals)
         │       ├── Mean ≈ 0?
         │       ├── ACF không vượt dải tin cậy?
         │       └── Ljung-Box p-value > 0.05?
         │               ├── ĐÚNG ──> Mô hình tốt, dùng để dự báo (Forecasting)
         │               └── SAI  ──> Còn sót quy luật, thêm tham số hoặc chuyển sang SARIMA
         │
         └── 6. Dự báo & Đánh giá (Test set)
                 └── MAE, RMSE, MAPE
```
