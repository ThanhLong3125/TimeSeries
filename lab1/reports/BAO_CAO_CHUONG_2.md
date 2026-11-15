# BÁO CÁO BÀI TẬP THỰC HÀNH CHƯƠNG 2
## PHÂN TÍCH CHUỖI THỜI GIAN (TIME SERIES ANALYSIS)

---

### 1. Mô tả Dataset
* **Tập dữ liệu:** `AirPassengers` (Dữ liệu kinh điển quốc tế về chuỗi thời gian do Box & Jenkins giới thiệu).
* **Chu kỳ và phạm vi:** Thu thập định kỳ theo từng tháng (**Monthly**) từ tháng **01/1949 đến 12/1960** (tổng cộng **144 quan sát**, tròn 12 năm).
* **Ý nghĩa biến:** Biến duy nhất `Passengers` biểu thị số lượng hành khách hàng không quốc tế (đơn vị: nghìn người).
* **Cấu trúc:** Chuỗi đơn biến (Univariate Time Series), giá trị nguyên dương liên tục, không có khoảng cách thời gian bị ngắt quãng.

---

### 2. Kiểm tra ACF / PACF (40 Lags)
* **Kết quả biểu đồ:** (Xem biểu đồ tại `plots/bai1_02_acf_pacf_40lags.png`)
* **Phân tích:**
  * **Độ tự tương quan:** Chuỗi có **tự tương quan rất mạnh**. Các giá trị ACF từ lag 1 đến lag 20+ đều nằm vượt rất xa dải tin cậy 95% ($\pm 1.96/\sqrt{N}$).
  * **Xu hướng (Trend):** Đồ thị ACF suy giảm rất chậm theo thời gian (*slow decay*), đây là đặc trưng không thể nhầm lẫn của một chuỗi **không dừng (Non-stationary)** có xu hướng tăng trưởng rõ rệt.
  * **Mùa vụ (Seasonality):** Đường ACF dao động hình sóng lượn với các đỉnh nhô cao đột biến tại các chu kỳ bội số của 12 (lag 12, lag 24, lag 36), khẳng định chu kỳ mùa vụ hàng năm $S = 12$.
  * **Lag quan trọng nhất:**
    * **Lag 1:** Có hệ số tự tương quan cao nhất ($\approx 0.95$), trên đồ thị PACF lag 1 áp đảo hoàn toàn, phản ánh ảnh hưởng trực tiếp của giá trị tháng liền kề trước đó.
    * **Lag 12:** Thể hiện độ phụ thuộc vào cùng kỳ năm trước.

---

### 3. Kiểm tra White Noise (Nhiễu trắng)
* **Dữ liệu đối chứng:** Sinh 1000 điểm dữ liệu ngẫu nhiên phân phối chuẩn $\mathcal{N}(0, 1)$ bằng NumPy.
* **Kết quả so sánh:** (Xem biểu đồ tại `plots/bai2_compare_white_noise_vs_airpassengers.png`)
* **5 Điểm khác biệt mấu chốt:**
  1. **Tính độc lập:** White Noise có hệ số ACF tại mọi lag $\ge 1$ xấp xỉ bằng 0 và nằm trọn vẹn trong dải tin cậy 95%, chứng minh các quan sát hoàn toàn độc lập, không lưu giữ ký ức.
  2. **Tính ghi nhớ:** AirPassengers có ACF tại lag 1 lên đến 0.95 và duy trì mức tương quan cao qua hàng chục lag, chứng minh tính phụ thuộc mạnh mẽ vào quá khứ.
  3. **Tính chu kỳ:** AirPassengers có hình sin tuần hoàn cứ mỗi 12 tháng, trong khi White Noise hoàn toàn ngẫu nhiên và phẳng lặng.
  4. **Tốc độ suy giảm:** ACF của White Noise triệt tiêu ngay lập tức về 0 sau lag 0; ACF của AirPassengers suy giảm cực kỳ chậm do ảnh hưởng của Trend.
  5. **Giá trị dự báo:** White Noise thuần túy là sai số không thể dự báo; AirPassengers có cấu trúc toán học chặt chẽ, hoàn toàn có thể xây dựng mô hình dự báo tương lai.

---

### 4. Phân rã chuỗi thời gian (Decomposition)
* **Phương pháp áp dụng:** Classical Decomposition dạng nhân (**Multiplicative Decomposition**):
  $$Y_t = \text{Trend}_t \times \text{Seasonal}_t \times \text{Residual}_t$$
  *(Lựa chọn mô hình dạng nhân là tối ưu nhất vì biên độ dao động mùa vụ tăng dần theo thời gian cùng với đà tăng của Trend).*
* **Kết quả 4 thành phần:** (Xem biểu đồ tại `plots/bai3_01_decomposition_multiplicative.png`)
  * **Xu hướng (Trend):** Tăng trưởng dốc và ổn định từ mức ~120 nghìn khách (1949) lên ~450 nghìn khách (1960).
  * **Mùa vụ (Seasonal):** Rất mạnh với chu kỳ đều đặn 12 tháng:
    * **Đỉnh cao nhất:** Tháng 7 và Tháng 8 (vượt mức trung bình năm từ 20% - 30% do cao điểm du lịch hè).
    * **Đáy thấp nhất:** Tháng 11 (thấp hơn khoảng 20% so với trung bình năm).
  * **Residual (Phần dư):** Dao động ngẫu nhiên xung quanh 1.0. Kiểm tra đồ thị ACF của Residual (`plots/bai3_02_residual_acf.png`) cho thấy hầu hết các lag đều nằm trong dải tin cậy 95%, chứng minh Residual đã đạt xấp xỉ tính chất của **White Noise** (nhiễu ngẫu nhiên).

---

### 5. Vấn đề chất lượng dữ liệu (Data Quality)
* **Xử lý Missing Data:**
  * Giả lập 10 điểm khuyết (`NaN`) ngẫu nhiên.
  * **Forward-fill (`ffill`):** Tạo ra các đoạn nằm ngang giả tạo, làm bẹp sóng dao động và gãy khúc xu hướng dốc lên.
  * **Time Interpolation:** Tính toán nội suy tuyến tính theo trục thời gian, bám sát độ dốc và biên độ chu kỳ của dữ liệu gốc.
* **Phát hiện Outliers:**
  * Giả lập 2 điểm dị biệt (1 điểm tăng vọt 692.9 và 1 điểm rơi xuống 20.0).
  * Sử dụng ngưỡng động:
    $$\text{Upper} = \text{RollingMean} + 2.5 \times \text{RollingStd}$$
    $$\text{Lower} = \text{RollingMean} - 2.5 \times \text{RollingStd}$$
  * Phương pháp này phát hiện chính xác 100% hai điểm dị biệt mà không bị nhận diện nhầm các đỉnh mùa hè.
* **Đồ thị minh họa:** `plots/bai4_data_cleaning_missing_outliers.png`.

---

### 6. Ý nghĩa trực quan hóa
* **Đồ thị Rolling Statistics (Window = 12):** (`plots/bai5_01_rolling_statistics.png`)
  * Đường Rolling Mean tăng dần từ 120 lên 450 xác nhận sự vi phạm tính dừng về kỳ vọng (Mean).
  * Đường Rolling Std tăng từ 15 lên gần 50 xác nhận phương sai của chuỗi nở rộng theo thời gian.
* **Seasonal Plot theo tháng & Heatmap (Tháng $\times$ Năm):** (`plots/bai5_02_seasonal_plot_monthly.png` & `plots/bai5_03_heatmap_seasonality.png`)
  * Phản ánh rõ "bản đồ nhiệt" của ngành vận tải hàng không: Tháng 7 và 8 luôn có sắc thái màu đậm nhất, thể hiện quy luật hoạt động của hành khách qua từng năm là nhất quán.

---

### 7. Kết luận: Dữ liệu có phù hợp để dự báo không?

> ### ⭐ KẾT LUẬN: DỮ LIỆU RẤT PHÙ HỢP ĐỂ DỰ BÁO
>
> 1. **Cơ sở khoa học:** Dù chuỗi là Non-stationary (không dừng), nhưng các thành phần cấu thành (Trend tăng trưởng dài hạn và Seasonality 12 tháng) có tính quy luật cực kỳ ổn định và tính chu kỳ lặp lại hoàn hảo.
> 2. **Kiểm chứng phần dư:** Sau khi loại bỏ Trend và Mùa vụ, thành phần còn lại là sai số thuần ngẫu nhiên (White Noise), nghĩa là không còn quy luật nào bị bỏ sót.
> 3. **Mô hình khuyến nghị:**
>    * **SARIMA(p, d, q)(P, D, Q)[12]:** Lấy sai phân bậc 1 ($d=1$) để loại bỏ xu hướng và sai phân mùa bậc 1 ($D=1$) để loại bỏ mùa vụ chu kỳ 12.
>    * **Holt-Winters Exponential Smoothing:** Phiên bản Multiplicative Seasonality.
>    * **Facebook Prophet:** Rất phù hợp để dự báo và tự động giải thích các thành phần.
