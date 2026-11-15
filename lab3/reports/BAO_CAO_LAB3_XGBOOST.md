# BÁO CÁO THỰC HÀNH LAB 3: DỰ BÁO CHUỖI THỜI GIAN PHỤ TẢI ĐIỆN VỚI MÔ HÌNH HỌC MÁY (XGBOOST, RANDOM FOREST & LINEAR REGRESSION)

---

## 1. TỔNG QUAN BÀI TOÁN & TẬP DỮ LIỆU
- **Tên bài toán:** Dự báo nhu cầu tiêu thụ phụ tải điện năng theo từng giờ (Hourly Electricity Consumption Demand Forecasting).
- **Tập dữ liệu:** `AEP_hourly.csv` thuộc tập đoàn **American Electric Power (AEP)** trong mạng lưới điện PJM Interconnection (Hoa Kỳ).
- **Phạm vi thời gian:** Từ `01/10/2004 01:00:00` đến `03/08/2018 00:00:00` (gần 14 năm dữ liệu liên tục).
- **Số lượng quan sát:** 121,273 mốc giờ ban đầu.
- **Đơn vị đo:** Megawatts (MW).
- **Đặc thù kỹ thuật:**
  - Chuỗi mang tính mùa vụ đa tầng (Multi-seasonal): chu kỳ 24 giờ ngày-đêm (Diurnal pattern), chu kỳ 7 ngày trong tuần (Weekly pattern) và chu kỳ thời tiết 4 mùa trong năm (Annual pattern).
  - Xuất hiện hiện tượng đổi giờ mùa hè (Daylight Saving Time - DST): gây trùng lặp 4 mốc giờ vào mùa thu (khi đồng hồ vặn lùi 1 giờ) và khuyết các mốc giờ vào mùa xuân (khi đồng hồ nhảy cóc 1 giờ).

---

## 2. QUY TRÌNH TIỀN XỬ LÝ & KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)

### 2.1. Tiền xử lý dữ liệu (Data Preprocessing)
1. **Chuyển đổi Datetime:** Định dạng `Datetime` chuẩn hóa và thiết lập làm chỉ mục thời gian (`pd.DatetimeIndex`).
2. **Khử trùng lặp:** Loại bỏ 4 bản ghi bị lặp do DST bằng phương pháp giữ bản ghi đầu tiên (`drop_duplicates`).
3. **Chuẩn hóa tần số cố định:** Đồng bộ hóa chuỗi với tần số 1 giờ liên tục (`asfreq('h')`).
4. **Nội suy mốc khuyết:** Áp dụng phương pháp nội suy theo thời gian (`interpolate(method='time')`) cho các điểm khuyết, đảm bảo chuỗi hoàn toàn không gián đoạn (121,296 mốc giờ).

### 2.2. Kỹ nghệ đặc trưng cho học có giám sát (Tabular Supervised Learning)
Để biến đổi bài toán chuỗi thời gian thành dạng bảng:
1. **Đặc trưng độ trễ (Lag Features):**
   - $t-1, t-2, t-3$: Đại diện cho quán tính tải điện ngắn hạn và nhiệt độ tức thời.
   - $t-24, t-48$: Đại diện cho mức tiêu thụ cùng giờ của hôm qua và hôm kia (chu kỳ ngày).
   - $t-168$: Đại diện cho mức tiêu thụ đúng giờ này vào tuần trước ($7 \times 24 = 168$ giờ).
2. **Đặc trưng thống kê trượt (Rolling Window Statistics):**
   - Áp dụng trên `shift(1)` để **ngăn ngừa hoàn toàn Look-ahead Bias (Data Leakage)**.
   - `rolling_mean_24h`, `rolling_std_24h`: Xu hướng trung bình và biên độ dao động trong ngày.
   - `rolling_mean_168h`, `rolling_std_168h`: Xu hướng nền tảng và độ biến thiên trong tuần.
3. **Đặc trưng lịch (Calendar Features):**
   - `hour` (0 - 23): Giờ trong ngày.
   - `dayofweek` (0 - 6): Ngày trong tuần (Thứ Hai = 0, Chủ Nhật = 6).
   - `quarter` (1 - 4), `month` (1 - 12), `year`: Mùa vụ và xu hướng dài hạn.
   - `is_weekend` (0 hoặc 1): Phân biệt ngày thường và cuối tuần.

---

## 3. THIẾT KẾ THỰC NGHIỆM & KẾT QUẢ SO CHUẨN (BENCHMARKING)

### 3.1. Phân chia Train / Test
- **Quy tắc:** Phân chia theo trình tự thời gian tuyệt đối (`shuffle=False`).
- **Tập Train:** Toàn bộ dữ liệu trước ngày `2017-01-01` (107,231 mẫu, từ 10/2004 đến 12/2016).
- **Tập Test:** Toàn bộ dữ liệu từ `2017-01-01` đến `03/08/2018` (13,897 mẫu, ~1.6 năm).

### 3.2. Bảng tổng hợp kết quả đánh giá (Evaluation Metrics)

| Mô hình (Model) | MAE (MW) | RMSE (MW) | MAPE (%) | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | 187.21 | 249.58 | 1.27% | Nhanh nhưng không học được tương tác phi tuyến phức tạp |
| **Random Forest Regressor** | 137.29 | 183.76 | 0.93% | Tốt, giảm phương sai hiệu quả, độ chính xác cao |
| **XGBoost Regressor (SOTA)** | **133.54** | **175.36** | **0.91%** | **Tối ưu nhất trên cả 3 chỉ số**, sai số chỉ 0.91% |

### 3.3. Phân tích Feature Importance (Tầm quan trọng đặc trưng)
Dựa trên thuật toán XGBoost:
1. **`lag_1` chiếm ~93.5% tổng Gain Importance:** Phụ tải điện có tính quán tính nhiệt cực kỳ lớn; giá trị giờ trước là yếu tố quyết định tới mức tiêu thụ giờ sau.
2. **`lag_3` và `hour`:** Chiếm tỷ trọng tiếp theo, giúp định hình biên độ tăng giảm theo pha trong ngày.
3. **`rolling_mean_24h` và `lag_168`:** Đóng vai trò cân chỉnh mức nền tảng theo tuần và chu kỳ làm việc.

---

## 4. DỰ BÁO ĐA BƯỚC ĐỆ QUY (RECURSIVE MULTI-STEP FORECASTING: 24 GIỜ)
- Thử nghiệm dự báo cho 24 giờ của ngày đầu tiên trong năm kiểm thử (`2018-01-01`).
- **Phương pháp:** Tại mỗi bước $t+k$, dự báo được sinh ra sẽ lập tức được chèn vào chuỗi lịch sử để tính toán lại các biến trễ `lag_1`, `lag_2`, `lag_3`, `rolling_mean_24h` cho bước $t+k+1$.
- **Kết quả:**
  - **MAE (24 giờ):** 300.99 MW.
  - **MAPE (24 giờ):** 1.83%.
  - Dự báo bám rất sát đường cong thực tế, thể hiện rõ pha giảm lúc rạng sáng (đáy 4 AM) và pha tăng vọt vào giờ cao điểm buổi tối.

---

## 5. TRẢ LỜI CHI TIẾT 3 CÂU HỎI BẮT BUỘC (MANDATORY QUESTIONS)

### Câu hỏi 1: Tại sao `lag_1` và `lag_24` lại có độ quan trọng cao nhất trong bài toán phụ tải điện?
- **Đối với `lag_1` (Quán tính nhiệt & thói quen sinh hoạt ngắn hạn):**
  - Hệ thống điện lưới quốc gia phục vụ hàng triệu thiết bị tiêu thụ công nghiệp và dân dụng. Các thiết bị như máy điều hòa, lò luyện kim, dây chuyền sản xuất, đèn chiếu sáng không thể đồng loạt bật hoặc tắt tức thì trong vòng 1 giờ.
  - Ngoài ra, nhiệt độ môi trường (yếu tố quyết định tải điều hòa/sưởi) có tính trễ nhiệt lớn (thermal inertia). Do đó, mức tải tại $t$ có tương quan tự hồi quy cực kỳ chặt chẽ với $t-1$.
- **Đối với `lag_24` (Chu kỳ nhịp sinh học ngày-đêm / Diurnal Rhythm):**
  - Hành vi con người lặp lại mang tính chu kỳ cao: thức dậy lúc 6:00, làm việc lúc 8:00–17:00, nấu nướng và giải trí lúc 19:00–21:00, đi ngủ lúc 23:00.
  - Mức tiêu thụ điện lúc 14:00 hôm nay phản ánh chính xác hoạt động của nhà máy và văn phòng tương tự như lúc 14:00 hôm qua. Vì vậy, `lag_24` là đặc trưng giải thích mạnh mẽ nhất cho tính mùa vụ trong ngày.

---

### Câu hỏi 2: Tại sao không được xáo trộn dữ liệu (`shuffle=True`) khi chia Train/Test trong bài toán chuỗi thời gian? Hiện tượng rò rỉ dữ liệu (Data Leakage) diễn ra như thế nào?
- **Tính tuần tự thời gian (Temporal Causality):** Bản chất của vũ trụ là quá khứ sinh ra hiện tại, và hiện tại quyết định tương lai ($t$ phụ thuộc vào $< t$).
- **Cơ chế rò rỉ dữ liệu khi `shuffle=True`:**
  - Nếu xáo trộn ngẫu nhiên, các điểm ở năm 2018 sẽ bị rơi vào tập Train, còn các điểm ở năm 2015 lại rơi vào tập Test.
  - Khi tạo biến trễ $lag_1$ cho mẫu ở tập Test (ví dụ: ngày 15/05/2015 10:00), mô hình ở tập Train thực tế đã được "nhìn thấy" mẫu của ngày 15/05/2015 11:00.
  - Mô hình không còn làm nhiệm vụ **dự báo (forecasting)** mà chuyển thành bài toán **nội suy (interpolation)** giữa hai mốc thời gian đã biết.
- **Hậu quả:**
  - Độ đo đánh giá trên tập kiểm thử đạt độ chính xác giả tạo (Over-optimistic / Overfitting).
  - Khi triển khai thực tế trên Production, mô hình hoàn toàn thất bại vì dữ liệu tương lai chưa từng tồn tại để cung cấp biến trễ.

---

### Câu hỏi 3: So sánh ưu điểm và nhược điểm của phương pháp Recursive Forecasting và Direct Forecasting khi dự báo nhiều bước (Multi-step ahead)?

| Tiêu chí so sánh | Phương pháp Recursive Multi-step | Phương pháp Direct Multi-step |
| :--- | :--- | :--- |
| **Bản chất kiến trúc** | Huấn luyện **duy nhất 1 mô hình** dự báo 1 bước ($t+1$). Sau đó lấy dự báo đưa vào làm đặc trưng đầu vào để đệ quy tính $t+2, \dots, t+H$. | Huấn luyện **$H$ mô hình độc lập**:<br>- Model 1 dự báo $t+1$<br>- Model 2 dự báo $t+2$<br>...<br>- Model $H$ dự báo $t+H$. |
| **Ưu điểm** | 1. **Tiết kiệm tài nguyên:** Chỉ cần 1 mô hình duy nhất, chi phí huấn luyện thấp.<br>2. **Linh hoạt:** Có thể dự báo đến bất kỳ độ dài $H$ nào mà không cần tạo hay huấn luyện lại mô hình mới. | 1. **Không bị tích lũy sai số (No Error Cascading):** Dự báo bước $t+24$ không bị ảnh hưởng bởi sai số của $t+1$.<br>2. **Tối ưu hóa chuyên biệt:** Mỗi mô hình chuyên trách học đặc thù của bước tương ứng. |
| **Nhược điểm** | **Tích lũy sai số (Compounding Error Accumulation):** Sai số ở bước $t+1$ sẽ trở thành dữ liệu đầu vào cho bước $t+2$, làm sai số tăng theo cấp số nhân ở các bước xa. | 1. **Chi phí tính toán rất lớn:** Nếu dự báo 1 tuần ($H=168$), cần lưu trữ và huấn luyện đồng thời 168 mô hình.<br>2. **Mất liên kết chuỗi:** Không tận dụng được mối quan hệ phụ thuộc chéo giữa các bước liên tiếp. |

---

## 6. DANH MỤC BIỂU ĐỒ & ARTIFACTS TRỰC QUAN ĐÃ TẠO

1. `lab3/artifacts/figures/lab3_01_eda_raw_series.png`: Biểu đồ toàn bộ chuỗi phụ tải điện AEP từ 2004 đến 2018 (đường trung bình 15,499 MW).
2. `lab3/artifacts/figures/lab3_02_eda_daily_pattern.png`: Chu kỳ tiêu thụ 24 giờ ngày-đêm với dải độ lệch chuẩn $\pm 1\sigma$ và chú thích đáy 4 AM, đỉnh 20 PM.
3. `lab3/artifacts/figures/lab3_03_model_benchmark.png`: Biểu đồ cột so chuẩn hiệu năng (MAE, RMSE, MAPE) giữa Linear Regression, Random Forest và XGBoost.
4. `lab3/artifacts/figures/lab3_04_feature_importance.png`: Thứ hạng đóng góp của các đặc trưng trong XGBoost (`lag_1` dẫn đầu áp đảo).
5. `lab3/artifacts/figures/lab3_05_forecast_sample.png`: So sánh chi tiết đường cong thực tế và dự báo trong 2 tuần mẫu (01/07/2017 – 14/07/2017).
6. `lab3/artifacts/figures/lab3_06_multistep_recursive.png`: Kết quả dự báo đệ quy 24 giờ tiếp theo (MAPE đạt 1.83%).
