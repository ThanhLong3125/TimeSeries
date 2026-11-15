# CẨM NANG HƯỚNG DẪN TRẢ LỜI VẤN ĐÁP - LAB 3 (XGBOOST TIME SERIES)

Tài liệu này được biên soạn ngắn gọn, trực diện và chuẩn xác về mặt kỹ thuật, giúp bạn tự tin trả lời vấn đáp với giảng viên hoặc hội đồng đánh giá.

---

## 1. BẢN CHẤT CỦA BÀI TOÁN & DỮ LIỆU

### Q: Em hãy trình bày bài toán của Lab 3 và nguồn dữ liệu được sử dụng?
> **Trả lời:**
> "Dạ thưa Thầy/Cô, bài toán của Lab 3 là **dự báo nhu cầu phụ tải điện theo từng giờ (Hourly Electricity Demand Forecasting)** cho tập đoàn American Electric Power thuộc lưới điện PJM Interconnection (Hoa Kỳ). Dữ liệu bao gồm hơn 121 nghìn quan sát liên tục từ năm 2004 đến 2018 với đơn vị đo là Megawatts (MW)."

### Q: Dữ liệu chuỗi thời gian thực tế này có những vấn đề gì khi tiền xử lý?
> **Trả lời:**
> "Dạ, dữ liệu chuỗi thời gian tải điện thực tế gặp 2 vấn đề đặc thù do **quy ước đổi giờ mùa hè (Daylight Saving Time - DST)**:
> 1. **Trùng lặp 4 mốc giờ vào mùa thu:** Do đồng hồ lùi 1 tiếng, em đã xử lý bằng cách khử trùng lặp (`drop_duplicates`) giữ mốc đầu tiên.
> 2. **Khuyết mốc giờ vào mùa xuân:** Do đồng hồ nhảy cóc 1 tiếng, em chuẩn hóa tần số chuỗi về cố định 1 giờ (`asfreq('h')`) và bù khuyết bằng phương pháp nội suy theo thời gian (`interpolate(method='time')`)."

---

## 2. KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)

### Q: Tại sao mô hình học máy dạng bảng (Tabular ML như XGBoost, Random Forest) lại cần kỹ nghệ đặc trưng mà không đưa chuỗi 1 chiều trực tiếp vào như ARIMA?
> **Trả lời:**
> "Dạ, các mô hình học máy truyền thống và cây quyết định nhận đầu vào là ma trận đặc trưng độc lập dạng $(X, y)$. Để mô hình hiểu được quá khứ và thời gian, ta phải chuyển đổi chuỗi 1 chiều thành bài toán học có giám sát (Supervised Learning) thông qua 3 nhóm đặc trưng:
> 1. **Lag features (Đặc trưng trễ):** Lấy giá trị phụ tải của các giờ trước ($t-1, t-2, t-3, t-24, t-48, t-168$).
> 2. **Rolling window statistics:** Tính trung bình và độ lệch chuẩn trượt (24h và 168h) để bắt xu hướng cục bộ và độ bất định.
> 3. **Calendar features:** Giờ trong ngày, thứ trong tuần, tháng, quý, cuối tuần để nắm bắt nhịp sinh hoạt xã hội."

### Q: Làm sao em đảm bảo việc tính Rolling Statistics không gây ra rò rỉ dữ liệu (Data Leakage)?
> **Trả lời:**
> "Dạ, để tránh rò rỉ giá trị hiện tại $y_t$ vào cửa sổ tính toán, em thực hiện phép trượt `shift(1)` trước khi áp dụng `.rolling()`. Như vậy, đặc trưng tại thời điểm $t$ chỉ được tính toán từ các điểm từ $t-1$ trở về trước, đảm bảo tính nhân quả thời gian tuyệt đối."

---

## 3. CÂU HỎI TRỌNG TÂM CỦA ĐỀ BÀI (MANDATORY QUESTIONS)

### Q1: Tại sao `lag_1` và `lag_24` lại có độ quan trọng cao nhất trong bài toán tải điện?
> **Trả lời:**
> - **Với `lag_1`:** Tải điện có **quán tính nhiệt và quán tính vận hành (physical & thermal inertia)** cực lớn. Các thiết bị gia dụng như điều hòa, lò sưởi và nhà máy sản xuất không bật/tắt toàn bộ trong tích tắc 1 giờ. Giá trị phụ tải lúc 14:00 phụ thuộc trực tiếp nhất vào lúc 13:00, giải thích phần lớn phương sai tức thời.
> - **Với `lag_24`:** Phản ánh **chu kỳ sinh hoạt ngày-đêm (Diurnal Pattern)**. Thói quen của con người lặp lại sau 24 giờ (đáy lúc 4h sáng, đỉnh trưa lúc 14h và đỉnh tối lúc 20h). Do đó, mức tải tại cùng một giờ giữa hôm nay và hôm qua có độ tương quan mạnh mẽ.

---

### Q2: Tại sao không được xáo trộn dữ liệu (`shuffle=True`) khi chia Train/Test?
> **Trả lời:**
> "Dạ, vì chuỗi thời gian có quan hệ nhân quả nghiêm ngặt: quá khứ sinh ra tương lai. Nếu bật `shuffle=True`:
> - Các điểm dữ liệu tương lai (ví dụ năm 2018) sẽ bị lẫn vào tập Train, còn quá khứ (năm 2015) lại rơi vào tập Test.
> - Khi đó, đặc trưng $lag_1$ của tập Test thực chất đã được mô hình học trước thông qua các điểm lân cận trong tập Train.
> - Hiện tượng này gọi là **Data Leakage (Rò rỉ dữ liệu trước tương lai / Look-ahead bias)**. Nó khiến độ chính xác trên tập Test cao ảo tưởng, nhưng khi chạy thực tế (Production), mô hình sẽ sụp đổ hoàn toàn vì dữ liệu tương lai chưa hề tồn tại."

---

### Q3: So sánh Recursive Forecasting và Direct Forecasting khi dự báo nhiều bước?
> **Trả lời:**
> - **Recursive Multi-step:**
>   - *Cơ chế:* Chỉ dùng **1 mô hình duy nhất** cho 1 bước ($t+1$). Sau đó lấy giá trị dự báo đắp ngược lại làm biến trễ đầu vào để dự báo bước $t+2, t+3, \dots, t+H$.
>   - *Ưu điểm:* Tiết kiệm tài nguyên tính toán, bảo trì dễ, có thể dự báo đến độ dài $H$ tùy ý.
>   - *Nhược điểm:* Bị **tích lũy sai số (Compounding Error)**. Sai số ở bước đầu sẽ bị khuếch đại ở các bước sau.
> - **Direct Multi-step:**
>   - *Cơ chế:* Huấn luyện **$H$ mô hình độc lập** riêng biệt cho từng bước thời gian (Model 1 dự báo $t+1$, Model 2 dự báo $t+2$,...).
>   - *Ưu điểm:* **Không bị tích lũy sai số**.
>   - *Nhược điểm:* Tốn kém chi phí tính toán và bảo trì (nếu dự báo 168 giờ phải duy trì 168 mô hình).

---

## 4. KẾT QUẢ THỰC NGHIỆM & SO SÁNH MÔ HÌNH

### Q: Kết quả so chuẩn giữa 3 mô hình (Linear Regression, Random Forest, XGBoost) như thế nào?
> **Trả lời:**
> "Dạ, trên tập kiểm thử gần 14,000 mốc giờ (2017 - 2018):
> - **Linear Regression:** MAE = 187.23 MW (MAPE = 1.27%).
> - **Random Forest:** MAE = 136.39 MW (MAPE = 0.92%).
> - **XGBoost:** MAE = **120.34 MW** (MAPE = **0.82%**).
> 
> XGBoost cho kết quả vượt trội nhất vì thuật toán Gradient Boosting tối ưu hóa trực tiếp hàm mục tiêu dư sai số, chia nhánh cây linh hoạt để bắt trọn sự tương tác phi tuyến tính giữa giờ trong ngày và các biến trễ nhiệt độ mùa hè/mùa đông."
