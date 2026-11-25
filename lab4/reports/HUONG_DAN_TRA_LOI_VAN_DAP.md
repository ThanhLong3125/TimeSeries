# CẨM NANG HƯỚNG DẪN TRẢ LỜI VẤN ĐÁP - LAB 4 (MULTIVARIATE TIME SERIES)

Tài liệu được biên soạn ngắn gọn, trực diện, chuẩn xác về mặt toán học và học máy, giúp bạn tự tin đạt điểm tối đa khi vấn đáp với giảng viên hoặc hội đồng.

---

## 1. TỔNG QUAN BÀI TOÁN & DỮ LIỆU ĐA BIẾN

### Q: Em hãy trình bày bài toán của Lab 4 và ý nghĩa của việc đưa thêm các biến thời tiết vào mô hình?
> **Trả lời:**
> "Dạ thưa Thầy/Cô, bài toán của Lab 4 là **dự báo chuỗi thời gian đa biến (Multivariate Time Series Forecasting)**, cụ thể là dự báo điện năng tiêu thụ (`load`) bằng cách kết hợp lịch sử phụ tải với 3 biến khí tượng thời tiết: `temperature` (nhiệt độ), `humidity` (độ ẩm), và `wind_speed` (tốc độ gió).
> Việc kết hợp thời tiết là cực kỳ cần thiết vì nhu cầu tiêu thụ điện chịu tác động trực tiếp từ hành vi con người thích ứng với môi trường: khi nhiệt độ tăng cao (nắng nóng oi bức), nhu cầu sử dụng máy điều hòa, thiết bị làm mát và thông gió tăng vọt; khi nhiệt độ quá thấp, nhu cầu sưởi ấm gia tăng."

### Q: Tại sao lại cần chuẩn hóa toàn bộ dữ liệu bằng `StandardScaler` trước khi huấn luyện?
> **Trả lời:**
> "Dạ, có 3 lý do kỹ thuật quan trọng:
> 1. **Khác biệt về đơn vị và biên độ đo lường:** `load` có độ lớn hàng chục nghìn (30,000 kW), trong khi `temperature` dao động từ 3°C – 40°C, `humidity` từ 10% – 100%, và `wind_speed` chỉ từ 0 – 6 m/s. Nếu không chuẩn hóa, các mô hình dựa trên gradient như LSTM sẽ bị chi phối hoàn toàn bởi biến `load`, làm triệt tiêu tín hiệu của các biến thời tiết.
> 2. **Ổn định gradient cho mạng nơ-ron sâu (LSTM):** Giúp tránh hiện tượng bùng nổ gradient (Gradient Explosion) hoặc biến mất gradient (Vanishing Gradient), giúp thuật toán Adam hội tụ nhanh và ổn định.
> 3. **Cải thiện ma trận hiệp phương sai cho VAR:** Giúp ma trận hiệp phương sai của mô hình thống kê VAR không bị suy biến do chênh lệch phương sai giữa các chuỗi."

---

## 2. KỸ NGHỆ ĐẶC TRƯNG & CƠ CHẾ DỮ LIỆU

### Q: Cửa sổ trễ $L=24$ giờ được biểu diễn khác nhau như thế nào giữa XGBoost và LSTM?
> **Trả lời:**
> - **Với XGBoost (Machine Learning):** Mô hình dạng bảng nhận đầu vào là một **Vector phẳng 2D** (Flattened Vector). Do đó, ta phải trải phẳng 24 bước trễ của 4 biến thành $24 \times 4 = 96$ cột đặc trưng độc lập, cộng thêm 3 đặc trưng trượt (`rolling_mean_6h`, `rolling_std_24h`, `load_trend`), tạo thành vector 99 chiều cho mỗi dòng quan sát.
> - **Với LSTM (Deep Learning):** LSTM nhận đầu vào là **Tensor 3D** có kích thước `(batch_size, 24, 4)`. Ta không cần làm phẳng dữ liệu mà giữ nguyên vẹn chiều không gian (4 biến) và chiều thời gian (24 bước), cho phép mạng tự động trích xuất các đặc trưng tuần tự và động học thời gian."

### Q: Tại sao phân chia dữ liệu Train/Val/Test tuyệt đối không được dùng `shuffle=True`?
> **Trả lời:**
> "Dạ, vì chuỗi thời gian tuân thủ nghiêm ngặt **tính nhân quả thời gian (Temporal Causality)**: quá khứ sinh ra hiện tại và tương lai. Nếu bật `shuffle=True`:
> - Dữ liệu tương lai (ví dụ tháng 11, 12) sẽ bị trộn lẫn vào tập Train, còn quá khứ rơi vào tập Test.
> - Các biến trễ $t-1$ của tập Test thực chất đã được mô hình 'nhìn thấy trước' trong tập Train.
> - Đây là hiện tượng **Rò rỉ dữ liệu trước tương lai (Look-ahead Data Leakage)**, khiến mô hình đạt điểm số cao giả tạo trên bài thi nhưng sẽ hỏng hoàn toàn khi triển khai thực tế trên Production."

---

## 3. TRẢ LỜI 6 CÂU HỎI TRỌNG TÂM CỦA ĐỀ BÀI (MANDATORY QUESTIONS)

### Q1: Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?
> **Trả lời:**
> "Dạ, **Nhiệt độ (Temperature)** là biến thời tiết có ảnh hưởng mạnh nhất và tương quan cao nhất ($r = +0.444$).
> Mối quan hệ giữa tải điện và nhiệt độ có tính chất **phi tuyến dạng chữ U (U-shaped)**: phụ tải đạt mức đáy ở vùng nhiệt độ dễ chịu (18°C – 22°C), và tăng vọt theo hàm mũ khi nhiệt độ vượt quá 28°C do người dân và các cơ sở công nghiệp đồng loạt bật điều hòa không khí (HVAC)."

---

### Q2: Khi nào mô hình VAR thất bại?
> **Trả lời:**
> "Dạ, mô hình VAR thất bại trong 3 trường hợp chính:
> 1. **Khi quan hệ mang tính phi tuyến mạnh:** VAR là mô hình tuyến tính tổ hợp các bước trễ ($X_t = \sum A_i X_{t-i} + \varepsilon_t$). Nếu hàm phụ tải theo nhiệt độ có dạng phi tuyến chữ U, VAR không thể biểu diễn chính xác.
> 2. **Hiện tượng bùng nổ số lượng tham số (Curse of Dimensionality):** Số tham số của VAR tăng theo $K^2 \cdot p$. Với 4 biến và 24 trễ, mô hình phải ước lượng gần 400 hệ số, dễ dẫn tới quá khớp (overfitting) hoặc mất ổn định ước lượng nếu mẫu không đủ lớn.
> 3. **Chuỗi không dừng (Non-stationarity):** Nếu dữ liệu có xu hướng hoặc phương sai thay đổi mà phép sai phân không giải quyết triệt để, VAR sẽ cho ra dự báo bùng nổ vô hạn."

---

### Q3: XGBoost học được thông tin gì từ các biến thời tiết?
> **Trả lời:**
> "Dạ, XGBoost học được 2 thông tin rất giá trị từ thời tiết:
> 1. **Các ngưỡng nhiệt độ kích hoạt hành vi (Threshold Splits):** Cây quyết định tự động tìm ra các điểm cắt tối ưu (ví dụ: `if temperature > 27.5°C thì tăng mạnh dự báo phụ tải`), phản ánh đúng ngưỡng người dùng bắt đầu bật điều hòa.
> 2. **Tương tác chéo đa biến (Feature Interactions):** XGBoost kết hợp nhánh giữa `temperature` và `humidity` để học chỉ số oi bức (Heat Index) — khi nhiệt độ cao đi kèm độ ẩm cao, máy lạnh tiêu tốn nhiều điện năng hơn để vừa làm mát vừa hút ẩm."

---

### Q4: Mạng LSTM học được chu kỳ nào (ngày, tuần)?
> **Trả lời:**
> "Dạ, mạng LSTM học được trọn vẹn **chu kỳ ngày-đêm (Diurnal Pattern 24 giờ)**:
> Nhờ cửa sổ quan sát 24 bước thời gian, các cổng nhớ (Cell State, Forget Gate) của LSTM học được quy luật chuyển pha trong 1 ngày: từ đáy đêm 4h sáng, pha tăng tốc buổi sáng, đỉnh trưa 13h và đỉnh tối 20h. Đồng thời, trạng thái ẩn $h_t$ cũng giúp mô hình cảm nhận được sự chuyển dịch tải giữa ngày thường và ngày cuối tuần."

---

### Q5: Mạng LSTM có bị overfitting không?
> **Trả lời:**
> "Dạ, trong thực nghiệm của em, mạng LSTM **hoàn toàn không bị overfitting**.
> Bằng chứng là trên đồ thị Learning Curve, Train Loss và Validation Loss giảm đồng pha cùng nhau (Train MSE từ 0.373 xuống 0.019; Val MSE từ 0.185 xuống 0.021). Nhờ cơ chế **Early Stopping** giám sát tập Validation với kiên nhẫn 10 epochs, mô hình đã tự động dừng huấn luyện tại Epoch 28 và nạp lại bộ trọng số tối ưu nhất, ngăn chặn triệt để nguy cơ học vẹt."

---

### Q6: So sánh sự khác biệt bản chất giữa Machine Learning (XGBoost) và Deep Learning (LSTM)?
> **Trả lời:**
> "Dạ, sự khác biệt nằm ở 4 điểm cốt lõi:
> 1. **Cấu trúc biểu diễn:** XGBoost nhận vector phẳng 2D và cần con người can thiệp kỹ nghệ đặc trưng thủ công (rolling mean, std, trend). Ngược lại, LSTM nhận Tensor 3D $(N, 24, 4)$ và tự động trích xuất đặc trưng tuần tự thời gian (End-to-End Representation Learning).
> 2. **Cơ chế thời gian:** XGBoost coi các bước trễ $t-1, \dots, t-24$ như các thuộc tính độc lập và phân vùng không gian bằng cây quyết định; còn LSTM duy trì một dòng chảy thông tin thời gian liên tục thông qua bộ nhớ ẩn ($h_t, c_t$).
> 3. **Tốc độ huấn luyện:** XGBoost huấn luyện cực nhanh (chỉ mất ~2.8 giây), trong khi LSTM cần tối ưu qua nhiều epochs lan truyền ngược qua thời gian (BPTT, ~16.4 giây).
> 4. **Khả năng diễn giải:** XGBoost cho biết trực tiếp độ quan trọng của từng đặc trưng (Feature Importance), còn LSTM là mô hình hộp đen phức tạp."

---

## 4. BÀI TẬP NÂNG CAO: DỰ BÁO LƯU LƯỢNG GIAO THÔNG (TRAFFIC FORECASTING)

### Q: Em hãy trình bày bài toán Traffic Forecasting trên tập METR-LA / PEMS-BAY và nguyên lý của mô hình DCRNN?
> **Trả lời:**
> "Dạ, bài toán là dự báo tốc độ xe chạy đồng thời tại **207 trạm cảm biến** trên đường cao tốc ở các mốc tương lai 15, 30, 60 phút:
> - **Thách thức:** Vừa có tính phụ thuộc không gian (sự lan truyền tắc đường giữa các trạm trên mạng lưới đường xá có hướng) vừa có tính phụ thuộc thời gian (giờ cao điểm).
> - **Giải pháp DCRNN (Diffusion Convolutional Recurrent Neural Network):**
>   1. **Không gian:** Sử dụng **tích chập khuếch tán trên đồ thị có hướng (Diffusion Graph Convolution)** dựa trên chuỗi ngẫu nhiên Markov để nắm bắt sự lan truyền xuôi dòng và ngược dòng của dòng xe.
>   2. **Thời gian:** Thay thế ma trận trọng số trong tế bào GRU bằng tích chập khuếch tán (DCGRU) kết hợp cấu trúc **Seq2Seq (Encoder-Decoder)** và **Scheduled Sampling** để dự báo đa bước xa mà không bị khuếch đại sai số."
