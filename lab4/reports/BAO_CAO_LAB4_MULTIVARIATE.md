# BÁO CÁO THỰC HÀNH LAB 4: DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (MULTIVARIATE FORECASTING)
## SO SÁNH BA TRƯỜNG PHÁI: VAR (THỐNG KÊ), XGBOOST (MACHINE LEARNING) VÀ LSTM (DEEP LEARNING)

---

## 1. TỔNG QUAN BÀI TOÁN & TẬP DỮ LIỆU
- **Tên bài toán:** Dự báo phụ tải điện năng đa biến kết hợp dữ liệu khí tượng thời tiết (Multivariate Short-term Electricity Load Forecasting with Weather).
- **Tập dữ liệu:** UCI Electricity Load & Weather Dataset (Power Consumption of Tetouan City, Morocco).
- **Phạm vi thời gian:** Ghi nhận liên tục 8,736 mốc giờ trong suốt năm 2017 (tần số 1 giờ liên tục, 0 missing values).
- **Vector quan sát tại mỗi thời điểm $t$:**
  $$X_t = [load_t, temp_t, humidity_t, wind_t]$$
  - `load`: Điện năng tiêu thụ phụ tải (kW).
  - `temperature`: Nhiệt độ không khí (°C).
  - `humidity`: Độ ẩm tương đối (%).
  - `wind_speed`: Tốc độ gió (m/s).
- **Mục tiêu bài toán:** Sử dụng lịch sử 24 giờ gần nhất của toàn bộ 4 biến để dự báo phụ tải điện ở bước tiếp theo:
  $$X_{t-24:t} \Longrightarrow load_{t+1}$$

---

## 2. PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (DATA UNDERSTANDING & EDA)

### 2.1. Chu kỳ tiêu thụ và tính dừng (Seasonality & Stationarity)
- **Chu kỳ ngày (Diurnal Pattern 24h):** Phụ tải chạm đáy sâu vào khoảng 4:00 – 5:00 sáng (~18,000 kW), sau đó tăng dốc khi các hoạt động sản xuất bắt đầu và đạt đỉnh đôi vào khoảng 13:00 – 14:00 (buổi trưa) và 20:00 – 21:00 (giờ cao điểm sinh hoạt gia đình).
- **Kiểm định Augmented Dickey-Fuller (ADF Test):**
  - `load`: ADF = -4.8271, p-value = 4.82e-05 $\rightarrow$ Chuỗi dừng ở mức ý nghĩa 1%.
  - `temperature`: ADF = -2.4740, p-value = 0.1219 $\rightarrow$ Chuỗi không dừng do tính mùa vụ nhiệt độ mùa hè/mùa đông.
  - `humidity`: ADF = -8.3883, p-value = 2.40e-13 $\rightarrow$ Chuỗi dừng.
  - `wind_speed`: ADF = -5.4006, p-value = 3.38e-06 $\rightarrow$ Chuỗi dừng.
  - Do có biến không dừng (`temperature`), mô hình VAR được áp dụng sai phân bậc 1 ($d=1$) để đảm bảo tính ổn định và tính khả nghịch.

### 2.2. Mối quan hệ giữa Phụ tải và Thời tiết
- **Tương quan tuyến tính:**
  - $r(load, temperature) = +0.444$: Tương quan dương rõ rệt. Khi nhiệt độ môi trường tăng cao, nhu cầu sử dụng điện tăng vọt.
  - $r(load, humidity) = -0.291$: Tương quan âm vừa phải.
  - $r(load, wind_speed) = +0.170$: Tương quan dương nhẹ.
- **Tính phi tuyến dạng chữ U (U-shaped Relationship):**
  - Đồ thị phân tán `load vs temperature` thể hiện rõ đường cong hồi quy bậc 2: Phụ tải điện ở mức thấp nhất khi nhiệt độ rơi vào vùng dễ chịu (18°C – 22°C), và tăng vọt theo cấp số nhân khi nhiệt độ vượt quá 28°C (kích hoạt hàng loạt máy lạnh công suất cao).

---

## 3. QUY TRÌNH TIỀN XỬ LÝ & KỸ NGHỆ ĐẶC TRƯNG

### 3.1. Chuẩn hóa & Phân chia theo thời gian
- **Phân chia không shuffle (Strict Chronological Split):**
  - **Train (70%):** 6,115 mốc giờ đầu tiên (01/01/2017 $\rightarrow$ 12/09/2017).
  - **Validation (15%):** 1,310 mốc giờ tiếp theo (12/09/2017 $\rightarrow$ 06/11/2017).
  - **Test (15%):** 1,311 mốc giờ cuối cùng (06/11/2017 $\rightarrow$ 30/12/2017).
- **Chuẩn hóa `StandardScaler`:** Fit duy nhất trên tập Train, sau đó transform cho Train, Validation và Test để chống rò rỉ dữ liệu (Data Leakage).

### 3.2. Cấu trúc đặc trưng cho từng mô hình
1. **XGBoost (Tabular Format):**
   - Cửa sổ trễ 24 giờ cho cả 4 biến: $24 \times 4 = 96$ lag features.
   - 3 Rolling Features trên `load`: `rolling_mean_6h`, `rolling_std_24h`, `load_trend = load_{t-1} - load_{t-24}`.
   - Tổng cộng: **99 đặc trưng đầu vào**.
2. **LSTM (3D Tensor Format):**
   - Cấu trúc Tensor: `(N, 24 timesteps, 4 features)`.
   - Giữ nguyên vẹn tính tuần tự thời gian và tương quan chéo giữa 4 kênh đo đạc.
3. **VAR (Multivariate Matrix):**
   - Ma trận liên tục 4 biến $Y_t \in \mathbb{R}^4$.

---

## 4. KẾT QUẢ THỰC NGHIỆM & SO CHUẨN HIỆU NĂNG

### 4.1. Bảng so sánh 3 mô hình trên tập Test (Thang đo gốc kW)

| Mô hình (Model) | RMSE (kW) | MAE (kW) | MAPE (%) | Thời gian huấn luyện | Nhận xét đặc thù |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **VAR (Vector Autoregression)** | **641.15** | **443.01** | **1.57%** | **~0.12 s** | Tối ưu xuất sắc ở bài toán 1 bước thời gian ngắn hạn |
| **XGBoost Regressor (ML)** | **970.12** | **683.33** | **2.40%** | **~2.85 s** | Học rất tốt các tương tác chéo và ngưỡng nhiệt độ |
| **LSTM (Deep Learning)** | **1,356.93** | **961.10** | **3.59%** | **~16.40 s** | Học biểu diễn không gian ẩn, không bị overfitting |

> **Phân tích kết quả:**
> - Cả 3 mô hình đều đạt độ chính xác rất cao với sai số MAPE < 3.6% trên tập kiểm thử hơn 1,300 giờ.
> - **VAR** đạt hiệu năng dẫn đầu ở bài toán One-step Ahead (dự báo 1 giờ tiếp theo) nhờ cấu trúc tự hồi quy trực tiếp trên chuỗi sai phân tuyến tính.
> - **XGBoost** có ưu thế vượt trội về việc kết hợp linh hoạt giữa các biến trễ và các chỉ số trượt (rolling mean, rolling std).
> - **LSTM** học được trọn vẹn hình thái biến thiên chu kỳ ngày đêm và không bị hiện tượng overfitting nhờ cơ chế Early Stopping tại Epoch 28 (Validation MSE Loss = 0.02177).

---

## 5. TRẢ LỜI CHI TIẾT 6 CÂU HỎI PHÂN TÍCH BẮT BUỘC

### Câu 1: Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?
- **Nhiệt độ (Temperature)** là biến có tác động mạnh mẽ và chi phối nhất đến phụ tải điện.
- **Phân tích cơ chế:**
  - Mối quan hệ giữa tải điện và nhiệt độ là phi tuyến dạng **chữ U (U-shaped Curve)**.
  - Khi nhiệt độ tăng vượt ngưỡng 26°C – 30°C, các hệ thống điều hòa không khí (HVAC) và tủ lạnh gia dụng/công nghiệp hoạt động hết công suất, làm phụ tải tăng vọt theo hàm mũ.
  - Ngược lại, khi nhiệt độ xuống thấp vào ban đêm hoặc mùa đông, nhu cầu sưởi ấm cũng làm phụ tải gia tăng.
  - Tốc độ gió và độ ẩm đóng vai trò điều chỉnh thứ cấp (độ ẩm cao làm gia tăng nhiệt độ cảm nhận Heat Index).

---

### Câu 2: Khi nào mô hình VAR thất bại?
1. **Mối quan hệ mang tính phi tuyến cao:** VAR giả định bản chất chuỗi là tổ hợp tuyến tính của các bước trễ ($X_t = \sum A_i X_{t-i} + \varepsilon_t$). Khi quan hệ giữa các biến là phi tuyến (như phụ tải vs nhiệt độ theo hàm bậc 2 hoặc bậc 3), VAR không thể nắm bắt được biến động này.
2. **Hiện tượng bùng nổ số lượng tham số (Curse of Dimensionality):**
   - Với $K$ biến và bậc trễ $p$, VAR phải ước lượng ma trận tham số có kích thước $K \times (K \cdot p + 1)$.
   - Với $K=4$ biến và $p=24$ lag, số tham số cần ước lượng là $4 \times (4 \times 24 + 1) = 388$ hệ số. Khi tập dữ liệu nhỏ hoặc chuỗi có nhiều biến ngoại sinh, VAR dễ bị quá khớp hoặc ma trận hiệp phương sai bị suy biến.
3. **Chuỗi không thỏa mãn tính dừng (Non-stationarity):** Nếu chuỗi có xu hướng mạnh hoặc biến động phương sai mà sai phân bậc 1 không khử được, ma trận trọng số $A_i$ của VAR sẽ không hội tụ và sinh ra dự báo bùng nổ vô hạn.

---

### Câu 3: XGBoost học được thông tin gì từ các biến thời tiết?
1. **Các ngưỡng kích hoạt nhiệt độ (Threshold Split Points):**
   - Cây quyết định trong XGBoost tự động tìm các điểm phân nhánh tối ưu (ví dụ: `if temperature > 27.5°C then load_increase = +4500 kW`). Điều này mô phỏng chính xác ngưỡng hành vi con người bắt đầu bật máy điều hòa.
2. **Tương tác chéo đa biến (Cross-feature Interactions):**
   - XGBoost tự động kết hợp nhánh cây giữa `temperature` và `humidity` để phản ánh chỉ số oi bức (Heat Index): cùng là nhiệt độ 30°C nhưng nếu độ ẩm 85% thì phụ tải cao hơn đáng kể so với khi độ ẩm 45%.
3. **Xu hướng tải nhiệt ngắn hạn:**
   - Kết hợp giữa biến `load_trend` ($load_{t-1} - load_{t-24}$) và độ trễ nhiệt độ `temperature_lag_1` để xác định pha phụ tải đang ở chu kỳ tăng nhiệt hay giảm nhiệt.

---

### Câu 4: LSTM học được chu kỳ nào (ngày, tuần)?
- **Chu kỳ ngày-đêm (Diurnal Cycle 24h):**
  - Với cửa sổ quan sát đầu vào $L=24$ bước thời gian, mạng LSTM bao quát toàn bộ một vòng quay ngày-đêm của hành vi con người và chu trình thời tiết (nắng lên lúc 6h, nóng nhất lúc 14h, dịu mát lúc 19h).
  - Các cổng quên (Forget Gate) và cổng vào (Input Gate) trong LSTM học cách reset và cập nhật thông tin theo nhịp 24 giờ, giúp mô hình dự báo chính xác các đỉnh tải buổi trưa, đỉnh tối và đáy đêm.
- **Chu kỳ tuần (Weekly Cycle):**
  - Mặc dù cửa sổ trực tiếp là 24h, nhưng thông qua trạng thái ẩn $h_t$ và chuỗi giá trị biến thiên của ngày hôm trước, LSTM vẫn nhận biết được sự khác biệt giữa ngày trong tuần (phụ tải cao) và ngày cuối tuần (phụ tải công nghiệp giảm).

---

### Câu 5: Mạng LSTM có bị overfitting không?
- **Kết quả thực nghiệm:** Mạng LSTM **không bị overfitting**.
- **Minh chứng từ đồ thị học tập (Loss Curve):**
  - Train MSE Loss giảm đều đặn từ 0.373 xuống 0.019.
  - Validation MSE Loss giảm đồng pha từ 0.185 xuống 0.021 và duy trì ổn định.
  - Không xuất hiện hiện tượng phân kỳ (Train Loss tiếp tục giảm sâu trong khi Val Loss bật ngược tăng vọt).
- **Yếu tố kiểm soát:**
  1. Kiến trúc vừa vặn, không quá sâu: $\text{LSTM}(64) \rightarrow \text{LSTM}(32) \rightarrow \text{Dense}(1)$.
  2. Kỹ thuật **Early Stopping** đã can thiệp tự động ngắt huấn luyện tại Epoch 28 khi Validation Loss không còn cải thiện trong 10 epochs liên tiếp, đồng thời khôi phục lại bộ trọng số tối ưu nhất (Best Val Loss = 0.02177).

---

### Câu 6: Sự khác biệt bản chất giữa Machine Learning (XGBoost) và Deep Learning (LSTM) trong bài toán này?

| Tiêu chí | Machine Learning (XGBoost) | Deep Learning (LSTM) |
| :--- | :--- | :--- |
| **Biểu diễn dữ liệu đầu vào** | **Vector phẳng 2D:** Buộc phải làm phẳng thời gian thành 96 cột rời rạc ($4 \text{ biến} \times 24 \text{ bước}$). Mô hình không tự hiểu được trật tự trước sau nếu không gán nhãn tên cột. | **Tensor 3D $(N, 24, 4)$:** Bảo toàn nguyên vẹn tính tuần tự thời gian và cấu trúc dòng chảy dữ liệu liên tục. |
| **Kỹ nghệ đặc trưng** | Đòi hỏi con người phải can thiệp thủ công: tạo thêm `rolling mean`, `rolling std`, `trend` để mô hình bắt được xu hướng. | **Học đặc trưng tự động (End-to-End Representation Learning):** Mạng tự học cách trích xuất xu hướng và bộ nhớ qua các cell state. |
| **Cơ chế mô hình hóa** | Phân vùng không gian bằng các siêu phẳng trực giao thông qua hàm phân nhánh cây quyết định (Rule-based Partitioning). | Sử dụng các hàm kích hoạt phi tuyến tính trơn ($\tanh, \sigma$) và các phép nhân ma trận trọng số hồi quy. |
| **Tốc độ huấn luyện** | Cực kỳ nhanh (~2.8s cho 8,700 mẫu), tối ưu hóa tốt trên CPU đa nhân. | Tốn nhiều tài nguyên tính toán hơn (~16.4s), cần huấn luyện lặp qua nhiều epochs bằng giải thuật BPTT. |
| **Khả năng diễn giải** | Dễ diễn giải: Xem được Feature Importance (Gain, Weight, Cover) của từng đặc trưng cụ thể. | Dạng hộp đen (Black-box): Rất khó để giải thích trực tiếp trạng thái của từng nơ-ron ảnh hưởng thế nào đến kết quả. |

---

## 6. BÀI TẬP NÂNG CAO: DỰ BÁO LƯU LƯỢNG GIAO THÔNG (TRAFFIC FORECASTING)

### 6.1. Bối cảnh bài toán & Tập dữ liệu METR-LA / PEMS-BAY
- **Đặc thù bài toán:** Dự báo tốc độ xe chạy trên mạng lưới 207 trạm cảm biến của đường cao tốc Los Angeles tại các chân trời tương lai:
  $$[X_{t-12}, \dots, X_t] \Longrightarrow X_{t+3}, X_{t+6}, X_{t+12} \quad (15, 30, 60 \text{ phút phía trước})$$
- **Tính chất đa chiều phức tạp:**
  1. **Phụ thuộc không gian (Spatial Dependency):** Sự ùn tắc tại một trạm trên đường cao tốc sẽ lan truyền ngược hoặc xuôi dòng đến các trạm lân cận trên cùng tuyến đường (đồ thị có hướng phi Euclid).
  2. **Phụ thuộc thời gian (Temporal Dependency):** Vận tốc xe thay đổi liên tục theo giờ cao điểm sáng/chiều và ngày làm việc.

### 6.2. Kiến trúc DCRNN (Diffusion Convolutional Recurrent Neural Network)
- **Tích chập khuếch tán trên đồ thị (Diffusion Graph Convolution):**
  - Mô phỏng quá trình di chuyển của dòng phương tiện bằng chuỗi Markov ngẫu nhiên trên đồ thị có hướng:
    $$P = D_O^{-1} W \quad \text{(quá trình di chuyển xuôi)}, \quad P^T = D_I^{-1} W^T \quad \text{(quá trình di chuyển ngược)}$$
  - Phép tích chập khuếch tán bậc $K$:
    $$x \star_{\mathcal{G}} f_\Theta = \sum_{k=0}^{K-1} \left( \theta_{k,1} (D_O^{-1} W)^k + \theta_{k,2} (D_I^{-1} W^T)^k \right) x$$
- **Cơ chế Hồi quy Tuần tự (Sequence-to-Sequence Architecture):**
  - Thay thế các phép nhân ma trận thông thường trong mạng GRU bằng tích chập khuếch tán DCGRU.
  - Sử dụng kiến trúc **Encoder - Decoder** kết hợp kỹ thuật **Scheduled Sampling** (chọn ngẫu nhiên giữa giá trị thực tế và giá trị dự báo trong quá trình train) để triệt tiêu hiện tượng tích lũy sai số khi dự báo đa bước thời gian dài (60 phút).

---

## 7. DANH MỤC ARTIFACTS ĐÃ TẠO

Toàn bộ các artifacts được lưu trữ chuẩn mực theo cấu trúc:
1. `lab4/artifacts/figures/lab4_01_eda_load_timeseries.png`: Chuỗi thời gian điện năng tiêu thụ thực tế năm 2017.
2. `lab4/artifacts/figures/lab4_02_eda_load_vs_temperature.png`: Biểu đồ quan hệ phi tuyến chữ U giữa Load và Temperature.
3. `lab4/artifacts/figures/lab4_03_eda_correlation_matrix.png`: Ma trận tương quan nhiệt động giữa 4 biến.
4. `lab4/artifacts/figures/lab4_04_model_benchmark.png`: Biểu đồ cột so sánh RMSE, MAE, MAPE của 3 mô hình.
5. `lab4/artifacts/figures/lab4_05_lstm_loss_curve.png`: Đường cong Train MSE Loss vs Val MSE Loss của LSTM qua các epochs.
6. `lab4/artifacts/figures/lab4_06_xgboost_feature_importance.png`: Thứ hạng đóng góp của các biến thời tiết và biến trễ trong XGBoost.
7. `lab4/artifacts/figures/lab4_07_forecast_comparison.png`: So sánh chi tiết dự báo của VAR, XGBoost, LSTM trên 1 tuần mẫu của tập Test.
8. `lab4/artifacts/metrics/benchmark_metrics.json`: Bảng số liệu định lượng chi tiết dạng JSON.
9. `lab4/artifacts/models/`: Chứa các checkpoint `var_model.joblib`, `xgboost_model.joblib`, `lstm_model.pt`, `data_scaler.joblib`.
