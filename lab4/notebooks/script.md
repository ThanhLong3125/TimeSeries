# KỊCH BẢN THUYẾT TRÌNH LAB 4 (10 – 15 PHÚT): DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (VAR vs XGBOOST vs LSTM)

**Tài liệu trình chiếu:** [thuc_hanh_du_bao_da_bien_var_xgboost_lstm.ipynb](file:///home/thanhlong/Documents/TimeSeries/lab4/notebooks/thuc_hanh_du_bao_da_bien_var_xgboost_lstm.ipynb)  
**Thời lượng thiết kế:** 10 – 15 phút (Notebook đã được tinh gọn tối đa để lướt mượt trên màn hình, toàn bộ phần nhận xét & giải thích chuyên sâu nằm tại kịch bản này).

---

## ⏱️ PHÂN BỔ THỜI GIAN (TIMELINE 12 PHÚT CHUẨN)
- **00:00 – 02:30 (2.5 phút):** Mục 1 & 2 — Mở đầu, Giải thích Fields dữ liệu & Đánh giá chất lượng dữ liệu.
- **02:30 – 05:00 (2.5 phút):** Mục 3 — EDA & Giải thích chi tiết 3 nhóm biểu đồ (Seasonality, U-shape, Correlation).
- **05:00 – 07:00 (2.0 phút):** Mục 4 & 5 — Preprocessing & Tại sao lại chọn các đặc trưng trong Feature Engineering?
- **07:00 – 10:00 (3.0 phút):** Mục 6, 7, 8 — Khám phá quá trình huấn luyện & Biểu đồ chẩn đoán của **cả 3 mô hình (VAR, XGBoost, LSTM)**.
- **10:00 – 12:30 (2.5 phút):** Mục 9 & 10 — Đánh giá so chuẩn (RMSE, MAE, MAPE), 6 câu hỏi vấn đáp & Traffic Forecasting (DCRNN).

---

## 🎙️ PHẦN 1: MỞ ĐẦU, GIẢI THÍCH FIELDS & ĐÁNH GIÁ DỮ LIỆU (00:00 – 02:30)
*(Trình chiếu **Tiêu đề**, **Mục 1** và **Mục 2** trên Notebook)*

### 1.1. Lời thoại Mở đầu & Giải thích Fields dữ liệu
> "Kính thưa Thầy/Cô và các bạn, hôm nay em xin trình bày bài thực hành **Lab 4: Dự báo Chuỗi Thời Gian Đa Biến (Multivariate Time Series Forecasting)** kết hợp Điện năng Phụ tải và Dữ liệu Thời tiết.
>
> Khác với dự báo đơn biến chỉ nhìn vào lịch sử của 1 biến, bài toán Lab 4 quan sát đồng thời một vector 4 biến tại mỗi giờ $t$: $X_t = [load_t, temp_t, humidity_t, wind\_speed_t]$, sử dụng cửa sổ lịch sử **24 giờ gần nhất** ($X_{t-24:t}$) để dự báo điện năng tiêu thụ ở giờ kế tiếp ($load_{t+1}$).
>
> *(Chỉ vào bảng `df.head(3)` ở **Mục 2** để giải thích các trường dữ liệu)*  
> Cụ thể ý nghĩa vật lý và vai trò của 5 trường dữ liệu (fields) trong bài toán như sau:
> 1. **`Datetime` (Index):** Mốc thời gian ghi nhận theo từng giờ liên tục (`YYYY-MM-DD HH:00:00`), định hình tính chu kỳ ngày/đêm 24h và mùa vụ.
> 2. **`load` (Đơn vị: $\text{kW}$ — Biến mục tiêu nội sinh / Target):** Tổng công suất điện năng tiêu thụ phụ tải của khu vực. Biến này phản ánh tổng hợp nhịp sinh hoạt hộ gia đình, sản xuất công nghiệp, chiếu sáng và thiết bị làm mát/sưởi ấm.
> 3. **`temperature` (Đơn vị: $^\circ\text{C}$ — Biến thời tiết chủ đạo):** Nhiệt độ không khí ngoài trời. Đây là yếu tố khí tượng chi phối mạnh nhất đến phụ tải thông qua nhu cầu bật máy điều hòa làm mát (HVAC) khi trời nóng hoặc sưởi ấm khi trời lạnh.
> 4. **`humidity` (Đơn vị: $\%$ — Biến thời tiết tương tác):** Độ ẩm tương đối. Độ ẩm kết hợp với nhiệt độ tạo nên **chỉ số nhiệt cảm nhận (Heat Index / Độ oi bức)**: khi trời vừa nóng vừa ẩm, máy điều hòa phải tiêu tốn thêm công suất lớn để ngưng tụ hơi nước (khử ẩm).
> 5. **`wind_speed` (Đơn vị: $\text{m/s}$ — Biến thời tiết điều hòa):** Tốc độ gió tự nhiên. Gió mạnh giúp tăng đối lưu tản nhiệt trên bề mặt tòa nhà, làm dịu bớt nhiệt độ đô thị."

### 1.2. Nhận xét & Đánh giá Chất lượng Dữ liệu (Bảng `df.describe()`)
> "*(Chỉ vào output của **Code Cell 2**)*  
> Từ kết quả kiểm tra và bảng thống kê mô tả `describe()`, chúng ta đánh giá dữ liệu qua 3 điểm chính:
> - **Thứ nhất — Độ sạch và tính liên tục (Missing values):** Tập dữ liệu gồm đúng **$8,736$ dòng $\times 4$ cột** (tương ứng 364 ngày $\times$ 24 giờ liên tục trong năm 2017). Cả 4 cột đều có **`Missing values = 0`**, không bị đứt quãng giờ nào, đảm bảo trượt cửa sổ 24h hoàn toàn chính xác.
> - **Thứ hai — Đánh giá phân phối & Outliers:**
>   - Biến **`load`** có giá trị trung bình `mean = 32,344.97 kW` trùng khít với trung vị `50% (median) = 32,342.30 kW`, chứng tỏ phân phối rất cân xứng, không bị lệch bởi nhiễu dị thường. Biên độ dao động trải từ đáy đêm mùa lạnh (`14,329.11 kW`) đến đỉnh cao điểm mùa hè (`51,844.26 kW`), độ lệch chuẩn `std = 7,068.92 kW`.
>   - Biến **`temperature`** có trung bình `18.81°C` (trải từ `3.60°C` đến `39.70°C`), 50% thời gian trong năm nằm ở vùng mát mẻ từ `14.40°C` (`25%`) đến `22.87°C` (`75%`).
>   - Biến **`humidity`** trung bình đạt `68.26%` (từ `12.71%` đến `94.75%`).
>   - Biến **`wind_speed`** có trung bình `1.96 m/s` nhưng trung vị chỉ `0.09 m/s` và phân vị `75%` là `4.92 m/s`, phản ánh đặc trưng phân phối 2 trạng thái: những giờ lặng gió xen kẽ những đợt gió mạnh lưu thông khí quyển.
> - **Thứ ba — Sự lệch thang đo:** Vì `load` nằm ở hàng chục nghìn $\text{kW}$ trong khi `wind_speed` chỉ ở mức vài $\text{m/s}$, việc chuẩn hóa bằng `StandardScaler` ở bước sau là bắt buộc."

---

## 🎙️ PHẦN 2: KHÁM PHÁ DỮ LIỆU (EDA) & GIẢI THÍCH BIỂU ĐỒ CHI TIẾT (02:30 – 05:00)
*(Trình chiếu **Mục 3 — Code Cell 3** trên Notebook)*

### 2.1. Giải thích Biểu đồ 1a & 1b: Chuỗi thời gian `load` toàn năm & Chu kỳ ngày 24h
> "*(Chỉ vào **Biểu đồ 1a bên trái** và **Biểu đồ 1b bên phải**)*
> - **Trên Biểu đồ 1a (Chuỗi thời gian `load` năm 2017):** Ta thấy rõ **tính mùa vụ (Seasonality)** theo năm. Trong những tháng mùa xuân và mùa thu, phụ tải dao động quanh đường trung bình nét đứt đỏ (`32,345 kW`). Nhưng khi bước vào cao điểm mùa hè (tháng 7 – tháng 8), phụ tải dâng cao vọt lên vùng `45,000 – 51,800 kW` do nhu cầu làm mát tăng đột biến.
> - **Trên Biểu đồ 1b (Chu kỳ phụ tải trung bình 24 giờ trong ngày - Diurnal Seasonality):** Đường cong thể hiện quy luật sinh hoạt 24 giờ cực kỳ rõ nét:
>   - Phụ tải giảm sâu chạm **đáy vào lúc 04:00 sáng** ($\sim 22,500\text{ kW}$) khi toàn thành phố nghỉ ngơi.
>   - Từ **06:00 sáng**, phụ tải tăng dốc đứng khi các nhà máy, công sở hoạt động và đạt **đỉnh thứ nhất vào buổi trưa (12:00 – 14:00, $\sim 36,000\text{ kW}$)** khi nắng nóng đạt đỉnh.
>   - Sau nhịp chùng nhẹ lúc 16:00 chiều, phụ tải vọt lên **đỉnh cao nhất trong ngày vào buổi tối (19:00 – 21:00, $\sim 40,500\text{ kW}$)** do sự cộng hưởng giữa chiếu sáng đô thị, sinh hoạt gia đình và máy lạnh phòng ngủ.  
>   $\implies$ **Quy luật lặp lại đúng mỗi 24 giờ này chính là căn cứ trực tiếp để ta chọn cửa sổ trễ $L = 24$ giờ!**"

### 2.2. Giải thích Biểu đồ 2: Quan hệ `Load vs Temperature` (Đường cong chữ U)
> "*(Chỉ vào **Biểu đồ 2 bên trái**)*  
> Đây là đồ thị phân tán giữa Nhiệt độ (trục hoành), Phụ tải điện (trục tung) và màu sắc là Độ ẩm:
> - Đường hồi quy bậc 2 màu đỏ cho thấy mối quan hệ giữa nhiệt độ và điện năng tiêu thụ có dạng **phi tuyến hình chữ U (U-shape)**:
>   - **Đáy chữ U ($14^\circ\text{C} - 18^\circ\text{C}$):** Là vùng nhiệt độ dễ chịu (Comfort Zone), nhu cầu làm mát hay sưởi ấm đều tối thiểu nên phụ tải ở mức thấp nhất.
>   - **Nhánh phải ($T > 24^\circ\text{C} \rightarrow 40^\circ\text{C}$):** Khi nhiệt độ vượt ngưỡng $24^\circ\text{C}$, phụ tải tăng vọt theo hàm bậc hai vì hàng loạt hệ thống điều hòa không khí bị kích hoạt đồng loạt.
>   - **Nhánh trái ($T < 12^\circ\text{C}$):** Khi trời lạnh về đêm và mùa đông, nhu cầu dùng bình nước nóng và thiết bị sưởi giữ cho phụ tải không giảm thêm mà uốn cong đi ngang/tăng nhẹ.
> - Đồng thời, thang màu `Humidity` cho thấy ở vùng nhiệt độ $18^\circ\text{C} - 25^\circ\text{C}$, các điểm có độ ẩm cao (màu vàng/xanh lục sáng $75\% - 95\%$) tập trung nhiều ở tầng phụ tải cao hơn do cảm giác oi bức và trùng với khung giờ cao điểm tối."

### 2.3. Giải thích Biểu đồ 3: Ma trận tương quan (`Correlation Matrix`)
> "*(Chỉ vào **Biểu đồ 3 bên phải**)*  
> Nhìn vào ma trận tương quan tuyến tính Pearson giữa 4 biến:
> - **`load` vs `temperature` đạt $r = +0.444$:** Là biến thời tiết có tương quan dương mạnh nhất với tải điện. Sở dĩ hệ số tuyến tính Pearson chỉ đạt $+0.444$ là vì mối quan hệ thực tế có độ cong chữ U phi tuyến và có độ trễ pha giữa đỉnh nhiệt độ (14h trưa) với đỉnh phụ tải tối (20h tối).
> - **`load` vs `humidity` đạt $r = -0.291$:** Tương quan âm vừa phải. Lý do là ban ngày khi trời nắng nóng (tiêu thụ điện cao) thì độ ẩm không khí giảm xuống ($r(\text{temp}, \text{humidity}) = -0.460$), còn lúc 4h sáng (điện chạm đáy) thì trời lạnh làm độ ẩm tăng cao.
> - **`load` vs `wind_speed` đạt $r = +0.170$:** Tương quan dương nhẹ."

---

## 🎙️ PHẦN 3: PREPROCESSING & FEATURE ENGINEERING — TẠI SAO CHỌN CÁC FEATURE ĐÓ? (05:00 – 07:00)
*(Trình chiếu **Mục 4** và **Mục 5** trên Notebook)*

### 3.1. Chia tập dữ liệu & Chuẩn hóa (Mục 4)
> "*(Chỉ vào **Mục 4**)*  
> Ở bước Tiền xử lý, dữ liệu được chia theo **thứ tự thời gian nghiêm ngặt (`shuffle=False`)** để tuyệt đối không rò rỉ dữ liệu tương lai về quá khứ:
> - **Train (70%):** $6,115$ giờ đầu (từ tháng 1 đến giữa tháng 9/2017).
> - **Validation (15%):** $1,310$ giờ tiếp theo (giữa tháng 9 đến đầu tháng 11/2017) — dùng để kiểm soát **Early Stopping**.
> - **Test (15%):** $1,311$ giờ cuối năm — dùng để đánh giá độc lập.
> - Bộ `StandardScaler` chỉ được `.fit()` trên tập Train rồi `.transform()` cho cả 3 tập để giúp mạng LSTM hội tụ ổn định và tránh thiên lệch đơn vị đo."

### 3.2. Feature Engineering: Tại sao lại chọn các đặc trưng này? (Mục 5)
> "*(Chỉ vào **Mục 5**)*  
> Tại sao ở bước Feature Engineering chúng ta lại thiết kế **96 biến trễ (`lag_1`..`lag_24`)** cộng với **3 biến Rolling/Trend** cho XGBoost và **Tensor 3D $(N, 24, 4)$** cho LSTM? Có 3 lý do cốt lõi:
>
> 1. **Tại sao chọn cửa sổ trễ $L = 24$ giờ ($24 \times 4 = 96$ lag features)?**
>    - Vì chuỗi phụ tải theo giờ có chu kỳ ngày-đêm 24h cực mạnh. Cửa sổ 24 giờ giúp mô hình luôn nhìn thấy trọn vẹn 1 vòng quay ngày-đêm trước đó, đặc biệt là 2 mốc quan trọng nhất: **`load_lag_1`** (quán tính phụ tải cách đó 1 giờ, tương quan $r = 0.919$) và **`load_lag_24`** (phụ tải cùng khung giờ ngày hôm qua, tương quan cao nhất $r = 0.958$).
>    - Với 3 biến thời tiết, 24 bước trễ phản ánh **quán tính tích nhiệt của tòa nhà** suốt cả ngày hôm trước.
> 2. **Tại sao chọn thêm 3 đặc trưng `rolling_mean_6h`, `rolling_std_24h` và `load_trend` cho XGBoost?**
>    - Vì cây quyết định trong XGBoost xem 96 cột trễ là rời rạc, không tự tính trung bình hay độ dốc hiệu quả như cổng nhớ của LSTM. Do đó ta tạo thêm:
>      - **`load_rolling_mean_6h`:** Trung bình trượt 6 giờ gần nhất (1 ca/buổi sinh hoạt), giúp làm mượt nhiễu ngắn hạn và xác định mặt bằng tải hiện tại.
>      - **`load_rolling_std_24h`:** Độ lệch chuẩn 24 giờ qua, đo lường **độ biến động (volatility)** đỉnh-đáy trong ngày (phân biệt ngày nắng nóng/đi làm có biên độ lớn với ngày mát/cuối tuần có biên độ phẳng).
>      - **`load_trend` ($load_{t-1} - load_{t-24}$):** Đo lường **xung lượng tăng/giảm tải so với cùng giờ hôm qua** (dương là hôm nay tiêu thụ cao hơn hôm qua, âm là đang hạ tải).
>    - *(Lưu ý kỹ thuật: Cả 3 biến này đều tính từ `load.shift(1)` trở về trước để chống Data Leakage)*.
> 3. **Tại sao LSTM dùng Tensor 3D `(6091, 24, 4)`?**
>    - Để giữ nguyên cấu trúc tuần tự `(Mẫu, 24 bước thời gian, 4 biến)` cho phép trạng thái ẩn $h_t$ truyền dọc theo dòng thời gian mà không bị phá vỡ bởi phép làm phẳng."

---

## 🎙️ PHẦN 4: TỔNG QUÁT Ý NGHĨA CỦA 3 MÔ HÌNH ĐƯỢC CHỌN & KHÁM PHÁ QUÁ TRÌNH HUẤN LUYỆN (07:00 – 10:30)
*(Trình chiếu lần lượt **Mục 6 (VAR)**, **Mục 7 (XGBoost)** và **Mục 8 (LSTM)** trên Notebook)*

### 4.0. Tổng quát Ý nghĩa & Lý do lựa chọn bộ 3 Mô hình (VAR vs XGBoost vs LSTM)
> "Trước khi đi vào từng mô hình cụ thể ở Mục 6, 7 và 8, em xin trình bày **ý nghĩa tổng quát tại sao bài thực hành này lại chọn đúng bộ 3 mô hình VAR, XGBoost và LSTM**:
>
> Bộ 3 mô hình này không được chọn ngẫu nhiên, mà chúng đại diện cho **3 trường phái (3 hệ tư tưởng) tiến hóa kế tiếp nhau** trong lịch sử dự báo chuỗi thời gian đa biến:
>
> 1. **Trường phái Thống kê Kinh tế lượng — VAR (Vector Autoregression):**
>    - **Ý nghĩa cốt lõi:** Là mô hình **chuẩn cơ sở tuyến tính (Linear Statistical Baseline)**. VAR không phân biệt cứng nhắc đâu là biến đầu vào, đâu là biến đầu ra, mà coi cả 4 biến $[load, temp, humidity, wind]$ là một **hệ thống tương tác hai chiều đồng thời**, nơi mỗi biến ở hiện tại là tổ hợp tuyến tính của chính nó và các biến khác trong $p$ giờ quá khứ: $X_t = c + \sum_{i=1}^p A_i X_{t-i} + \varepsilon_t$.
>    - **Vai trò trong bài toán:** Dùng để kiểm chứng xem **chỉ với giả định tương quan tuyến tính và tính dừng toán học**, mô hình thống kê cổ điển có thể giải quyết bài toán phụ tải điện tốt đến mức nào.
>
> 2. **Trường phái Học máy Dạng bảng — XGBoost (Extreme Gradient Boosting Trees):**
>    - **Ý nghĩa cốt lõi:** Đại diện cho đỉnh cao của **Machine Learning phi tuyến dựa trên cây quyết định (Tree-based Ensemble)**. XGBoost chuyển hóa bài toán chuỗi thời gian thành bài toán **Hồi quy có giám sát trên bảng dữ liệu phẳng 2D** ($99$ đặc trưng), sử dụng hàng trăm cây quyết định nối tiếp nhau để học phần dư sai số.
>    - **Vai trò trong bài toán:** Khắc phục điểm yếu tuyến tính của VAR bằng cách tự động tìm ra các **ngưỡng chia phi tuyến (Threshold Rules)** — ví dụ ngưỡng nhiệt độ bật điều hòa — và học sự **tương tác chéo đa biến** (nhiệt độ $\times$ độ ẩm $\times$ xu hướng tải) mà không cần giả định dữ liệu phải dừng.
>
> 3. **Trường phái Học sâu Tuần tự — LSTM (Long Short-Term Memory Neural Network):**
>    - **Ý nghĩa cốt lõi:** Đại diện cho **Deep Learning biểu diễn chuỗi tuần tự (Sequential Representation Learning)**. Thay vì đập phẳng thời gian thành các cột rời rạc như XGBoost, LSTM giữ nguyên khối **Tensor 3D $(N, 24, 4)$** và dùng cơ chế **3 cổng nhớ (Forget, Input, Output Gate)** cùng đường truyền trạng thái tế bào $c_t$ để cho thông tin chảy liên tục từ giờ $t-24$ đến giờ $t-1$.
>    - **Vai trò trong bài toán:** Tự động trích xuất đặc trưng động học và hình thái chu kỳ ngày-đêm (End-to-End) trực tiếp từ dữ liệu thô mà không cần con người phải thiết kế các biến `rolling mean`, `std` hay `trend` thủ công.
>
> | Trường phái | Mô hình đại diện | Bản chất toán học & Cơ chế học | Yêu cầu dữ liệu đầu vào | Điểm mạnh đặc trưng |
> | :--- | :---: | :--- | :---: | :--- |
> | **1. Thống kê (Statistical)** | **VAR** | Hệ phương trình tự hồi quy tuyến tính đa biến trên chuỗi sai phân dừng | Ma trận chuỗi dừng 4 biến $\Delta X_t \in \mathbb{R}^{T \times 4}$ | Cực nhanh, tối ưu cho dự báo tuyến tính 1 bước ngắn hạn ($t+1$) |
> | **2. Học máy (Machine Learning)** | **XGBoost** | Quần thể cây quyết định tăng cường Gradient Boosting cắt ngưỡng phi tuyến | Bảng phẳng 2D $(N, 99)$ kèm Feature Engineering thủ công | Bắt đỉnh nhọn & ngưỡng thời tiết phi tuyến xuất sắc, giải thích được Feature Importance |
> | **3. Học sâu (Deep Learning)** | **LSTM** | Mạng nơ-ron hồi quy có cổng nhớ lưu giữ trạng thái dài-ngắn hạn qua thời gian | Khối Tensor 3D $(N, 24, 4)$ giữ nguyên thứ tự thời gian | Học tự động (End-to-End) quy luật sóng chu kỳ mượt mà, dễ mở rộng lên bài toán đồ thị lớn |
>
> Sau đây, em xin đi vào quá trình huấn luyện và biểu đồ chẩn đoán của từng mô hình."

### 4.1. Quá trình của Mô hình 1 – VAR (Statistical Baseline — Mục 6)
> "*(Chỉ vào bảng ADF và 2 biểu đồ của **Mục 6**)*  
> Trước khi đến bước so sánh cuối cùng, chúng ta khảo sát quá trình hoạt động của từng mô hình. Đầu tiên là **Mô hình thống kê VAR**:
> 1. **Kiểm định dừng ADF (Bảng ADF):** Trên chuỗi gốc ($d=0$), trong khi `load`, `humidity`, `wind_speed` đã dừng ($p < 0.05$) thì biến **`temperature` có $\text{ADF} = -2.474$, $p\text{-value} = 0.122 > 0.05 \implies$ Không dừng**. Do đó, ta lấy **sai phân bậc 1 ($d=1$)** cho toàn bộ 4 biến. Sau khi lấy sai phân $d=1$, cả 4 biến đều dừng tuyệt đối ($p \approx 0$).
> 2. **Biểu đồ chọn bậc trễ $p$ (Biểu đồ bên trái):** Đường cong `AIC` giảm nhanh ở các giờ đầu, đi ngang ở giữa rồi **sụt giảm sâu đạt cực tiểu tại đúng $p = 24$ giờ**. Điều này chứng minh về mặt toán học: hệ phương trình VAR cần nhìn đủ 24 giờ để bắt được chu kỳ ngày.
> 3. **Biểu đồ dự báo của VAR (Biểu đồ bên phải):** Nhờ dự báo cuốn chiếu 1 bước từ chuỗi sai phân ($\hat{Y}_t = Y_{t-1} + \Delta\hat{Y}_t$), đường dự báo màu xanh của VAR bám rất sát đường thực tế."

### 4.2. Quá trình của Mô hình 2 – XGBoost (Machine Learning — Mục 7)
> "*(Chỉ vào 2 biểu đồ của **Mục 7**)*  
> Tiếp theo là quá trình học của **Mô hình 2: XGBoost Regressor** trên 99 đặc trưng:
> 1. **Đường cong học tập & Early Stopping (Biểu đồ bên trái):** Cả `Train RMSE` và `Validation RMSE` đều giảm cực nhanh trong 100 cây đầu tiên. Đến **vòng lặp thứ 252 (`Best Iter = 252` - vạch đỏ)**, `Validation RMSE` đạt cực tiểu là **$0.1316$**. Sau 30 vòng tiếp theo không cải thiện thêm, cơ chế **Early Stopping** tự động dừng huấn luyện và chốt mô hình ở cây 252 để chống overfitting.
> 2. **Biểu đồ Top 10 Feature Importance (Biểu đồ bên phải):** Cho thấy rõ XGBoost đã học như thế nào:
>    - Hai biến quyết định khung phụ tải lớn nhất là **`load_lag_1`** (giờ ngay trước đó) và **`load_lag_24`** (cùng giờ hôm qua).
>    - Các biến kỹ nghệ như **`load_trend`** và **`load_rolling_mean_6h`** cùng với các biến trễ nhiệt độ, độ ẩm lọt vào nhóm đóng góp cao nhất, giúp mô hình điều chỉnh chính xác biên độ đỉnh phụ tải theo thời tiết."

### 4.3. Quá trình của Mô hình 3 – LSTM (Deep Learning — Mục 8)
> "*(Chỉ vào 2 biểu đồ của **Mục 8**)*  
> Thứ ba là quá trình huấn luyện của **Mô hình 3: Mạng học sâu LSTM** với kiến trúc $\text{LSTM}(64) \rightarrow \text{LSTM}(32) \rightarrow \text{Dense}(1)$:
> 1. **Đường cong hội tụ Train vs Val MSE Loss (Biểu đồ bên trái):** Qua từng Epoch, `Train MSE Loss` (xanh) và `Validation MSE Loss` (cam) giảm đồng pha từ $\sim 0.36$ xuống vùng $\sim 0.022$. Tại **Epoch 34 (vạch đỏ)**, Validation Loss đạt mức thấp nhất là **$0.0225$**, và **Early Stopping (`patience=10`)** ngắt tại Epoch 44 để khôi phục trọng số Epoch 34. Hai đường Train và Val bám sát nhau, khẳng định mạng LSTM **không hề bị overfitting**.
> 2. **Biểu đồ dự báo của LSTM (Biểu đồ bên phải):** Đường dự báo màu đỏ của LSTM tái hiện rất mượt mà và chính xác nhịp sóng ngày-đêm 24h của phụ tải điện trên tập Test."

---

## 🎙️ PHẦN 5: ĐÁNH GIÁ SO CHUẨN 3 MÔ HÌNH & CÂU HỎI PHÂN TÍCH (10:00 – 12:30)
*(Trình chiếu **Mục 9** và **Mục 10** trên Notebook)*

### 5.1. Đánh giá So chuẩn Hiệu năng 3 Mô hình (Mục 9)
> "*(Chỉ vào **Bảng tổng hợp**, **Biểu đồ Bar Chart 3 chỉ số** và **Biểu đồ So sánh 168 giờ** ở Mục 9)*  
> Sau khi huấn luyện cả 3 mô hình, ta đặt chúng lên cùng một tập Test ($1,287$ giờ) để đánh giá qua **RMSE**, **MAE** và **MAPE**:
> - **VAR (Statistical):** $\text{RMSE} = \mathbf{641.15\text{ kW}}$, $\text{MAE} = \mathbf{443.01\text{ kW}}$, $\text{MAPE} = \mathbf{1.57\%}$.
> - **XGBoost (ML):** $\text{RMSE} = \mathbf{970.12\text{ kW}}$, $\text{MAE} = \mathbf{683.33\text{ kW}}$, $\text{MAPE} = \mathbf{2.40\%}$.
> - **LSTM (Deep Learning):** $\text{RMSE} = \mathbf{1,305.30\text{ kW}}$, $\text{MAE} = \mathbf{926.27\text{ kW}}$, $\text{MAPE} = \mathbf{3.44\%}$.
>
> **Nhận xét cốt lõi:**
> 1. Cả 3 mô hình đều đạt sai số rất thấp ($\text{MAPE} < 3.5\%$, vượt chuẩn vận hành ngành điện $<5\%$) và bám sát đường thực tế màu đen trên biểu đồ 168 giờ.
> 2. **Tại sao VAR đạt sai số thấp nhất ở đây?** Vì đây là bài toán dự báo cuốn chiếu **1 bước ngắn hạn ($t+1\text{ giờ}$)** và VAR dự báo trên chuỗi sai phân bậc 1 ($\hat{Y}_{t+1} = Y_t + \Delta\hat{Y}_{t+1}$), tức là nó lấy trực tiếp giá trị thực tế vừa xảy ra ở giờ $t$ làm điểm tựa rồi cộng thêm lượng biến thiên tuyến tính. Tuy nhiên, nếu phải dự báo xa nhiều bước (24h – 72h tới mà không có $Y_t$ thực tế mỗi giờ) hoặc gặp biến động phi tuyến gắt, VAR sẽ mất lợi thế này.
> 3. **Giữa XGBoost và LSTM:** Trên dữ liệu bảng cỡ vừa ($\sim 6,000$ dòng) đã có Feature Engineering tốt (99 biến), cây quyết định **XGBoost** bắt đỉnh nhọn sắc bén hơn và huấn luyện cực nhanh ($\sim 2\text{ giây}$). Trong khi đó, **LSTM** học hoàn toàn tự động từ Tensor 3D thô mà không cần tạo biến `rolling/trend` thủ công, cho đường dự báo mượt mà."

### 5.2. Tóm tắt 6 Câu hỏi Vấn đáp & Bài tập nâng cao Traffic Forecasting (Mục 10)
> "*(Chỉ vào **Mục 10**)*  
> Cuối cùng, em xin chốt nhanh 6 câu hỏi phân tích và bài toán mở rộng:
> 1. **Biến thời tiết ảnh hưởng mạnh nhất:** Là **Nhiệt độ (`temperature`, $r = +0.444$)** theo hàm phi tuyến chữ U (tăng tải làm mát mạnh khi $T > 24^\circ\text{C}$).
> 2. **Khi nào VAR thất bại:** Khi quan hệ giữa các biến là **phi tuyến mạnh**, khi nhiều biến gây **bùng nổ tham số bậc hai ($K^2 \cdot p$)** dẫn đến overfitting, hoặc khi chuỗi **không dừng**.
> 3. **XGBoost học được gì từ thời tiết:** Học được các **ngưỡng chia nhiệt độ (Threshold splits)** bật máy lạnh và **tương tác chéo** giữa Nhiệt độ + Độ ẩm cao (chỉ số oi bức Heat Index).
> 4. **LSTM học được chu kỳ nào:** Học trọn vẹn **chu kỳ ngày-đêm 24h (Diurnal cycle)** từ Tensor $(N, 24, 4)$ và nhận diện mức nền chu kỳ tuần qua bộ nhớ tế bào $c_t$.
> 5. **LSTM có bị overfitting không:** **Không**, nhờ mạng gọn ($64 \rightarrow 32 \rightarrow 1$) và **Early Stopping** (Val Loss giảm đồng pha sát Train Loss).
> 6. **Khác biệt giữa ML (XGBoost) và DL (LSTM):** XGBoost cần **làm phẳng thời gian thành vector 2D (99 biến)** và tạo đặc trưng thủ công, nhưng chạy nhanh và có Feature Importance minh bạch; còn LSTM giữ nguyên **Tensor 3D tuần tự**, tự học đặc trưng qua cổng nhớ nhưng tốn chi phí tính toán hơn và là hộp đen.
> - **Với bài tập nâng cao Traffic Forecasting (`METR-LA` / `PEMS-BAY` — 207 trạm cảm biến giao thông):** Để dự báo đa bước $15, 30, 60$ phút tới, mô hình **DCRNN** kết hợp **Tích chập khuếch tán trên đồ thị có hướng (Diffusion Graph Convolution)** để bắt sự lan truyền kẹt xe không gian giữa các trạm và **GRU Seq2Seq + Scheduled Sampling** để mô hình hóa chiều thời gian đa bước.
>
> Phần trình bày của em xin kết thúc tại đây. Em cảm ơn Thầy/Cô và các bạn đã lắng nghe ạ!"
