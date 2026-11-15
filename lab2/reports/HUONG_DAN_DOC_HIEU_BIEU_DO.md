# 📘 CẨM NANG HƯỚNG DẪN ĐỌC HIỂU & PHÂN TÍCH BIỂU ĐỒ CHUỖI THỜI GIAN (LAB 2)
## Diễn giải Chuyên sâu 9 Biểu đồ Thực nghiệm (Dataset 1: a10 & Dataset 2: AirPassengers)

Tài liệu này được biên soạn nhằm giúp bạn **nắm vững bản chất lý thuyết**, **biết cách quan sát từng chi tiết trên 9 đồ thị thực nghiệm**, **giải thích tự tin và thuyết phục** trong mọi bài kiểm tra, đồ án và thuyết trình môn Phân tích Chuỗi thời gian.

---

# 📑 MỤC LỤC
1. [Bộ đồ thị Dataset 1 (Monthly Drug Sales - a10)](#1-bộ-đồ-thị-dataset-1-monthly-drug-sales---a10)
   - [Đồ thị 1] ds1_01_eda_raw_series.png: Chuỗi gốc a10 & Các mốc thống kê Mean, Min, Max
   - [Đồ thị 2] ds1_02_differenced_series.png: Chuỗi sau sai phân bậc 1 quanh trục 0
   - [Đồ thị 3] ds1_03_acf_pacf.png: Biểu đồ ACF & PACF và Giải mã 3 điểm dị biệt
   - [Đồ thị 4] ds1_04_residual_diagnostics.png: Bộ 4 đồ thị chẩn đoán phần dư chuẩn hóa
   - [Đồ thị 5] ds1_05_forecast_arima.png: Đối đầu dự báo Train / Test giữa ARIMA(0,1,1) và ARIMA(1,1,1)
2. [Bộ đồ thị Dataset 2 (AirPassengers - Bài tập nâng cao)](#2-bộ-đồ-thị-dataset-2-airpassengers---bài-tập-nâng-cao)
   - [Đồ thị 6] ds2_01_eda_seasonality.png: Nhận diện tính mùa vụ và Mô hình nhân tính
   - [Đồ thị 7] ds2_02_seasonal_diff.png: So sánh sai phân thường vs Sai phân kép (d=1, D=1, s=12)
   - [Đồ thị 8] ds2_03_acf_pacf_sarima.png: Biểu đồ ACF/PACF xác định tham số SARIMA
   - [Đồ thị 9] ds2_04_sarima_vs_arima.png: Đối chuẩn năng lực dự báo ngoài mẫu SARIMA vs ARIMA thuần
3. [Quy tắc phản xạ nhanh khi thuyết trình bảo vệ biểu đồ](#3-quy-tắc-phản-xạ-nhanh-khi-thuyết-trình-bảo-vệ-biểu-đồ)

---

# 1. BỘ ĐỒ THỊ DATASET 1 (MONTHLY DRUG SALES - a10)

---

### [ĐỒ THỊ 1] `ds1_01_eda_raw_series.png`
**Tiêu đề:** Đồ thị Chuỗi Thời Gian Monthly Sales (a10) với Mean, Min, Max

#### 1. Mục đích của biểu đồ:
Quan sát hiện trạng ban đầu của chuỗi dữ liệu thô doanh số bán lẻ thuốc trị tiểu đường tại Úc (từ 07/1991 đến 06/2008, 204 tháng quan sát) để trả lời 3 câu hỏi của Bài 1: Có xu hướng không? Có mùa vụ không? Có dừng không?

#### 2. Các chi tiết then chốt cần chỉ ra cho người chấm:
- **Đường nét đứt màu đỏ (Mean = 10.69 Tr$)**: Mức doanh số trung bình trong suốt 17 năm là $10.69$ triệu USD.
- **Đường chấm tím (Min = 3.53 Tr$ - Tháng 07/1991)**: Mức thấp nhất ghi nhận ngay tại điểm bắt đầu lịch sử quan sát.
- **Đường chấm xanh lá (Max = 29.67 Tr$ - Tháng 01/2008)**: Mức cao nhất mọi thời đại đạt được ở năm cuối cùng của chuỗi dữ liệu.
- **Vùng bóng mờ cam (Mean ± 1 Std: 5.08 – 16.31 Tr$)**: Khoảng phân tán chuẩn bao trọn phần lớn các tháng bình thường.
- **Xu hướng tăng trưởng (Trend)**: Đường chuỗi dốc lên liên tục và rõ rệt từ trái qua phải (tăng trưởng gấp hơn 8.4 lần).
- **Tính mùa vụ (Seasonality)**: Các sóng nhọn hình răng cưa lặp lại chính xác mỗi 12 tháng: Doanh số luôn tăng vọt đạt đỉnh vào tháng 12 – tháng 1, sau đó sụt giảm đột ngột chạm đáy vào tháng 2.

#### 3. Câu trả lời chuẩn xác:
Chuỗi vi phạm hoàn toàn tính dừng yếu (Covariance Stationarity) vì giá trị trung bình $\mathbb{E}[y_t]$ trôi dạt tăng theo thời gian và phương sai mở rộng dần theo độ lớn của doanh số.

---

### [ĐỒ THỊ 2] `ds1_02_differenced_series.png`
**Tiêu đề:** Chuỗi sau Sai phân Bậc 1: y'(t) = y(t) - y(t-1)

#### 1. Mục đích của biểu đồ:
Kiểm tra xem phép lấy sai phân thường bậc 1 ($d=1$) có triệt tiêu được xu hướng tăng dài hạn và đưa chuỗi về trạng thái dừng hay chưa.

#### 2. Các chi tiết then chốt cần chỉ ra:
- **Trục trung bình 0 (Đường nét đứt đen)**: Chuỗi sai phân $\Delta y_t$ dao động đối xứng quanh mức trung bình $0$ (trung bình thực tế $\approx 0.11$ triệu USD).
- **Xu hướng đã biến mất**: Không còn hiện tượng dốc lên từ trái qua phải như ở Đồ thị 1.
- **Sóng mùa vụ vẫn còn nguyên**: Các bước nhảy vọt dương (tháng 1) và rơi thẳng đứng âm (tháng 2) vẫn lặp đi lặp lại đều đặn mỗi năm với chu kỳ 12 tháng.

#### 3. Câu trả lời chuẩn xác:
Phép sai phân bậc 1 đã thành công đưa chuỗi về trạng thái **dừng về trung bình (Mean-Stationary)**, loại bỏ nghiệm đơn vị tại tần số 0. Tuy nhiên, nó chưa loại bỏ được tính mùa vụ, do đó kiểm định ADF cho $p\text{-value} = 0.1166$ (chưa nhỏ hơn $0.05$). Ta vẫn chọn $d=1$ cho mô hình ARIMA phi mùa vụ để tránh nguy cơ over-differencing (sai phân bậc 2 làm phương sai tăng vọt gấp 2.5 lần).

---

### [ĐỒ THỊ 3] `ds1_03_acf_pacf.png`
**Tiêu đề:** Phân tích Tự tương quan ACF và Tự tương quan riêng phần PACF

#### 1. Mục đích của biểu đồ:
Ứng dụng quy trình Box-Jenkins để nhận diện các bậc $(p, d, q)$ thông qua hành vi cắt cụt (cut-off) hoặc suy giảm (tails off) của hàm ACF và PACF.

#### 2. Giải mã 3 điểm dị biệt then chốt:
- **Dải bóng màu xanh nhạt**: Ngưỡng tin cậy $95\%$ ($\pm 0.1376$). Bất kỳ cột nào nằm lọt bên trong dải xanh được coi là nhiễu ngẫu nhiên (bằng 0 về mặt thống kê).
- **Điểm dị biệt 1 (Lag 1 âm sâu trên cả 2 đồ thị: $-0.3651$)**:
  - *Giải thích*: Đây là **tự tương quan âm nhân tạo do sai phân (Differencing-induced Negative Autocorrelation)**. Khi lấy $\Delta y_t = y_t - y_{t-1}$, hai sai phân kế tiếp cùng chia sẻ chung nhiễu $\epsilon_{t-1}$ nhưng trái dấu ($-\epsilon_{t-1}$ ở $t$ và $+\epsilon_{t-1}$ ở $t-1$), tạo ra hiệp phương sai âm $-\sigma^2$ ($\rho_1 \approx -0.5$).
  - *Nhận diện*: ACF cắt cụt ngay sau lag 1 $\implies$ Chọn bậc trung bình trượt $q = 1$ để bù trừ hiện tượng này.
- **Điểm dị biệt 2 (Các gai dương khổng lồ tại Lag 12, 24, 36 trên ACF: $+0.7569, +0.6152, +0.5395$)**:
  - *Giải thích*: Sai phân bậc 1 chỉ khử xu hướng chứ không khử mùa vụ. Doanh số tháng này tương quan thuận cực mạnh với chính tháng này năm ngoái. Các gai này giảm rất chậm (Slow Decay), phản ánh tính phi dừng mùa vụ.
- **Điểm dị biệt 3 (Gai âm kỷ lục tại Lag 11 trên PACF: $-0.6340$ và các gai âm tại Lag 11, 13 trên ACF)**:
  - *Giải thích*: Cơ chế **Đối nghịch Đỉnh – Đáy**. Tháng 1 là đỉnh cao nhất năm, tháng 2 liền kề là đáy thấp nhất năm. Độ lệch 11 tháng và 13 tháng đối chiếu một Đỉnh của năm nay với một Đáy của năm trước, tạo ra tích chéo mang giá trị âm rất lớn. Trên PACF, sau khi trừ hết các trễ trung gian, tương quan đối nghịch đỉnh-đáy này lộ rõ nhất tại lag 11 với giá trị âm $-0.6340$.

#### 3. Đề xuất mô hình:
- Mô hình chính: $\text{ARIMA}(0, 1, 1)$ (tuân theo nguyên lý Parsimony, đơn giản và ít tham số).
- Mô hình phụ: $\text{ARIMA}(1, 1, 1)$ (hỗn hợp dự phòng).

---

### [ĐỒ THỊ 4] `ds1_04_residual_diagnostics.png`
**Tiêu đề:** Bộ 4 Đồ Thị Chẩn Đoán Phần Dư Mô Hình ARIMA (Residual Diagnostics — Chuẩn Senior Data Analyst)

#### 1. Mục đích của biểu đồ:
Đánh giá chất lượng của mô hình sau khi ước lượng: Mô hình đã "học" hết quy luật của chuỗi dữ liệu chưa? Phần dư $e_t = y_t - \hat{y}_t$ (sai số còn lại) đã đạt tiêu chuẩn của **Nhiễu trắng (White Noise)** hay chưa?

#### 2. Cách đọc chi tiết 4 góc đồ thị (Chuẩn Senior Data Analyst):
1. **Góc trên - trái: Phần dư theo thời gian (Residuals vs Time)**:
   - **Số liệu tích hợp tiêu đề:** $\text{Mean} = 0.32$, $\text{Std} = 1.42$.
   - **Cách quan sát:** Trục ngang là thời gian (1991 – 2005), trục đứng là sai số.
   - **Ý nghĩa:** Đường sai số dao động đồng đều quanh trục chuẩn $0$ (đường đen nét liền) và bám sát đường trung bình thực tế $\text{Mean} = 0.32$ (đường đỏ nét đứt). Không có hiện tượng trôi dạt (no drift), biên độ dao động đồng đều, khẳng định phần dư đã **dừng về trung bình và ổn định phương sai**.
2. **Góc trên - phải: Phân phối sai số vs Chuẩn Gauss (Distribution vs Normal Fit)**:
   - **Số liệu tích hợp tiêu đề:** $\text{Skewness} = -0.95$, $\text{Kurtosis} = 3.63$.
   - **Cách quan sát:** Biểu đồ mật độ màu xanh ngọc (thực tế) được ốp trực tiếp với **đường cong phân phối chuẩn lý thuyết $N(\mu, \sigma^2)$ màu xám nét đứt**.
   - **Ý nghĩa:** Phân phối thực nghiệm có dạng hình chuông tương đối đối xứng. $\text{Skewness} = -0.95$ cho thấy đuôi trái hơi dài nhẹ (do một số tháng sụt giảm mạnh vào tháng 2 sau mùa cao điểm). Độ nhọn $\text{Kurtosis} = 3.63$ (nhỉnh hơn mức chuẩn $3.0$ đôi chút) phản ánh các cú sốc mùa vụ tạo ra đuôi hơi dày (fat tails), nhưng nhìn chung vẫn thỏa mãn tốt giả định phân phối chuẩn của sai số.
3. **Góc dưới - trái: Hàm tự tương quan của phần dư (Residual ACF)**:
   - **Thông điệp tiêu đề:** `ACF Phần dư | Gai nhọn tại Lag 12, 24 cảnh báo mùa vụ`.
   - **Cách quan sát:** Dải màu xanh nhạt là khoảng tin cậy $95\%$ ($\pm 1.96/\sqrt{N}$). Hai vạch đỏ chấm mảnh đánh dấu tại **Lag 12** và **Lag 24**.
   - **Ý nghĩa then chốt:** 
     + Ở các bậc trễ ngắn (Lag 1 đến 10), toàn bộ các que ACF đều co nhỏ lại và nằm hoàn toàn bên trong dải xanh $\implies$ **Mô hình đã triệt tiêu hoàn toàn tương quan ngắn hạn!**
     + Tuy nhiên, tại **Lag 12** và **Lag 24**, các que ACF lại **nhô vọt ra ngoài dải tin cậy**. Điều này chứng minh đanh thép: *Mô hình ARIMA phi mùa vụ hoàn toàn "bó tay" trước chu kỳ 12 tháng, để sót lại nguyên vẹn quy luật mùa vụ trong phần dư.*
4. **Góc dưới - phải: Đồ thị Normal Q-Q Plot**:
   - **Thông điệp tiêu đề:** `Normal Q-Q Plot | Độ bám đường chuẩn 45°`.
   - **Cách quan sát:** Trục hoành là phân vị lý thuyết (Theoretical Quantiles), trục tung là phân vị thực tế (Sample Quantiles).
   - **Ý nghĩa:** Các điểm tròn màu xanh bám khít chằn chặn vào đường thẳng đỏ $45^\circ$ trên suốt dải giá trị từ $-2$ đến $+2$, chỉ uốn nhẹ ở hai đầu mút cực trị. Điều này xác nhận sai số tuân theo phân phối chuẩn, đảm bảo tính vững cho các khoảng tin cậy dự báo.

#### 3. Kết luận chẩn đoán phục vụ vấn đáp:
- **Phần dư có phải là White Noise tuyệt đối không?**  
  $\rightarrow$ **Trả lời:** *Chưa phải là White Noise thuần túy tuyệt đối.* Mặc dù phần dư đã đạt trung bình $\approx 0$ và sạch tương quan ngắn hạn (kiểm định Ljung-Box tại lag 10 có $p\text{-value} = 0.385 > 0.05$), nhưng việc ACF bị vi phạm ở Lag 12 và 24 là bằng chứng rõ ràng cho thấy mô hình ARIMA thường bị thiếu hụt thành phần mùa vụ. Đây là cơ sở bắt buộc phải nâng cấp lên **SARIMA** ở phần sau.

---

### [ĐỒ THỊ 5] `ds1_05_forecast_arima.png`
**Tiêu đề:** Đồ Thị Đối Chiếu Dự Báo Ngoài Mẫu (Out-of-sample Forecast) — ARIMA(1,1,1)

#### 1. Mục đích của biểu đồ:
"Bài thi tốt nghiệp" thực tế của mô hình: Đưa mô hình đã học trên tập Train (80% - 163 tháng quá khứ) đi dự báo một tương lai hoàn toàn chưa biết là tập Test (20% - 41 tháng tương lai, từ 02/2005 đến 06/2008), sau đó đối chiếu trực tiếp với số liệu thực tế ngoài mẫu.

#### 2. Các chi tiết then chốt trên thiết kế chuẩn Senior Data Analyst:
1. **Thanh KPI Banner điều hành (Góc trên - trái):**
   - Tích hợp cô đọng 4 chỉ số đo lường hiệu năng cốt lõi:
     $$\mathbf{ARIMA(1,1,1) \ Evaluation \ — \ MAE:\ 3.67M \ \vert \ RMSE:\ 4.62M \ \vert \ MAPE:\ 18.0\% \ \vert \ AIC:\ 693.6}$$
   - Đặt tại khoảng trống trần của giai đoạn 1991–1995, vừa mắt người đọc, không đè lên bất kỳ nét vẽ nào.
2. **Vùng nền kiểm thử Test (Màu xám nhạt `#F2F4F4`):**
   - Vạch nét đứt xám đánh dấu ranh giới phân tách Train/Test tại thời điểm **tháng 02/2005**. Vùng màu nền xám phân định rõ ràng giữa "quá khứ được học" và "tương lai thách thức".
3. **Đường dự báo ARIMA (Màu đỏ Coral `#E74C3C` nét đứt):**
   - **Hiện tượng trực quan:** Đường màu đỏ phóng vào tương lai **gần như phẳng lì (tiệm cận một đường thẳng nằm ngang)** ở độ cao khoảng $17.5 - 19.2$ triệu USD.
   - **Giải mã bản chất toán học (Rất hay bị hỏi thi):**
     * *Tại sao ARIMA lại dự báo phẳng lì?*
       Mô hình $\text{ARIMA}(1, 1, 1)$ có phương trình sai phân: $y'_t = c + \phi_1 y'_{t-1} + \epsilon_t + \theta_1 \epsilon_{t-1}$. Khi dự báo đa bước dài hạn (41 tháng):
       - Thành phần $\text{MA}(1)$ ($\theta_1 \epsilon_{t}$) **tắt ngấm ngay sau bước đầu tiên** vì sai số các tháng tương lai chưa biết được coi bằng $0$ ($E[\epsilon] = 0$).
       - Thành phần $\text{AR}(1)$ bị suy biến lũy thừa $\phi_1^k \to 0$ và **tiêu biến sau 4–5 tháng**.
       - Kết quả: Dự báo chỉ còn lại hằng số tăng trưởng mặc định $y'_t = y_t - y_{t-1} \approx c \approx 0.08$. Mỗi tháng tăng một lượng cố định rất nhỏ $0.08$ triệu USD khiến đồ thị trở thành **một đường thẳng đơn điệu, phẳng lì như nằm ngang**.
4. **Đường thực tế Test (Màu than đen Charcoal `#1A252C` nét đậm $2.0$ px):**
   - Biến động dữ dội theo chu kỳ hình răng cưa: Tháng 1 nào cũng vọt lên đỉnh cao chót vót ($25 - 30$ triệu USD), tháng 2 nào cũng rơi thẳng đứng xuống đáy ($12 - 15$ triệu USD).
5. **Điểm chỉ báo sai lệch cực đại (Peak Bias Annotation):**
   - Mũi tên thanh lịch chỉ vào đỉnh tháng **01/2008**:
     $$\text{Đỉnh lệch mùa vụ (01/2008): Thực tế 29.7M vs Dự báo 17.5M } (\Delta\ 12.1\text{M USD})$$
   - Đây là nơi mô hình phạm sai lầm lớn nhất vì hoàn toàn "mù" trước sự bùng nổ nhu cầu mua thuốc đầu năm của người dân Úc.

#### 3. Bộ câu hỏi vấn đáp bảo vệ biểu đồ 5:
- **Câu hỏi 1: Đường dự báo có bám sát xu hướng không?**  
  $\rightarrow$ **Trả lời:** Có bám sát xu hướng trung bình dài hạn (Mean Level) ở mức $18 - 19$ triệu USD, nhưng thất bại hoàn toàn trong việc mô phỏng các đỉnh/đáy mùa vụ ngắn hạn.
- **Câu hỏi 2: Chỉ số nào nhạy với Outlier nhất?**  
  $\rightarrow$ **Trả lời:** **RMSE ($4.62$ triệu USD)** nhạy hơn hẳn **MAE ($3.67$ triệu USD)**. Do có phép bình phương $(y_t - \hat{y}_t)^2$, khoảng lệch khổng lồ $12.1$ triệu USD tại các đỉnh tháng 1 bị phóng to theo cấp số nhân, kéo giá trị RMSE lên rất cao.
- **Câu hỏi 3: Giá trị thực tiễn của biểu đồ này là gì?**  
  $\rightarrow$ **Trả lời:** Đồ thị này là **bằng chứng thực nghiệm phản biện hoàn hảo**: Nó chứng minh mô hình ARIMA phi mùa vụ không đủ năng lực dự báo cho chuỗi có tính chu kỳ hàng năm, đặt ra tính cấp thiết phải chuyển sang **Mô hình SARIMA** (ở Phần B) để đưa đường dự báo uốn lượn theo hình sóng sin thực tế.

---

# 2. BỘ ĐỒ THỊ DATASET 2 (AIRPASSENGERS - BÀI TẬP NÂNG CAO)

---

### [ĐỒ THỊ 6] `ds2_01_eda_seasonality.png`
**Tiêu đề:** Khám phá Chuỗi AirPassengers (1949 - 1960): Trend & Mùa vụ

#### 1. Mục đích của biểu đồ:
Khám phá đặc trưng của chuỗi số lượng hành khách hàng không quốc tế để thực hiện Bước 1 của bài tập nâng cao.

#### 2. Các chi tiết then chốt cần quan sát:
- **Đường màu cam**: Thể hiện số hành khách thực tế qua 144 tháng (1949 – 1960).
- **Xu hướng tăng trưởng**: Tăng từ mức thấp nhất $104$k (11/1949) lên mức cao nhất $622$k (07/1960), trung bình $\text{Mean} = 280.30$k hành khách.
- **Tính mùa vụ ($s = 12$)**: Đỉnh cao điểm du lịch hè luôn rơi vào **tháng 7 và tháng 8**, đáy thấp điểm mùa đông luôn rơi vào **tháng 11**.
- **Hiện tượng phương sai thay đổi (Heteroskedasticity)**: Biên độ dao động giữa đỉnh hè và đáy đông ở đầu chuỗi (1949) chỉ khoảng $20 - 30$k khách, nhưng đến cuối chuỗi (1960) đã mở rộng lên tới hơn $150$k khách.

#### 3. Ý nghĩa phương pháp luận:
Đây là bằng chứng rõ rệt của **Mô hình nhân tính ($Y_t = T_t \times S_t \times I_t$)**. Bắt buộc phải áp dụng biến đổi **Logarit tự nhiên $\ln(Y_t)$** để nén phẳng phương sai, biến mô hình nhân thành mô hình cộng trước khi lấy sai phân.

---

### [ĐỒ THỊ 7] `ds2_02_seasonal_diff.png`
**Tiêu đề:** So sánh Sai phân thường d=1 vs Sai phân kép (d=1, D=1, s=12)

#### 1. Mục đích của biểu đồ:
Chứng minh một cách trực quan và định lượng sự khác biệt giữa sai phân thường và sai phân mùa vụ (Bước 3).

#### 2. So sánh 2 tầng biểu đồ:
- **Tầng trên (Sai phân thường $d=1$ của chuỗi Log)**:
  - Xu hướng dốc lên đã bị loại bỏ, nhưng dạng sóng mùa vụ hình sin vẫn còn nguyên vẹn.
  - Kiểm định ADF: $\text{ADF} = -2.7171, p\text{-value} = 0.0711 > 0.05 \implies$ **Chưa dừng ở mức ý nghĩa 5%**.
- **Tầng dưới (Sai phân kép $d=1, D=1, s=12$)**:
  - Triệt tiêu phẳng lì toàn bộ cả xu hướng và chu kỳ mùa vụ, các điểm dao động hoàn toàn ngẫu nhiên quanh trục 0.
  - Kiểm định ADF: $\text{ADF} = -4.4433, p\text{-value} = 0.00025 \ll 0.01 \implies$ **Bác bỏ dứt khoát $H_0$, chuỗi đạt tính dừng tuyệt đối**.

---

### [ĐỒ THỊ 8] `ds2_03_acf_pacf_sarima.png`
**Tiêu đề:** ACF và PACF Chuỗi Sai phân Kép (48 độ trễ)

#### 1. Mục đích của biểu đồ:
Xác định đồng thời các tham số phi mùa vụ $(p, q)$ và tham số mùa vụ $(P, Q)$ theo Box-Jenkins (Bước 4).

#### 2. Cách phân tách 2 miền độ trễ để đọc đồ thị:
- **Miền phi mùa vụ (Độ trễ ngắn $1 \le lag \le 3$)**:
  - ACF: Có một gai âm đơn độc tại lag 1 ($r_1 \approx -0.34$, vượt ra ngoài dải xanh) và cắt cụt ngay lập tức ở lag 2, 3 $\implies$ **Xác định bậc trung bình trượt phi mùa vụ $q = 1$**.
  - PACF: Gai âm tại lag 1 và giảm dần $\implies p \in \{0, 1\}$.
- **Miền mùa vụ (Bội số của $s=12$ là $12, 24, 36, 48$)**:
  - ACF: Tại lag 12 có một gai âm rất nổi bật ($r_{12} \approx -0.38$, vượt xa dải xanh), trong khi tại lag 24 và lag 36 thì hệ số ACF hoàn toàn nằm bên trong dải xanh $\implies$ Cắt cụt sau đúng 1 chu kỳ mùa vụ $\implies$ **Xác định bậc trung bình trượt mùa vụ $Q = 1$**.
  - PACF: Có gai tại lag 12 và giảm dần ở các bội số tiếp theo $\implies P \in \{0, 1\}$.

#### 3. Mô hình tối ưu:
Mô hình hàng không kinh điển **$\text{SARIMA}(0, 1, 1)(0, 1, 1)_{12}$** và mô hình tổng quát **$\text{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$**.

---

### [ĐỒ THỊ 9] `ds2_04_sarima_vs_arima.png`
**Tiêu đề:** So Sánh Năng Lực Dự Báo: SARIMA (Có mùa vụ) vs ARIMA Thuần

#### 1. Mục đích của biểu đồ:
Đối chuẩn trực tiếp hiệu năng dự báo ngoài mẫu trên 29 tháng tập Test (08/1958 – 12/1960) để hoàn thành Bước 6 và Bước 7.

#### 2. So sánh trực quan giữa 2 mô hình:
- **Đường màu đen (Test thực tế)**: Số hành khách thực tế biến động mạnh từ $359$k đến $622$k khách.
- **Đường màu xanh lá nét đứt (SARIMA)**: Dự báo bám sát tuyệt đối từng đỉnh sóng mùa hè tháng 7-8 và đáy sóng mùa đông tháng 11. Dải bóng mờ tin cậy 95% màu xanh bao trọn toàn bộ các điểm thực tế.
- **Đường màu đỏ chấm chấm (ARIMA thuần)**: Suy biến thành một đường thẳng tiệm cận phẳng, hoàn toàn không có khả năng mô phỏng các đợt bùng nổ du lịch hè, tạo ra sai lệch khổng lồ tại các đỉnh (lệch gần 150 nghìn khách).

#### 3. Bảng đối chuẩn định lượng:
| Chỉ số | ARIMA(1,1,1) thuần | SARIMA(1,1,1)(1,1,1)[12] | Mức độ cải thiện của SARIMA |
| :--- | :---: | :---: | :---: |
| **MAE (Hành khách)** | $84.25$ | **$12.46$** | Giảm $85.2\%$ sai số tuyệt đối |
| **RMSE (Hành khách)** | $96.33$ | **$15.25$** | **Giảm $84.2\%$ sai số toàn phương** |
| **MAPE (%)** | $21.01\%$ | **$2.82\%$** | Sai số phần trăm chưa đầy $3\%$ |

---

# 3. QUY TẮC PHẢN XẠ NHANH KHI THUYẾT TRÌNH BẢO VỆ BIỂU ĐỒ

Khi thầy cô hoặc hội đồng hỏi về biểu đồ, bạn chỉ cần bám sát 4 câu hỏi cốt lõi:

1. **Hỏi: "Biểu đồ này biểu diễn cái gì?"**  
   $\rightarrow$ Trả lời ngay: Trục hoành là gì (thời gian), trục tung là gì (đơn vị đo), đường màu nào thể hiện cái gì (thực tế hay dự báo).
2. **Hỏi: "Dựa vào đâu trên biểu đồ để chọn tham số $p, d, q$?"**  
   $\rightarrow$ Trả lời: Bậc $d$ dựa vào việc chuỗi đã dao động quanh trục 0 chưa; bậc $q$ dựa vào điểm cắt cụt của que ACF; bậc $p$ dựa vào điểm cắt cụt hoặc suy giảm của que PACF.
3. **Hỏi: "Tại sao ACF lại có que âm ở lag 1 và que âm ở lag 11?"**  
   $\rightarrow$ Trả lời: Que âm lag 1 là tương quan âm nhân tạo do phép lấy sai phân $\Delta y_t$; que âm lag 11 là sự đối nghịch giữa Đỉnh tháng 1 năm nay với Đáy tháng 2 năm ngoái (cách nhau 11 tháng).
4. **Hỏi: "Tại sao đường dự báo của ARIMA lại phẳng lì trong khi SARIMA lại nhấp nhô sóng?"**  
   $\rightarrow$ Trả lời: ARIMA phi mùa vụ chỉ có toán tử $(1-B)$ nên sau vài bước trễ, thành phần trung bình trượt tắt về 0 khiến hàm dự báo suy biến thành đường tiệm cận phẳng; SARIMA có toán tử mùa vụ $(1-B^{12})$ giúp duy trì bộ nhớ chu kỳ 12 tháng vĩnh viễn trong tương lai.
