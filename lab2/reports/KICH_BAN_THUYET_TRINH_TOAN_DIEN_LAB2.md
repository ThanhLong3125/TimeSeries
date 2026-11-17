# KỊCH BẢN THUYẾT TRÌNH TOÀN DIỆN LAB 2: PHÂN TÍCH CHUỖI THỜI GIAN & MÔ HÌNH HÓA ARIMA / SARIMA

**Khóa học:** Phân tích Chuỗi thời gian (Time Series Analysis)  
**Tập dữ liệu báo cáo:**
1. `Dataset 1 (a10.csv)`: Doanh số bán lẻ dược phẩm trị tiểu đường hàng tháng tại Úc (1991 – 2008, 204 quan sát).
2. `Dataset 2 (AirPassengers.csv)`: Số lượng hành khách hàng không quốc tế hàng tháng (1949 – 1960, 144 quan sát).  
**Thời lượng dự kiến:** 12 – 15 phút  
**Phong cách trình bày:** Chuyên nghiệp, tập trung đào sâu vào **bản chất dữ liệu, đối soát đồ thị trực quan, phân tích chỉ số kinh tế lượng** kết hợp với tư duy kiến trúc **Production Pipeline / MLOps**.

---

## BẢNG CHỈ DẪN NHANH DÀNH CHO DIỄN GIẢ (SPEAKER'S QUICK MAP)

| Phút | Phần thuyết trình | Thao tác trên màn hình (Hành động & Điểm chỉ) | Mục tiêu cần đạt |
| :---: | :--- | :--- | :--- |
| **00 - 02** | Mở đầu & Kiến trúc Pipeline | Chiếu file `configs/arima_a10.yaml` và sơ đồ luồng dữ liệu | Gây ấn tượng về tư duy MLOps tách biệt Code & Config |
| **02 - 04** | Bài 1: EDA Chuỗi Gốc `a10` | Chiếu ảnh `a10_arima_baseline_eda_01_raw.png` (chỉ vào Trend, Seasonality, Min/Max) | Chứng minh trực giác chuỗi không dừng |
| **04 - 06** | Bài 2: Kiểm định ADF Chuỗi Gốc | Chiếu đoạn log `ADF Test [Candidate Lags 0-14]` trong terminal / `pipeline.log` | Phân tích giả thuyết $H_0$, $p\text{-value}=0.9989$ |
| **06 - 08** | Bài 3 & 4: Sai phân & ACF/PACF | Chiếu `a10_..._eda_02_differenced.png` và `a10_..._eda_03_acf_pacf.png` | Phân tích bẫy Over-differencing ($d=1$ vs $d=2$), đọc bậc $p=1, q=1$ |
| **08 - 11** | Bài 5: Huấn luyện, Chẩn đoán Phần dư & Dự báo Test Set | Chiếu bảng `SARIMAX Results`, ảnh `residuals.png` 4 góc và ảnh `forecast.png` | Trả lời trọn vẹn các câu hỏi: Nghiệm roots, Ljung-Box, MAE, RMSE, MAPE, Peak Error |
| **11 - 13** | Nâng cao SARIMA: `AirPassengers` | Chiếu ảnh `airpassengers_sarima_forecast.png` và bảng so sánh với ARIMA thuần | Chứng minh SARIMA giảm $84\%$ sai số, bám sát từng đỉnh sóng |
| **13 - 15** | Cổng kiểm định Champion vs Challenger & Kết luận | Chiếu bảng `CHAMPION VS CHALLENGER GATE` trong log và file `audit_trail.jsonl` | Chốt hạ điểm sáng chống phình đĩa, bảo vệ mô hình tối ưu |

---

# LỜI THOẠI CHI TIẾT TỪNG PHÚT (WORD-BY-WORD PRESENTATION SCRIPT)

---

## 🎙️ PHẦN 1: MỞ ĐẦU — GIỚI THIỆU BÀI TOÁN & TƯ DUY KIẾN TRÚC (Phút 00 – 02)

> **Lời thoại mở màn:**  
> "Kính thưa Thầy/Cô và các bạn sinh viên,  
> Hôm nay em xin đại diện nhóm trình bày báo cáo thực hành **Lab 2: Kiểm định ADF và Mô hình hóa Dự báo Chuỗi Thời Gian với ARIMA và SARIMA**.
>
> Thay vì tiếp cận theo cách làm truyền thống là viết các đoạn mã phân tích rời rạc trên Jupyter Notebook rất khó tái lập và khó triển khai thực tế, toàn bộ bài thực hành của nhóm em được đóng gói thành một **Production Time Series Pipeline hoàn chỉnh**.

*(Hành động: Mở tệp [lab2/configs/arima_a10.yaml](file:///home/thanhlong/Documents/TimeSeries/lab2/configs/arima_a10.yaml) trên màn hình)*

> "Điểm mấu chốt đầu tiên của kiến trúc này là **Tách biệt tuyệt đối giữa Mã nguồn (Code) và Cấu hình (Config)**. 
> Toàn bộ tham số bài toán từ đường dẫn dữ liệu, tần suất chuỗi thời gian `MS` (*Month Start*), tỷ lệ chia Train/Test 80/20, cho đến bộ tham số mô hình $(p, d, q)$ đều được khai báo declarative trong tệp YAML này. Khi cần đổi mô hình hoặc chuyển sang tập dữ liệu mới, ta không cần can thiệp vào logic code Python bên dưới.
>
> Toàn bộ quá trình thực thi được giám sát đa tầng: Tệp `pipeline.log` ghi lại chi tiết từng bước tính toán, tệp `audit_trail.jsonl` lưu vết lịch sử bất biến dạng append-only, và tệp JSON lưu giữ chỉ số của mô hình Champion tốt nhất.
>
> Sau đây, em xin đi sâu vào kết quả thực nghiệm chi tiết trên hai tập dữ liệu chuẩn mực của môn học."

---

## 🎙️ PHẦN 2: BÀI 1 — KHÁM PHÁ DỮ LIỆU THÔ (EDA) TRÊN TẬP A10 (Phút 02 – 04)

*(Hành động: Chiếu tệp biểu đồ [lab2/artifacts/figures/a10_arima_baseline_eda_01_raw.png](file:///home/thanhlong/Documents/TimeSeries/lab2/artifacts/figures/a10_arima_baseline_eda_01_raw.png) lên toàn màn hình)*

> **Lời thoại:**  
> "Xin mời Thầy/Cô cùng quan sát biểu đồ EDA chuỗi thời gian gốc của tập dữ liệu **Doanh số bán lẻ dược phẩm trị tiểu đường tại Úc (`a10.csv`)** gồm 204 quan sát tháng, từ tháng 07/1991 đến tháng 06/2008.
>
> Nhìn vào đồ thị, hệ thống tự động phân tích và trực quan hóa 4 đặc tính then chốt:
> 
> 1. **Xu hướng tăng trưởng dài hạn rất dốc (Deterministic Upward Trend):**  
>    *(Chỉ tay từ góc dưới bên trái lên góc trên bên phải)*  
>    Doanh số bắt đầu ở mức đáy lịch sử là **2.81 triệu USD** vào tháng 02/1993, liên tục leo dốc qua các năm và đạt đỉnh cao nhất mọi thời đại là **29.67 triệu USD** vào tháng 01/2008 (tăng trưởng hơn 10 lần). Đường mức trung bình toàn chuỗi được pipeline vẽ nét đứt màu đỏ ở mức **10.69 triệu USD**, với dải phân tán chuẩn $\pm 1\text{ Std}$ trải rộng từ $4.96$ đến $16.42$ triệu USD.
>
> 2. **Tính chu kỳ mùa vụ hàng năm rõ rệt (Strong Seasonality với chu kỳ $s=12$):**  
>    *(Chỉ vào các đỉnh nhọn lặp lại đều đặn mỗi năm)*  
>    Chuỗi có quy luật dao động lặp lại theo chu kỳ 12 tháng cực kỳ chuẩn xác: Cứ đến tháng 12 và tháng 1 hàng năm, doanh số lại dựng đứng tạo đỉnh nhọn hoắt (do người bệnh tích trữ thuốc trước kỳ nghỉ lễ và chính sách trợ giá y tế cuối năm của chính phủ Úc), ngay sau đó vào tháng 2, doanh số lại sụt giảm chạm đáy của năm.
>
> 3. **Hiện tượng phương sai mở rộng nhẹ (Mild Heteroskedasticity):**  
>    Biên độ dao động đỉnh - đáy trong những năm đầu (1991 - 1995) chỉ dao động khoảng 2 - 3 triệu USD, nhưng về những năm cuối (2005 - 2008), biên độ này đã phình to lên tới gần 10 triệu USD.
>
> 👉 **Trả lời câu hỏi Bài 1:**  
> - *Chuỗi có xu hướng không?* $\rightarrow$ **CÓ**, xu hướng tăng trưởng tuyến tính rất mạnh.  
> - *Có dấu hiệu mùa vụ không?* $\rightarrow$ **CÓ**, chu kỳ mùa vụ hàng năm $s=12$ tháng lặp lại rất nhất quán.  
> - *Chuỗi có khả năng dừng không?* $\rightarrow$ **KHÔNG DỪNG**, vì cả kỳ vọng toán $\mathbb{E}[y_t]$ lẫn phương sai $\text{Var}(y_t)$ đều trôi dạt và biến thiên liên tục theo thời gian."

---

## 🎙️ PHẦN 3: BÀI 2 — KIỂM ĐỊNH ADF TRÊN CHUỖI GỐC (Phút 04 – 06)

*(Hành động: Chuyển sang cửa sổ Terminal hoặc chiếu đoạn trích log kiểm định ADF từ [pipeline.log](file:///home/thanhlong/Documents/TimeSeries/lab2/logs/pipeline.log))*

```text
[INFO] [stationarity.py:22] - --- ADF Test [a10_arima_baseline (Train Raw)] Autolag Evaluation (Criterion: AIC) ---
[INFO] [stationarity.py:29] -    Candidate Lag  0: AIC=  568.2089
[INFO] [stationarity.py:29] -    Candidate Lag  1: AIC=  556.8780
...
[INFO] [stationarity.py:29] -    Candidate Lag 14: AIC=  262.0001 [MINIMUM AIC - SELECTED]
[INFO] [stationarity.py:34] - ADF Result: stat=2.2309, p_value=9.9891e-01, used_lag=14/14, crit_5%=-2.8813 -> NON-STATIONARY
```

> **Lời thoại:**  
> "Để xác nhận kết luận trực quan bằng định lượng toán học, pipeline thực thi **Kiểm định Augmented Dickey-Fuller (ADF)** trên tập Train của chuỗi gốc.
>
> Nhìn vào nhật ký log, pipeline của tụi em đã trích xuất toàn bộ quá trình thuật toán tự động dò tìm độ trễ tối ưu (*Autolag Search*):
> - Thuật toán đã duyệt qua toàn bộ 15 bậc lag ứng viên (từ Lag 0 đến Lag 14).
> - Tại Lag 0, điểm số AIC rất cao ($568.21$). Thuật toán tự động lựa chọn **Lag 14** vì đây là bậc trễ đạt giá trị AIC cực tiểu ($262.00$), giúp loại bỏ sạch hiện tượng tự tương quan trong phần dư của phương trình hồi quy ADF.
>
> 👉 **Phân tích kết quả & Trả lời câu hỏi Bài 2:**  
> - **Giả thuyết $H_0$ của ADF:** Chuỗi chứa nghiệm đơn vị ($\gamma = 0$ hay $\rho = 1$), tức **Chuỗi không dừng (Non-stationary)**. Giả thuyết đối $H_1$: Chuỗi dừng ($\gamma < 0$).
> - **Giá trị thống kê ADF:** Đạt **$+2.2309$**. Để bác bỏ $H_0$ ở mức ý nghĩa $5\%$, thống kê kiểm định bắt buộc phải âm hơn giá trị tới hạn $-2.8813$. Ở đây $+2.2309$ nằm hoàn toàn về phía bên phải vùng chấp nhận $H_0$.
> - **Giá trị $p\text{-value}$:** Đạt **$0.9989$** (gần như bằng 100%), lớn hơn rất nhiều so với ngưỡng $\alpha = 0.05$.
> - **Ý nghĩa của $p$-value:** Cho biết xác suất xảy ra sai lầm loại 1 khi bác bỏ $H_0$ lên tới $99.89\%$. Do đó, ta hoàn toàn **không thể bác bỏ $H_0$**.  
> $\implies$ **Kết luận:** Khẳng định bằng toán học vững chắc rằng chuỗi gốc là **CHUỖI KHÔNG DỪNG**."

---

## 🎙️ PHẦN 4: BÀI 3 & 4 — SAI PHÂN & NHẬN DIỆN BẬC BẰNG ACF/PACF (Phút 06 – 08)

*(Hành động: Chiếu biểu đồ sai phân [lab2/artifacts/figures/a10_arima_baseline_eda_02_differenced.png](file:///home/thanhlong/Documents/TimeSeries/lab2/artifacts/figures/a10_arima_baseline_eda_02_differenced.png))*

> **Lời thoại:**  
> "Để triệt tiêu xu hướng tăng và đưa chuỗi về trạng thái dừng, chúng em áp dụng sai phân bậc 1: $\Delta y_t = y_t - y_{t-1}$.
>
> Nhìn vào biểu đồ so sánh sai phân trên màn hình, pipeline của tụi em tích hợp một cơ chế an toàn kinh tế lượng gọi là **Over-differencing Safeguard (Kiểm soát bẫy quá sai phân)**:
> 
> - **Ở bậc $d=1$ (Đồ thị phía trên):** Xu hướng dốc đứng trước đó đã hoàn toàn biến mất, chuỗi dao động quanh mức trung bình 0. Đặc biệt, phương sai của chuỗi đã **giảm sốc tới $83.6\%$** (từ $15.52$ xuống còn **$2.54$**), và hệ số tự tương quan độ trễ 1 là $r_1 = -0.320$.
> - **Nếu ta cố tình sai phân tiếp lên bậc $d=2$ (Đồ thị phía dưới):** Phương sai không những không giảm mà bị **dội ngược tăng vọt tới $+164.6\%$** (lên $6.73$), đồng thời hệ số tự tương quan $r_1$ rơi tụt xuống mức $-0.592 < -0.5$.
> 
> 👉 **Trả lời câu hỏi Bài 3:**  
> - *Sau sai phân chuỗi đã dừng chưa?* Xu hướng trung bình đã phẳng hoàn toàn, phương sai đạt mức tối thiểu. Giá trị $p\text{-value}$ kiểm định ADF sau sai phân đạt $0.1166$ (tiệm cận ngưỡng 10%). Chuỗi chưa đạt dừng tuyệt đối ở mức 5% là do **vẫn còn thành phần mùa vụ 12 tháng** ẩn bên dưới, chứ không phải do chưa đủ sai phân thường.
> - *Giá trị $d$ trong ARIMA là bao nhiêu?* Bắt buộc chọn **$d = 1$**. Nếu tăng lên $d=2$, chuỗi sẽ rơi vào bẫy **Quá sai phân (Over-differencing)**, làm tăng phương sai và đưa vào các cấu trúc tương quan âm giả mạo."

*(Hành động: Chiếu biểu đồ [lab2/artifacts/figures/a10_arima_baseline_eda_03_acf_pacf.png](file:///home/thanhlong/Documents/TimeSeries/lab2/artifacts/figures/a10_arima_baseline_eda_03_acf_pacf.png))*

> "Sau khi đã chốt $d=1$, chúng em tiến hành vẽ biểu đồ hàm tự tương quan ACF và PACF trên chuỗi đã sai phân để nhận diện bậc $p$ và $q$ theo phương pháp Box-Jenkins:
>
> - **Quan sát đồ thị PACF (Bên dưới):** Ta thấy có 2 thanh nhọn vượt ra khỏi dải tin cậy 95% màu xanh tại **Lag 1** và **Lag 2**, sau đó giảm dần vào trong dải tin cậy $\implies$ Gợi ý thành phần tự hồi quy **$p \in \{1, 2\}$**.
> - **Quan sát đồ thị ACF (Bên trên):** Hệ số tương quan đột ngột **cắt cụt (sharp cutoff)** sau **Lag 1** với giá trị âm rõ rệt, tiếp sau đó xuất hiện sóng hình sin chu kỳ nhô lên tại các độ trễ bội số của 12 (Lag 12, Lag 24, Lag 36) $\implies$ Gợi ý thành phần trung bình trượt **$q = 1$**.
>
> 👉 **Trả lời câu hỏi Bài 4:**  
> - ACF cắt cụt ở Lag 1 $\rightarrow$ Đề xuất $q = 1$.  
> - PACF cắt cụt ở Lag 1 và 2 $\rightarrow$ Đề xuất $p \in \{1, 2\}$.  
> - **Hai mô hình ARIMA ứng viên sáng giá nhất:** **$\text{ARIMA}(1, 1, 1)$** và **$\text{ARIMA}(2, 1, 1)$**."

---

## 🎙️ PHẦN 5: BÀI 5 — HUẤN LUYỆN, CHẨN ĐOÁN PHẦN DƯ & DỰ BÁO NGOÀI MẪU (Phút 08 – 11)

> **Lời thoại:**  
> "Bước sang Bài 5, dữ liệu được chia theo tỷ lệ 80% Train (163 tháng: 1991-07 đến 2005-01) và 20% Test (41 tháng: 2005-02 đến 2008-06).

*(Hành động: Chiếu bảng kết quả huấn luyện từ [pipeline.log](file:///home/thanhlong/Documents/TimeSeries/lab2/logs/pipeline.log))*

```text
==============================================================================
Dep. Variable:                  value   No. Observations:                  163
Model:                 ARIMA(1, 1, 1)   Log Likelihood                -288.234
AIC                            582.468   BIC                            591.731
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.3322      0.237      1.404      0.160      -0.132       0.796
ma.L1         -0.8059      0.134     -6.017      0.000      -1.068      -0.543
sigma2         2.0487      0.187     10.957      0.000       1.682       2.415
==============================================================================
CHẨN ĐOÁN HỘI TỤ, TÍNH DỪNG & KHẢ NGHỊCH:
 • Thuật toán tối ưu (MLE) : HỘI TỤ THÀNH CÔNG
 • Nghiệm AR (Min Modulus) : 3.0098 (> 1.0 -> DỪNG)
 • Nghiệm MA (Min Modulus) : 1.2408 (> 1.0 -> KHẢ NGHỊCH)
```

> "Khi so sánh hai mô hình trên tập Train:
> - Mô hình $\text{ARIMA}(1, 1, 1)$ đạt $AIC = 582.47, BIC = 591.73$.
> - Mô hình $\text{ARIMA}(2, 1, 1)$ đạt $AIC = 584.46, BIC = 596.81$.  
> $\implies$ Cả AIC và BIC của $\text{ARIMA}(1, 1, 1)$ đều thấp hơn, nên mô hình $\text{ARIMA}(1, 1, 1)$ được chọn làm mô hình tối ưu.
>
> Thuật toán tối ưu hóa ước lượng hợp lý cực đại (MLE) hội tụ hoàn toàn. Cả hai nghiệm đặc trưng của AR và MA đều có mô-đun lớn hơn 1.0 ($3.01$ và $1.24$), khẳng định mô hình **hoàn toàn dừng và khả nghịch** về mặt toán học.

*(Hành động: Chiếu biểu đồ 4 góc chẩn đoán phần dư [lab2/artifacts/figures/a10_arima_baseline_residuals.png](file:///home/thanhlong/Documents/TimeSeries/lab2/artifacts/figures/a10_arima_baseline_residuals.png))*

> **Chẩn đoán phần dư (Residual Diagnostics) qua bộ 4 đồ thị:**
> 1. *Góc trên bên trái (Residuals vs Time):* Sai số $e_t$ dao động ổn định quanh mức trung bình **$0.318 \approx 0$**, không có hiện tượng phương sai trôi dạt vô định.
> 2. *Góc trên bên phải (Residual Distribution):* Đường phân phối KDE màu xanh ngọc bám khá sát đường cong chuẩn Gaussian nét đứt, hệ số bất đối xứng $\text{Skewness} = -0.955$ và độ nhọn $\text{Kurtosis} = 3.63$ (đuôi hơi dày nhẹ).
> 3. *Góc dưới bên trái (Residual ACF):* Đây là đồ thị quan trọng nhất:
>    - Tại các lag ngắn hạn (Lag 1 đến 10), tất cả các thanh đều nằm lọt bên trong dải tin cậy 95%. Kiểm định Ljung-Box tại Lag 10 cho $p\text{-value} = 0.3857 > 0.05 \implies$ **Phần dư đạt tính White Noise trong ngắn hạn**.
>    - **Tuy nhiên**, tại đúng Lag 12 và Lag 24, ta thấy các đỉnh nhọn màu đỏ vọt hẳn ra ngoài dải tin cậy, khiến kiểm định Ljung-Box tại Lag 12 có $p < 0.001$. Điều này chứng minh phần dư vẫn còn sót lại tự tương quan mùa vụ mà mô hình ARIMA thuần chưa giải quyết được!
> 4. *Góc dưới bên phải (Normal Q-Q Plot):* Phần lớn các điểm phân vị thực nghiệm nằm bám thẳng trên đường 45 độ, chỉ bị tách nhẹ ở hai đầu mút do các đỉnh mùa vụ đột biến.

*(Hành động: Chiếu biểu đồ đối chiếu dự báo [lab2/artifacts/figures/a10_arima_baseline_forecast.png](file:///home/thanhlong/Documents/TimeSeries/lab2/artifacts/figures/a10_arima_baseline_forecast.png))*

> **Đánh giá Hiệu năng Dự báo Ngoài Mẫu (Out-of-sample Forecast trên Test Set):**
> 
> Quan sát 41 tháng trong vùng Test nền xám (từ 2005 đến 2008):
> - **Đường xu hướng:** Đường dự báo nét đứt màu đỏ bám sát đường xu hướng tăng trưởng trung bình thực tế màu đen. Tuy nhiên, vì là mô hình phi mùa vụ, đường dự báo bị phẳng hóa (flattening) thành một đường thẳng tiệm cận, không thể uốn lượn tái tạo lại các đỉnh tháng 12 - tháng 1 và các đáy tháng 2.
> - **Chỉ số KPI đạt được (Gắn trên Banner góc trái):**
>   - $\text{MAE} = \mathbf{3.67}$ triệu USD (Mỗi tháng dự báo lệch trung bình 3.67 triệu USD).
>   - $\text{RMSE} = \mathbf{4.62}$ triệu USD (Độ lệch chuẩn của sai số dự báo).
>   - $\text{MAPE} = \mathbf{18.04\%}$ (Sai số tương đối ở mức chấp nhận được đối với mô hình baseline).
> - **Vị trí sai lệch lớn nhất (Peak Error Annotation):** Điểm mũi tên đỏ chỉ rõ đỉnh sai số xảy ra đúng vào **tháng 01/2008**, khi doanh số thực tế lập đỉnh $29.67$ triệu USD nhưng mô hình chỉ đoán được $17.54$ triệu USD (chênh lệch tới **$12.13$ triệu USD**).
>
> 👉 **Tổng kết trả lời 9 câu hỏi Bài 5:**  
> 1. *Vì sao không shuffle?* Để tránh rò rỉ dữ liệu tương lai (Lookahead bias) và bảo toàn tính tự tương quan thời gian. Test set đóng vai trò dữ liệu tương lai chưa từng thấy để thẩm định độc lập.  
> 2. *Ý nghĩa $p=1, d=1, q=1$?* $d=1$ là sai phân khử trend, $p=1$ là phụ thuộc sai phân 1 kỳ trước, $q=1$ là bù trừ sai số sốc ngẫu nhiên 1 kỳ trước.  
> 3. *Mô hình có hội tụ không?* Có hội tụ hoàn toàn (MLE thành công).  
> 4. *Residuals có phải White Noise không?* Đạt White noise ở các lag ngắn hạn (Lag 1-10), nhưng chưa phải White noise hoàn hảo ở lag mùa vụ 12 và 24.  
> 5. *Nếu residuals còn tự tương quan thì điều chỉnh gì?* Bắt buộc chuyển sang mô hình **SARIMA** để bổ sung các tham số mùa vụ $(P, D, Q)_{12}$.  
> 6. *Dự đoán có bám sát xu hướng không?* Bám sát xu hướng trung bình nhưng không bắt được sóng mùa vụ.  
> 7. *Sai lệch lớn nhất ở đâu?* Tại các tháng đỉnh (tháng 12 - 1) và đáy (tháng 2).  
> 8. *Chỉ số nào nhạy với outlier nhất?* **RMSE** nhạy nhất vì có hàm bình phương sai số.  
> 9. *RMSE lớn hay nhỏ nói lên điều gì?* RMSE đo lường mức độ phân tán chuẩn của sai số; RMSE càng nhỏ thì khoảng cách giữa dự báo và thực tế càng hẹp, mô hình càng đáng tin cậy."

---

## 🎙️ PHẦN 6: BÀI TẬP NÂNG CAO — SARIMA TRÊN AIRPASSENGERS (Phút 11 – 13)

*(Hành động: Chiếu biểu đồ dự báo SARIMA [lab2/artifacts/figures/airpassengers_sarima_forecast.png](file:///home/thanhlong/Documents/TimeSeries/lab2/artifacts/figures/airpassengers_sarima_forecast.png))*

> **Lời thoại:**  
> "Để giải quyết triệt để bài toán mùa vụ mà ARIMA thuần bó tay, nhóm em thực hiện phần bài tập nâng cao với mô hình **SARIMA trên tập dữ liệu số lượng hành khách hàng không quốc tế `AirPassengers` (144 tháng, 1949 – 1960)**.
>
> 1. **Nhận diện mùa vụ & Tiền xử lý Logarit:**  
>    Dữ liệu có biên độ mùa hè (tháng 7-8) phình to theo thời gian (mô hình nhân tính Multiplicative). Trong cấu hình YAML, pipeline kích hoạt cờ `apply_log: true` để lấy logarit tự nhiên $z_t = \ln(y_t)$ nhằm ép phẳng phương sai.
> 2. **Sai phân kép ($d=1, D=1, s=12$):**  
>    Áp dụng sai phân bậc 1 để loại bỏ trend và sai phân mùa vụ bậc 1 tại chu kỳ $s=12$ để triệt tiêu mùa. Kiểm định ADF sau sai phân kép đạt kết quả xuất sắc: $ADF = -4.44, p\text{-value} = 0.0002 \ll 0.01 \implies$ **Chuỗi dừng hoàn hảo!**
> 3. **Huấn luyện mô hình:** Huấn luyện $\text{SARIMA}(0, 1, 1)(0, 1, 1)_{12}$.
>
> *(Chỉ tay vào đường dự báo nét đứt màu đỏ trên biểu đồ AirPassengers)*
>
> Nhìn vào 29 tháng của tập Test (1958 - 1960), mô hình SARIMA uốn lượn bám sát từng đỉnh sóng tháng 7-8 và từng đáy sóng mùa đông một cách kinh ngạc!
>
> **Bảng So sánh Đối chuẩn Hiệu năng: SARIMA vs. ARIMA thuần:**
>
> | Tiêu chí đánh giá | ARIMA(1, 1, 1) thuần | SARIMA(0, 1, 1)(0, 1, 1)[12] | Mức độ cải thiện |
> | :--- | :---: | :---: | :---: |
> | **Test MAE (Khách hàng)** | $84.25$ | **$12.95$** | **Giảm $84.6\%$** |
> | **Test RMSE (Khách hàng)** | $96.33$ | **$16.16$** | **Giảm $83.2\%$ (hơn 6 lần)** |
> | **Test MAPE (%)** | $21.01\%$ | **$2.89\%$** | **Sai số chỉ còn dưới $3\%$** |
>
> $\implies$ Kết quả này chứng minh rằng với dữ liệu có tính chu kỳ mùa vụ mạnh, việc đưa các thành phần mùa vụ $(P, D, Q)_s$ vào mô hình là **bước đi bắt buộc mang tính quyết định**."

---

## 🎙️ PHẦN 7: ĐIỂM SÁNG MLOPS — CỔNG KIỂM ĐỊNH CHAMPION VS. CHALLENGER (Phút 13 – 15)

*(Hành động: Chiếu bảng đối đầu trong [lab2/logs/pipeline.log](file:///home/thanhlong/Documents/TimeSeries/lab2/logs/pipeline.log))*

```text
================================================================================
CHAMPION VS CHALLENGER MODEL EVALUATION GATE (MLOps Time Series Gate)
================================================================================
Tiêu chí so sánh           | Champion     | Challenger   | Chênh lệch      | Đánh giá
--------------------------------------------------------------------------------
Test RMSE (Tiêu chí chính) | 4.6243       | 11.7526      |  +154.15%       | CHAMPION TỐT HƠN
Test MAE                   | 3.6667       | 10.9696      |  +199.17%       | CHAMPION TỐT HƠN
Test MAPE (%)              | 18.04        | 54.05        |   +36.01% pts   | CHAMPION TỐT HƠN
AIC (Mức độ tối giản)      | 582.47       | 791.67       |  +209.20 pts     | CHAMPION GỌN HƠN
BIC (Phạt số tham số)      | 591.73       | 800.95       |  +209.22 pts     | CHAMPION GỌN HƠN
--------------------------------------------------------------------------------
KẾT QUẢ PHÂN XỬ : TỪ CHỐI (REJECTED)
LÝ DO CHI TIẾT  : Challenger kém hơn Champion: Test RMSE tăng +154.15% (từ 4.6243 lên 11.7526).
================================================================================
```

> **Lời thoại chốt hạ:**  
> "Kính thưa Thầy/Cô, trước khi kết thúc bài thuyết trình, em xin dành 1 phút để giới thiệu điểm sáng tạo kỹ thuật của đồ án: Module **Cổng Kiểm Định Mô Hình (Champion vs. Challenger Model Gate)** trong file [model_comparator.py](file:///home/thanhlong/Documents/TimeSeries/lab2/src/model_comparator.py).
>
> Trong các dự án thực tế, nếu ta ghi đè mù quáng, một mô hình thử nghiệm kém sẽ xóa mất mô hình tốt nhất trước đó. Còn nếu lưu mọi file, ổ đĩa sẽ nhanh chóng bị phình to bởi các file nhị phân mô hình nặng hàng chục MB và hàng loạt hình ảnh rác.
>
> Để giải quyết triệt để bài toán này, pipeline thiết lập mô hình tốt nhất hiện tại là **Champion**. Khi một mô hình mới (**Challenger**) được huấn luyện, nó bắt buộc phải bước qua Cổng kiểm định dựa trên **Bộ tứ tiêu chuẩn Time Series**:
> 1. **Test RMSE (Trọng số chính 50%):** Bắt buộc phải có xu hướng giảm sai số đỉnh đáy.
> 2. **Test MAE & MAPE (25%):** Đảm bảo sai số trung bình và tỷ lệ sai lệch không bị thoái hóa.
> 3. **AIC & BIC (15%):** Áp dụng nguyên lý tối giản Box-Jenkins (Parsimony). Nếu 2 mô hình có RMSE ngang nhau (chênh lệch $\le 1\%$), mô hình nào có AIC thấp hơn (ít tham số hơn) sẽ thắng để tránh Overfitting.
> 4. **Kiểm định White Noise của phần dư (10%):** Đảm bảo tính sạch tương quan.
>
> Nhóm em đã thực nghiệm giả lập chạy một mô hình Challenger kém là $\text{ARIMA}(0, 0, 1)$ trên tập `a10`. Ngay lập tức trên màn hình:
> - Test RMSE của Challenger vọt lên $11.75$ (kém hơn Champion tới $+154\%$).
> - AIC của Challenger lên tới $791$ (kém hơn Champion 209 điểm).
> - Cổng kiểm định lập tức ra phán quyết: `🛑 REJECTED`. Toàn bộ file model `.joblib` và 5 biểu đồ của Champion cũ được **bảo toàn nguyên vẹn 100%**. Đồng thời, lần thử nghiệm thất bại này vẫn được ghi nhận minh bạch vào tệp kiểm toán `audit_trail.jsonl`.
>
> Nhờ cơ chế này, thư mục `artifacts/` của chúng em luôn giữ kích thước cố định, gọn nhẹ, và luôn luôn đại diện cho phiên bản thông minh nhất!
>
> Em xin chân thành cảm ơn Thầy/Cô và các bạn đã chú ý lắng nghe. Em rất mong nhận được những câu hỏi và đóng góp quý báu từ Thầy/Cô ạ!"
