# 📘 CẨM NANG HƯỚNG DẪN ĐỌC HIỂU & PHÂN TÍCH BIỂU ĐỒ CHUỖI THỜI GIAN
## (Từ Bài tập 1 đến Bài tập 5 - Dataset: AirPassengers)

Tài liệu này được biên soạn nhằm giúp bạn **nắm vững bản chất lý thuyết**, **biết cách quan sát từng chi tiết trên đồ thị** và **tự tin giải thích thuyết phục** trong mọi bài kiểm tra hoặc thuyết trình.

---

# 📑 MỤC LỤC
1. [BÀI TẬP 1: Chuỗi thời gian gốc & Cặp đồ thị ACF / PACF](#-bài-tập-1-chuỗi-thời-gian-gốc--cặp-đồ-thị-acf--pacf)
2. [BÀI TẬP 2: Kiểm tra White Noise (Nhiễu trắng) vs Chuỗi thực tế](#-bài-tập-2-kiểm-tra-white-noise-nhiễu-trắng-vs-chuỗi-thực-tế)
3. [BÀI TẬP 3: Phân rã chuỗi thời gian (Decomposition)](#-bài-tập-3-phân-rã-chuỗi-thời-gian-decomposition)
4. [BÀI TẬP 4: Kiểm tra và xử lý Missing Values & Outliers](#-bài-tập-4-kiểm-tra-và-xử-lý-missing-values--outliers)
5. [BÀI TẬP 5: Trực quan hóa chuyên sâu (Rolling Stats, Seasonal Plot, Heatmap)](#-bài-tập-5-trực-quan-hóa-chuyên-sâu-rolling-stats-seasonal-plot-heatmap)

---

# 📌 BÀI TẬP 1: Chuỗi thời gian gốc & Cặp đồ thị ACF / PACF

---

## 1. Cơ sở lý thuyết trực quan (Dễ hiểu nhất)

### a. Khái niệm độ trễ (Lag)
* Trong chuỗi thời gian theo tháng, **Lag $k$** là khoảng cách lùi về quá khứ $k$ tháng:
  * **Lag 0:** Tháng hiện tại ($t$) so với chính nó (tương quan luôn = $1.0$).
  * **Lag 1:** Tháng hiện tại ($t$) so với tháng liền kề trước đó ($t-1$).
  * **Lag 12:** Tháng hiện tại ($t$) so với cùng kỳ năm ngoái ($t-12$).

### b. Sự khác nhau giữa ACF và PACF (Ẩn dụ gia đình)
* **ACF (Autocorrelation Function - Tự tương quan toàn phần):**
  * Đo lường mối liên hệ giữa $Y_t$ và $Y_{t-k}$ bao gồm **cả trực tiếp lẫn gián tiếp**.
  * *Ẩn dụ:* Chiều cao của **Cháu** giống chiều cao của **Ông nội**. Nhưng thực chất Ông truyền gen cho **Bố**, rồi Bố mới truyền gen cho **Cháu**. ACF ghi nhận tất cả mối quan hệ dây chuyền này.
* **PACF (Partial Autocorrelation Function - Tự tương quan riêng phần):**
  * Đo lường **tác động trực tiếp thuần túy** giữa $Y_t$ và $Y_{t-k}$ sau khi đã **loại trừ hoàn toàn** ảnh hưởng của tất cả các mắt xích trung gian ($Y_{t-1}, Y_{t-2}, \dots$).
  * *Ẩn dụ:* Đặt câu hỏi: *Nếu trừ hết ảnh hưởng của Bố ra, thì giữa Ông và Cháu còn có liên kết di truyền trực tiếp nào độc lập không?*

### c. Quy tắc dải bóng màu xanh (Ngưỡng tin cậy 95%)
* **Nằm lọt bên trong dải xanh:** Hệ số tương quan quá nhỏ $\rightarrow$ Coi như bằng 0 $\rightarrow$ **Là ngẫu nhiên, KHÔNG có tương quan.**
* **Vượt ra ngoài dải xanh:** Có ý nghĩa thống kê $\rightarrow$ **CÓ TƯƠNG QUAN THỰC SỰ.**

```
       +1.0 |
Hệ số       |   🔴 QUE THÒ RA NGOÀI = CÓ TƯƠNG QUAN THỰC SỰ
            |------------------------------------------- [Mép trên dải xanh]
        0.0 |   🟦 QUE NẰM TRONG VÙNG XANH = LÀ NHIỄU NGẪU NHIÊN (COI NHƯ = 0)
            |------------------------------------------- [Mép dưới dải xanh]
       -1.0 |
```

---

## 2. Phân tích Biểu đồ 1.1: Chuỗi thời gian gốc

![Chuỗi thời gian gốc](plots/bai1_01_raw_time_series.png)

### 🔍 Giải phẫu biểu đồ:
* **Trục hoành (X):** Thời gian liên tục từ năm 1949 đến năm 1960 (144 tháng).
* **Trục tung (Y):** Lượng hành khách bay quốc tế (đơn vị: nghìn người).

### 💡 Mắt thấy gì $\rightarrow$ Bản chất là gì?
1. **Đường dốc đứng lên trên (Xu hướng - Trend):**
   * *Mắt thấy:* Năm 1949 chỉ có ~112 nghìn khách, nhưng đến năm 1960 đã vọt lên ~600 nghìn khách.
   * *Bản chất:* Chuỗi có **Trend tăng trưởng dài hạn**, kỳ vọng trung bình không cố định $\rightarrow$ Đây là **chuỗi không dừng (Non-stationary)**.
2. **Sóng nhấp nhô lặp lại đều đặn (Mùa vụ - Seasonality):**
   * *Mắt thấy:* Mỗi năm đều vẽ một ngọn núi nhọn vào giữa năm (kỳ nghỉ hè) và tụt sâu vào cuối năm.
   * *Bản chất:* Chuỗi có chu kỳ lặp lại theo mùa hàng năm ($S = 12$ tháng).
3. **Biên độ sóng phình to dần (Multiplicative Pattern):**
   * *Mắt thấy:* Chiều cao ngọn sóng năm 1949 chỉ chênh lệch khoảng 30 nghìn khách, nhưng đến năm 1960 độ chênh lệch lên tới gần 200 nghìn khách.
   * *Bản chất:* Mức độ dao động mùa vụ tỷ lệ thuận với quy mô thị trường (phù hợp với mô hình dạng Nhân).

---

## 3. Phân tích Biểu đồ 1.2: Cặp đồ thị ACF & PACF (40 Lags)

![ACF và PACF 40 lags](plots/bai1_02_acf_pacf_40lags.png)

### 🔍 Giải phẫu đồ thị ACF (Đồ thị trên):
* **Trục hoành (Lags):** Độ trễ từ 0 đến 40 tháng.
* **Hình dạng dải xanh:** Khởi đầu ở $\pm 0.163$ tại Lag 1 và **nở rộng dần như hình chiếc loa kèn** (theo công thức sai số tích lũy Bartlett).
* **Hiện tượng mấu chốt:**
  1. **14 que đầu tiên (Lag 1 đến Lag 14) vượt ra ngoài dải xanh:** 
     * *Tại sao lại là 14 que?* Vì dải xanh nở to dần, tại Lag 14 ngưỡng phình lên $\pm 0.645$ (que đạt $0.646$ nên vừa vặn vượt ngưỡng). Từ Lag 15 phễu phình lên $\pm 0.662$ nên các que bắt đầu chui vào trong.
     * *Ý nghĩa:* Sự phụ thuộc kéo dài suốt **14 tháng liên tiếp** (hơn 1 năm 2 tháng) chứng minh chuỗi có **tự tương quan cực kỳ mạnh** và **xu hướng tăng rất bền vững**.
  2. **Suy giảm chậm (Slow Decay):** Các que không rớt về 0 ngay mà hạ dần từ tốn $\rightarrow$ Dấu hiệu nhận diện **chuỗi không dừng**.
  3. **Hình sin lượn sóng:** Đỉnh sóng nhô cao trở lại tại **Lag 12, 24, 36** $\rightarrow$ Dấu hiệu nhận diện **chu kỳ mùa vụ 12 tháng**.

### 🔍 Giải phẫu đồ thị PACF (Đồ thị dưới):
* **Lag 1 cao chót vót ($\approx 0.95$):** Vượt xa dải xanh $\rightarrow$ Tháng liền trước ($t-1$) có tác động trực tiếp áp đảo đến tháng hiện tại ($t$).
* **Từ Lag 2 đến Lag 11 chui tọt vào trong dải xanh:** Sau khi loại bỏ ảnh hưởng gián tiếp của Lag 1, các tháng xa hơn không còn tác động trực tiếp độc lập nữa.
* **Lag 12 và 13 hơi nhô nhẹ:** Dấu ấn trực tiếp còn sót lại của chu kỳ mùa năm trước.

---

# 📌 BÀI TẬP 2: Kiểm tra White Noise (Nhiễu trắng) vs Chuỗi thực tế

---

## 1. Cơ sở lý thuyết: White Noise là gì?
White Noise (Nhiễu trắng) là chuỗi các biến ngẫu nhiên độc lập thuần túy:
* Giống như tiếng "xè xè" của TV mất sóng hoặc việc tung đồng xu liên tục.
* **Đặc tính:** Hoàn toàn không có quy luật, không có trí nhớ (quá khứ không giúp đoán được tương lai).
* **ACF lý thuyết của White Noise:** Tại mọi Lag $k \ge 1$, hệ số tự tương quan $\rho_k = 0$.

---

## 2. Phân tích Biểu đồ Bài 2: Đối chiếu song song 4 ô

![So sánh White Noise và AirPassengers](plots/bai2_compare_white_noise_vs_airpassengers.png)

```
┌─────────────────────────────────┬─────────────────────────────────┐
│ Ô 1: Đường White Noise (N=1000) │ Ô 2: Đường AirPassengers        │
│ Dao động hỗn loạn quanh số 0    │ Dốc lên rõ rệt, có nhịp điệu    │
├─────────────────────────────────┼─────────────────────────────────┤
│ Ô 3: ACF của White Noise        │ Ô 4: ACF của AirPassengers      │
│ 100% que lọt thỏm trong dải xanh│ 14 que đầu vượt ra ngoài dải    │
└─────────────────────────────────┴─────────────────────────────────┘
```

### 💡 Hướng dẫn so sánh & Đối chiếu:

| Tiêu chí | Ô 3: ACF của White Noise | Ô 4: ACF của AirPassengers | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- | :--- |
| **Vị trí các que** | **Toàn bộ que ($Lag \ge 1$) nằm gọn trong dải xanh 95%** | **14 que đầu vượt ra ngoài dải xanh** | White Noise là ngẫu nhiên vô nghĩa; AirPassengers có quy luật phụ thuộc chặt chẽ. |
| **Tốc độ suy giảm** | Rơi thẳng đứng về 0 ngay sau Lag 0 (*Instant Drop*) | Suy giảm rất từ từ (*Slow Decay*) | White Noise không có xu hướng; AirPassengers có xu hướng tăng dài hạn. |
| **Dạng hình học** | Phẳng lặng, nhấp nhô ngẫu nhiên quanh trục 0 | Lượn sóng hình sin đỉnh chu kỳ 12 | White Noise không có mùa vụ; AirPassengers có mùa vụ rõ nét. |
| **Khả năng dự báo** | **Không thể dự báo** (Dự đoán tốt nhất chỉ là lấy trung bình = 0) | **Dự báo rất tốt** (Vì có cấu trúc tự tương quan sâu sắc) | Đủ điều kiện để áp dụng các mô hình chuỗi thời gian như SARIMA. |

---

# 📌 BÀI TẬP 3: Phân rã chuỗi thời gian (Decomposition)

---

## 1. Cơ sở lý thuyết: Bóc tách chuỗi thành 3 thành phần

Bất kỳ chuỗi thời gian kinh tế nào cũng có thể xem là sự kết hợp của 3 nhân tố:
1. **Trend ($T_t$):** Xu hướng phát triển dài hạn.
2. **Seasonal ($S_t$):** Dao động mùa vụ lặp đi lặp lại hàng năm.
3. **Residual ($R_t$):** Phần dư, sai số ngẫu nhiên còn lại.

### ❓ Tại sao chọn mô hình Nhân (`Multiplicative`) thay vì mô hình Cộng (`Additive`)?
* **Mô hình Cộng:** $Y_t = T_t + S_t + R_t$ $\rightarrow$ Giả định độ cao ngọn sóng mùa hè là cố định (ví dụ luôn tăng đúng 50 khách mỗi năm).
* **Mô hình Nhân:** $Y_t = T_t \times S_t \times R_t$ $\rightarrow$ Giả định mùa hè tăng theo **tỷ lệ phần trăm** (ví dụ luôn tăng thêm 25% so với mức trung bình năm).
* 👉 **Với AirPassengers:** Vì ngọn sóng mùa hè năm 1960 cao hơn nhiều so với năm 1949 $\rightarrow$ **Bắt buộc dùng mô hình Nhân**.

---

## 2. Phân tích Biểu đồ 3.1: Bốn tầng phân rã

![Phân rã chuỗi thời gian](plots/bai3_01_decomposition_multiplicative.png)

### 🔍 Giải phẫu từng tầng đồ thị:
1. **Tầng 1 - Observed ($Y_t$):**
   * Đồ thị gốc ban đầu, chứa đựng tất cả sự phức tạp đan xen.
2. **Tầng 2 - Trend ($T_t$):**
   * Trích xuất bằng kỹ thuật trung bình trượt 12 tháng.
   * *Quan sát:* Một đường cong mượt mà, dốc đều từ 120 lên 450 nghìn khách. Loại bỏ hoàn toàn mọi nhấp nhô mùa vụ.
3. **Tầng 3 - Seasonal ($S_t$):**
   * Trong mô hình nhân, giá trị mùa vụ là các **hệ số dao động quanh mức 1.0 (tức 100%)**:
     * **Đỉnh tháng 7 và 8:** Hệ số đạt khoảng **1.25** $\rightarrow$ Cao hơn mức trung bình năm 25%.
     * **Đáy tháng 11:** Hệ số tụt xuống **0.80** $\rightarrow$ Thấp hơn mức trung bình năm 20%.
4. **Tầng 4 - Residual ($R_t = \frac{Y_t}{T_t \times S_t}$):**
   * Phần còn lại sau khi lấy dữ liệu gốc chia cho (Trend $\times$ Seasonal).
   * *Quan sát:* Dao động ngẫu nhiên quanh đường cơ sở **1.0**. Biên độ rất hẹp (chủ yếu từ 0.95 đến 1.05, tức sai số chỉ $\pm 5\%$).

---

## 3. Phân tích Biểu đồ 3.2: Kiểm tra ACF của Residual

![ACF của Residual](plots/bai3_02_residual_acf.png)

### 💡 Hướng dẫn đọc & Ý nghĩa sống còn:
* **Mắt thấy gì?** Hầu như **toàn bộ các que ACF của Residual đều nằm lọt thỏm bên trong dải màu xanh 95%**!
* **Bản chất toán học:** Residual lúc này đã trở thành **White Noise** (nhiễu trắng ngẫu nhiên).
* **Ý nghĩa:** Mô hình phân rã đã "bóc sạch" mọi quy luật (Trend và Seasonality). Không còn bất kỳ quy luật nào bị bỏ sót hay ẩn nấp trong phần dư. Phân rã đã thành công mỹ mãn!

---

# 📌 BÀI TẬP 4: Kiểm tra và xử lý Missing Values & Outliers

---

## 1. Cơ sở lý thuyết về chất lượng dữ liệu

### a. Xử lý dữ liệu thiếu (Missing Value):
* **Forward-fill (`ffill`):** Lấy giá trị của tháng trước đắp vào tháng bị thiếu. 
  * *Nhược điểm:* Nếu chuỗi đang dốc đứng đi lên, nó sẽ tạo ra những đoạn "bậc thang nằm ngang" giả tạo, bóp méo hình dạng mùa vụ.
* **Time Interpolation (Nội suy thời gian):** Kẻ một đường nối mượt mà giữa mốc thời gian trước và sau dựa trên tỷ lệ thời gian. Giữ trọn độ dốc và biên độ tự nhiên.

### b. Phát hiện Outlier bằng Rolling Window:
* Vì chuỗi có xu hướng tăng từ 100 lên 600, ta **không thể dùng ngưỡng cố định** cho toàn bộ chuỗi.
* **Giải pháp:** Dùng ngưỡng động trượt theo từng năm:
  $$\text{Upper Bound} = \text{Rolling Mean} + 2.5 \times \text{Rolling Std}$$
  $$\text{Lower Bound} = \text{Rolling Mean} - 2.5 \times \text{Rolling Std}$$

---

## 2. Phân tích Biểu đồ Bài 4: So sánh 3 giai đoạn xử lý

![Xử lý dữ liệu](plots/bai4_data_cleaning_missing_outliers.png)

### 🔍 Giải phẫu từng đồ thị:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Dữ liệu lỗi: Vạch cam đứt nét (NaN) + 2 chấm đỏ to (Outlier) │
├─────────────────────────────────────────────────────────────────┤
│ 2. So sánh: Đường cam (ffill - bậc thang) vs Xanh (Interpolation)│
├─────────────────────────────────────────────────────────────────┤
│ 3. Dữ liệu sạch: Bám khít 100% vào chuỗi gốc (Ground Truth)     │
└─────────────────────────────────────────────────────────────────┘
```

1. **Đồ thị 1 (Dữ liệu lỗi ban đầu):**
   * 10 vạch cam đứt đoạn: Vị trí bị mất dữ liệu.
   * 2 chấm tròn đỏ nổi bật: Điểm Outlier giả lập (tháng 10/1952 tăng vọt lên ~700 và tháng 02/1956 rơi tụt xuống 20).
2. **Đồ thị 2 (So sánh Ffill vs Interpolation):**
   * **Đường màu cam (ffill):** Bị "gãy khúc", tạo ra các đoạn phẳng lì nằm ngang.
   * **Đường màu xanh lá (Interpolation):** Điểm điền khuyết lượn theo sườn dốc tự nhiên của chuỗi.
3. **Đồ thị 3 (Chuỗi sạch sau khi làm sạch hoàn toàn):**
   * Sau khi thay Outlier bằng NaN rồi áp dụng Time Interpolation, đường màu xanh dương (đã làm sạch) trùng khít gần như hoàn hảo với đường nét chấm đen (dữ liệu thực tế ban đầu).

---

# 📌 BÀI TẬP 5: Trực quan hóa chuyên sâu (Rolling Stats, Seasonal Plot, Heatmap)

---

## 1. Phân tích Biểu đồ 5.1: Rolling Mean & Rolling Std (Window = 12)

![Rolling Statistics](plots/bai5_01_rolling_statistics.png)

### 🔍 Giải phẫu biểu đồ:
* **Đường xanh dương mờ:** Dữ liệu gốc hàng tháng.
* **Đường đỏ đậm (Rolling Mean 12 tháng):** Trung bình trượt của 12 tháng liên tiếp.
  * *Quan sát:* Triệt tiêu hoàn toàn các ngọn sóng mùa hè để lộ ra một đường dốc thẳng tắp từ 120 lên 450.
  * *Ý nghĩa:* Khẳng định xu hướng tăng trưởng bền vững; vi phạm tính dừng về kỳ vọng trung bình.
* **Đường xanh lá nét đứt (Rolling Std 12 tháng):** Độ lệch chuẩn trượt của 12 tháng.
  * *Quan sát:* Không nằm ngang mà dốc dần lên từ mức ~15 (năm 1949) lên tới ~45 (năm 1960).
  * *Ý nghĩa:* Độ biến động (phương sai) tăng gấp 3 lần theo thời gian $\rightarrow$ Vi phạm tính dừng về phương sai (*Heteroscedasticity*).

---

## 2. Phân tích Biểu đồ 5.2: Seasonal Plot theo tháng qua các năm

![Seasonal Plot](plots/bai5_02_seasonal_plot_monthly.png)

### 🔍 Giải phẫu biểu đồ:
* **Trục hoành:** 12 tháng (từ T1 đến T12).
* **Trục tung:** Lượng khách (nghìn người).
* **Mỗi đường màu:** Đại diện cho một năm (từ màu tím năm 1949 đến màu vàng năm 1960).

### 💡 Hiện tượng mắt thấy $\rightarrow$ Ý nghĩa:
1. **Các đường chạy song song và cùng dạng hình học:**
   * Mọi năm đều có hình "lưng lạc đà": hơi nhô ở tháng 3, chững lại ở tháng 4-5, **bùng nổ vút lên đỉnh nhọn ở tháng 7 và tháng 8**, rồi **rơi tụt xuống đáy sâu ở tháng 11**.
   * *Ý nghĩa:* Quy luật tâm lý và hành vi du lịch hè của hành khách là nhất quán tuyệt đối qua hơn 1 thập kỷ.
2. **Các đường năm sau luôn nằm đè lên trên năm trước:**
   * Đường năm 1960 nằm ở tầng cao nhất, đường năm 1949 nằm ở tầng đáy.
   * *Ý nghĩa:* Minh chứng trực quan cho sự tăng trưởng quy mô liên tục qua từng năm.

---

## 3. Phân tích Biểu đồ 5.3: Heatmap Seasonality (Tháng $\times$ Năm)

![Heatmap Seasonality](plots/bai5_03_heatmap_seasonality.png)

### 🔍 Giải phẫu bản đồ nhiệt:
* **Trục hoành:** Các năm từ 1949 đến 1960.
* **Trục tung:** 12 tháng trong năm.
* **Quy ước màu:** Màu vàng nhạt (khách ít) $\rightarrow$ Màu xanh ngọc (khách trung bình) $\rightarrow$ Màu xanh đậm đen (khách cực đông). Con số ghi trong từng ô là lượng khách thực tế.

### 💡 Cách đọc bản đồ nhiệt:
1. **Đọc theo cột dọc (Thời gian trôi qua từng năm):**
   * Màu sắc chuyển dần từ tông vàng nhạt (bên trái) sang tông xanh đậm (bên phải) $\rightarrow$ Thấy ngay sự phát triển thần tốc của ngành hàng không.
2. **Đọc theo hàng ngang (Các tháng trong năm):**
   * **Hàng T7 và T8:** Luôn sở hữu những ô màu xanh đậm nhất trong năm (đỉnh cao là 622 và 606 nghìn khách vào tháng 7, 8 năm 1960).
   * **Hàng T11:** Luôn sở hữu ô màu nhạt nhất của năm đó.

---

# 🎓 TỔNG KẾT BẢNG TRA CỨU NHANH KHI BẢO VỆ BÀI

| Đồ thị | Hiện tượng mấu chốt quan sát được | Kết luận rút ra |
| :--- | :--- | :--- |
| **Line plot gốc** | Đường dốc lên, sóng nhấp nhô, biên độ mở rộng | Chuỗi có Trend tăng, có Seasonal 12 tháng, phù hợp mô hình Nhân. |
| **ACF 40 lags** | 14 que đầu vượt ra ngoài dải phễu Bartlett; hình sin chu kỳ 12 | Tự tương quan cực mạnh, không dừng (Non-stationary), mùa vụ hàng năm rõ nét. |
| **PACF 40 lags** | Duy nhất Lag 1 cao chót vót (~0.95), các lag sau tắt nhanh | Tháng trước chi phối trực tiếp tháng sau; dấu hiệu của mô hình tự hồi quy AR(1). |
| **White Noise ACF** | 100% các que nằm lọt thỏm trong dải xanh 95% | Dữ liệu ngẫu nhiên thuần túy, không có trí nhớ, không thể dự báo. |
| **Decomposition** | Trend dốc đều; Seasonal đỉnh T7-8 (+25%), đáy T11 (-20%); Residual dao động quanh 1.0 và đạt tính chất White Noise | Mô hình phân rã bóc tách triệt để quy luật, chuỗi rất lý tưởng để dự báo bằng SARIMA / Holt-Winters. |
| **Cleaning Data** | Interpolation mượt hơn Ffill; Rolling 2.5*Std bắt trúng 2 Outlier | Giữ trọn vẹn nhịp sóng của chuỗi thời gian khi làm sạch. |
| **Rolling Stats** | Rolling Mean dốc lên, Rolling Std nở rộng từ 15 lên 45 | Vi phạm tính dừng về cả trung bình lẫn phương sai. |
| **Seasonal & Heatmap** | Các đường song song; dải màu đậm nhất ở T7-8, nhạt nhất ở T11 | Quy luật mùa hè bùng nổ diễn ra đều đặn, không có đứt gãy bất thường. |
