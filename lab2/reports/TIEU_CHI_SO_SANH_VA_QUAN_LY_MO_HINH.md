# BỘ TIÊU CHÍ SO SÁNH & QUẢN TRỊ MÔ HÌNH CHUỖI THỜI GIAN (CHAMPION VS. CHALLENGER)

**Hệ thống:** Model Quality Gate & Champion-Challenger Promotion  
**Áp dụng:** Time Series Production Pipeline (Lab 2)

---

## 1. VÌ SAO PHẢI CÓ CƠ CHẾ SO SÁNH TRƯỚC KHI GHI ĐÈ?

Trong quy trình phát triển mô hình chuỗi thời gian (Time Series Modeling), các nhà phân tích liên tục thử nghiệm các bộ tham số $(p, d, q)$ và $(P, D, Q)_s$ khác nhau. 

Nếu áp dụng cơ chế **ghi đè mù quáng (Blind Overwrite)**:
- Mỗi khi bạn thử một mô hình kém hơn (ví dụ tăng bậc quá đà gây Overfitting hoặc chọn sai bậc mùa vụ), mô hình kém này sẽ **xóa sạch mô hình tốt nhất trước đó**.
- Bạn mất đi cột mốc đối chuẩn (*Benchmark*) và có nguy cơ đưa mô hình sai lệch vào sử dụng.

Ngược lại, nếu lưu **mọi phiên bản vào từng file riêng biệt**:
- Ổ đĩa sẽ nhanh chóng bị phình to bởi các file nhị phân mô hình (`.joblib` nặng từ vài trăm KB đến 16MB/file) và hàng chục bức ảnh đồ thị dư thừa.

👉 **Giải pháp tối ưu chuẩn MLOps: Cơ chế Phân Tách Thư Mục Champions & Challengers ($O(1)$ File Space)**:
- **Cấu trúc lưu trữ phân cấp rõ ràng**:
  - `artifacts/figures/eda/` : Chứa toàn bộ ảnh phân tích thăm dò ban đầu.
  - `artifacts/figures/champions/` vs `artifacts/figures/challengers/`
  - `artifacts/models/champions/` vs `artifacts/models/challengers/`
  - `artifacts/metrics/champions/` vs `artifacts/metrics/challengers/`
- **Quy trình hoạt động**:
  1. Khi một cấu hình mới được chạy, hệ thống **luôn luôn xuất bản tài sản của lần chạy đó vào các thư mục `challengers/`** (ghi đè lên Challenger cũ).
  2. Cổng kiểm duyệt thống kê (**Model Evaluation Gate**) so sánh Challenger với Champion:
     - **Nếu Challenger THẮNG (PROMOTED)**: Thăng hạng làm Champion mới, cập nhật sang các thư mục `champions/`.
     - **Nếu Challenger THUA (REJECTED)**: Giữ nguyên vẹn tài sản trong `champions/`, nhưng **VẪN GIỮ NGUYÊN TÀI SẢN TRONG `challengers/`**.
  3. **Lợi ích thực tế cho Data Scientist**:
     - Bạn chỉ cần mở đồng thời 2 folder `figures/champions/` và `figures/challengers/` cạnh nhau để so sánh trực quan (*Side-by-side Visual Inspection*), xem mô hình mới dự báo lệch ở đỉnh nào, hụt ở đáy nào để biết hướng tinh chỉnh tham số tiếp theo.
     - Thư mục tuyệt đối ngăn nắp, không bị lẫn lộn giữa ảnh EDA, mô hình cũ và mô hình mới; tổng số lượng file luôn cố định ($O(1)$).
- Mọi lịch sử thử nghiệm (kể cả thất bại) vẫn được lưu vết bất biến trong `pipeline.log` và `audit_trail.jsonl`.

---

## 2. BỐN TIÊU CHÍ SO SÁNH CỐT LÕI CỦA MÔN HỌC TIME SERIES

Hệ thống chấm điểm đa tiêu chí dựa trên 4 trụ cột kinh tế lượng:

### 🏆 Tiêu chí 1: Test RMSE (Root Mean Squared Error) — *Trọng số cao nhất (50%)*
- **Công thức:**
  $$\text{RMSE} = \sqrt{\frac{1}{H} \sum_{t=1}^{H} (y_{T+t} - \hat{y}_{T+t})^2}$$
- **Ý nghĩa & Lý do chọn:**
  - Trong dự báo chuỗi thời gian ngoài mẫu (*Out-of-sample Forecast*), sai số ở các giai đoạn cực trị (đỉnh mùa hè tháng 7-8 của ngành hàng không, hoặc đỉnh mùa đông tháng 12 của dược phẩm) gây tổn thất kinh tế rất lớn.
  - Phép toán bình phương sai số của RMSE **phạt cực nặng các sai số lớn (Penalize Large Outliers)**. Mô hình nào có RMSE thấp hơn chứng tỏ khả năng bao quát và kiểm soát rủi ro đỉnh đáy tốt hơn.

---

### 📏 Tiêu chí 2: Test MAE & MAPE (Độ lệch tuyệt đối & Phần trăm sai số) — *Trọng số 25%*
- **Công thức:**
  $$\text{MAE} = \frac{1}{H} \sum_{t=1}^{H} |y_{T+t} - \hat{y}_{T+t}|, \quad \text{MAPE} = \frac{100\%}{H} \sum_{t=1}^{H} \left|\frac{y_{T+t} - \hat{y}_{T+t}}{y_{T+t}}\right|$$
- **Ý nghĩa & Lý do chọn:**
  - **MAE** phản ánh sai số trung bình thực tế ở đúng đơn vị đo lường ban đầu (triệu USD hoặc nghìn hành khách), không bị méo mó bởi một vài quan sát ngoại lai.
  - **MAPE** chuẩn hóa sai số về tỷ lệ phần trăm (%), cho phép người quản lý biết mô hình đang sai lệch bao nhiêu % so với quy mô thực tế. Mô hình mới cần có xu hướng cải thiện đồng thời cả MAE và MAPE.

---

### ⚖️ Tiêu chí 3: AIC & BIC (Tiêu chuẩn thông tin & Nguyên lý Tối giản Occam's Razor) — *Trọng số 15%*
- **Công thức:**
  $$\text{AIC} = 2k - 2\ln(\hat{L}), \quad \text{BIC} = k\ln(N) - 2\ln(\hat{L})$$
  *(trong đó $k$ là số tham số ước lượng, $\hat{L}$ là hàm hợp lý cực đại, $N$ là số quan sát)*
- **Ý nghĩa & Lý do chọn (Cực kỳ quan trọng trong phương pháp Box-Jenkins):**
  - **Nguyên lý Tối giản (Principle of Parsimony):** Trong kinh tế lượng chuỗi thời gian, một mô hình có ít tham số hơn luôn được ưa chuộng hơn nếu nó đạt hiệu năng tương đương, vì mô hình nhiều tham số dễ bị hiện tượng học vẹt nhiễu (*Overfitting*).
  - **Vai trò làm trọng tài hòa (Tie-breaker):** Nếu hai mô hình cho ra Test RMSE xấp xỉ nhau (chênh lệch $\le 1\%$), hệ thống sẽ xét đến AIC/BIC. Mô hình nào có AIC thấp hơn (tối giản hơn) sẽ thắng cuộc.

---

### 🔍 Tiêu chí 4: Kiểm định Tính White Noise của Phần dư (Ljung-Box Test) — *Trọng số 10%*
- **Bản chất:**
  - Mục tiêu tối thượng của mô hình ARIMA/SARIMA là trích xuất sạch toàn bộ cấu trúc phụ thuộc thời gian (xu hướng, tự hồi quy, mùa vụ), biến phần dư $\epsilon_t$ thành **chuỗi nhiễu trắng thuần túy (White Noise)** với kỳ vọng bằng 0 và không còn tự tương quan:
    $$\rho_k(\epsilon) = 0, \quad \forall k \ge 1$$
  - Nếu mô hình A có RMSE tương đương mô hình B, nhưng mô hình B triệt tiêu được tự tương quan ở các bậc mùa vụ (như SARIMA), mô hình B có giá trị thống kê bền vững hơn.

---

## 3. LUẬT PHÂN XỬ & RA QUYẾT ĐỊNH CỦA PIPELINE (DECISION GATE)

Hệ thống tự động thực hiện thuật toán phân xử như sau:

| Tình huống thực nghiệm | Kết quả phân xử | Hành vi của Pipeline |
| :--- | :---: | :--- |
| **Lần chạy đầu tiên** (chưa có Champion) | `INITIAL_CHAMPION` | Lưu model, metrics và xuất bản bộ ảnh đầu tiên làm mốc chuẩn. |
| **Challenger có $\text{RMSE} < \text{RMSE}_{\text{champ}}$** | `PROMOTED_SUPERIOR_FORECAST` | **Thăng hạng**: Ghi đè file model `.joblib`, cập nhật `metrics.json` và ghi đè 5 biểu đồ `.png`. |
| **Challenger hòa $\text{RMSE}$ ($\Delta \le 1\%$) nhưng $\text{AIC} < \text{AIC}_{\text{champ}}$** | `PROMOTED_PARSIMONIOUS` | **Thăng hạng**: Mô hình mới gọn nhẹ hơn, ít tham số hơn mà hiệu năng tương đương. |
| **Challenger hòa $\text{RMSE}$ nhưng $\text{AIC} \ge \text{AIC}_{\text{champ}}$** | `REJECTED_PARSIMONY` | **Từ chối**: Giữ nguyên Champion vì mô hình mới phức tạp hơn không cần thiết. |
| **Challenger có $\text{RMSE} > \text{RMSE}_{\text{champ}}$** | `REJECTED_INFERIOR_ACCURACY` | **Từ chối**: Bỏ qua ghi đè model và biểu đồ; giữ nguyên Champion. |

---

## 4. TỔNG KẾT

Cơ chế này bảo đảm 3 mục tiêu vàng của một hệ thống xử lý chuỗi thời gian chuyên nghiệp:
1. **Chất lượng**: Artifacts luôn đại diện cho phiên bản mô hình thông minh nhất.
2. **Tiết kiệm tài nguyên**: Không làm rác và phình to ổ đĩa.
3. **Minh bạch**: Người dùng luôn biết chính xác vì sao một mô hình được chọn hay bị loại thông qua bảng so sánh hiển thị trực quan trong log.
