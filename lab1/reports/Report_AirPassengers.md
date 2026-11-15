# Báo cáo phân tích khám phá dữ liệu (EDA)
## AirPassengers: Số hành khách hàng không quốc tế theo tháng (1949–1960)

**Phạm vi báo cáo.** Dữ liệu gồm 144 tháng, được chia thành tập huấn luyện (train, 132 tháng, từ 1949-01 đến 1959-12) và tập kiểm tra (test, 12 tháng của năm 1960). Mọi phân tích từ Giai đoạn 3 trở đi chỉ dùng train; tập test chưa được sử dụng.
---

## GIAI ĐOẠN 1: XÁC ĐỊNH BÀI TOÁN

### 1.1 Mục tiêu nghiên cứu

| Nội dung | Chi tiết |
|---|---|
| **Loại bài toán** | Dự báo chuỗi thời gian đơn biến, với một chuỗi dữ liệu duy nhất |
| **Câu hỏi nghiệp vụ** | “Từ dữ liệu lịch sử tháng 1/1949 đến tháng 12/1959, liệu có thể dự báo chính xác số hành khách hàng không quốc tế theo tháng trong 12 tháng tiếp theo hay không?” |
| **Chỉ số đánh giá chính** | **MAPE**: sai số theo tỷ lệ phần trăm, cho phép so sánh giữa các tháng có lượng hành khách chênh lệch lớn, chẳng hạn 104 nghìn ở tháng thấp nhất và 622 nghìn ở tháng cao nhất của toàn bộ dữ liệu. |
| **Chỉ số đánh giá bổ sung** | **RMSE**: đặt trọng số lớn hơn vào các sai số tuyệt đối lớn, cùng đơn vị với dữ liệu gốc (nghìn hành khách). |

**Tổn thất do dự báo sai:**

| Loại sai số | Hệ quả đối với hoạt động kinh doanh |
|---|---|
| Dự báo cao hơn thực tế | Hãng bố trí nhiều máy bay, tổ bay và nhân viên mặt đất hơn nhu cầu, chi phí vận hành tăng nhưng doanh thu không tăng tương ứng. |
| Dự báo thấp hơn thực tế | Năng lực phục vụ không đủ ở giai đoạn cao điểm, dẫn đến mất cơ hội bán vé, đặt chỗ vượt số ghế và khách chuyển sang hãng cạnh tranh. |

> Dự báo thấp hơn thực tế thường gây tổn thất lớn hơn dự báo cao hơn thực tế, vì doanh thu bị mất khó bù đắp hơn chi phí của phần năng lực dư. MAPE và RMSE đối xứng nên không phản ánh sự chênh lệch này. Khi triển khai thực tế, có thể dùng hàm mất mát có trọng số, chẳng hạn MAPE bất đối xứng với tỷ lệ 2:1 cho sai số dự báo thấp.

---

### 1.2 Tổng thể nghiên cứu, phạm vi dữ liệu và tính đại diện

| Nội dung | Thông tin |
|---|---|
| **Tổng thể nghiên cứu** | Lượng hành khách hàng không quốc tế theo tháng |
| **Phạm vi địa lý** | Nguồn gốc không nêu rõ. Bộ dữ liệu được dẫn từ tài liệu của Box, Jenkins và Reinsel, và thường được cho là dữ liệu của Pan American World Airways. |
| **Thời gian** | Tháng 1/1949 đến tháng 12/1960, gồm 12 năm đầy đủ |
| **Tần suất** | Hằng tháng |
| **Số quan sát** | 144 (train 132, test 12) |
| **Đơn vị đo** | Nghìn hành khách |
| **Nguồn dữ liệu** | Bộ dữ liệu `AirPassengers` trong gói `datasets` của R |

---

### 1.3 Đơn vị quan sát và mức độ chi tiết của dữ liệu

| Nội dung | Thông tin |
|---|---|
| **Raw data grain** | Tổng số hành khách hàng không quốc tế trong một tháng dương lịch |
| **Observation unit** | Một tháng dương lịch |
| **Uniqueness key** | Chỉ số thời gian, được biểu diễn bằng năm có phần thập phân |
| **Analytical grain** | Một quan sát cho mỗi tháng dương lịch; giữ nguyên raw data grain |

**Cách biểu diễn thời gian:** cột `time` gốc mã hóa thời gian bằng năm có phần thập phân. Với tần suất tháng, hai quan sát liên tiếp cách nhau $\frac{1}{12} \approx 0.0833$.

| Giá trị `time` gốc | Thời gian tương ứng |
|---|---|
| 1949.000 | Tháng 1/1949 |
| 1949.083 | Tháng 2/1949 |
| 1949.167 | Tháng 3/1949 |

**Sự phụ thuộc theo thời gian của dữ liệu (số liệu tính trên train):**

| Khía cạnh | Đánh giá |
|---|---|
| **Tự tương quan** | Rất mạnh. Hệ số tự tương quan ở độ trễ 1 là 0.94, và có thêm một đỉnh nhỏ ở độ trễ 12 (mục 4.1). |
| **Tính mùa vụ** | Chu kỳ 12 tháng. Tháng 7 và tháng 8 có số hành khách cao hơn khoảng 23% so với trung bình năm; tháng 11, tháng 1 và tháng 2 thấp hơn 14–17% (mục 3.2). |
| **Hệ quả** | **Không phù hợp** khi dùng K-fold cross validation với cách chia ngẫu nhiên. Dữ liệu phải được chia theo thời gian để giữ thứ tự và tránh rò rỉ thông tin (mục 2.7). |

---

### 1.4 Xác định biến mục tiêu và thông tin được phép dùng khi dự báo

| Nội dung | Thông tin |
|---|---|
| **Biến mục tiêu** | `value`, được đổi tên thành `Passengers`, đơn vị nghìn hành khách mỗi tháng |
| **Khoảng thời gian dự báo** | $h = 12$ tháng |

| Đặc trưng | Có sẵn tại thời điểm dự báo? | Ghi chú |
|---|---|---|
| Lịch sử `Passengers` đến thời điểm $t-1$ | Có | Dùng để tạo các biến trễ và thống kê trên cửa sổ trượt |
| Thông tin lịch, như tháng và năm | Có | Biểu diễn mùa vụ và vị trí trên xu hướng |
| `Passengers` tại thời điểm $t$ | Không | Đây là biến mục tiêu, không được dùng làm đầu vào |

---

## GIAI ĐOẠN 2: TÌM HIỂU DỮ LIỆU

### 2.1 Khảo sát cấu trúc dữ liệu

| Nội dung | Thông tin |
|---|---|
| Shape | (144, 3): 144 dòng × 3 cột |
| Memory (RAM) | 3.584 bytes, khoảng 3,5 KB |
| Date range | Từ 1949-01-01 đến 1960-12-01 |
| Null values | 0 trên tất cả các cột |
| Duplicate timestamps | 0 |

**Bảng thống kê:**

| Thống kê | Ý nghĩa |
|---|---|
| `count` = 144 ở cả 3 cột | Không có giá trị NaN, số dòng đúng như kỳ vọng (12 năm × 12 tháng). Tuy nhiên, nó chưa đảm bảo không bị thiếu tháng |
| `value`: min 104, max 622 | Mọi giá trị dương|
| `time`: min 1949.000, max 1960.917 | Phạm vi đúng từ 1/1949 đến 12/1960 (cộng thêm 1/12 sẽ ra 1961.000) |
| `time` và `rownames`: mean bằng median; std của `time` bằng std của `rownames` chia 12 | `time = 1949 + (rownames − 1)/12`: các mốc thời gian cách đều đúng 1 tháng|

Các chỉ số này không xét đến thứ tự thời gian.  Vì dữ liệu thay đổi theo thời gian, bảng thống kê tổng hợp cho 12 năm đã làm mất thông tin về xu hướng và mùa vụ.


### 2.2 Mô tả các cột dữ liệu gốc

| Cột | Mô tả | Kiểu dữ liệu | Vai trò | Cách xử lý |
|---|---|---|---|---|
| `rownames` | Số thứ tự của quan sát, từ 1 đến 144. | `int64` | Không có vai trò phân tích | **Loại bỏ** |
| `time` | Thời điểm quan sát, biểu diễn năm và phần lẻ của năm | `float64` | Nguồn để tạo mốc thời gian | **Thay thế** bằng cột `Date` kiểu ngày giờ |
| `value` | Tổng số hành khách hàng không quốc tế trong tháng | `int64` | **Biến mục tiêu** | **Đổi tên** thành `Passengers` |

### 2.3 Kiểm tra kiểu dữ liệu và chuyển đổi thời gian

Cột `time` có kiểu `float64`, chưa phù hợp để pandas thực hiện các thao tác theo thời gian như đặt tần suất, tính độ trễ hay phân rã mùa vụ. `pd.to_datetime(df['time'])` không diễn giải `1949.083333` theo quy ước “năm có phần thập phân”, nên cần tách năm và tháng trước khi tạo cột ngày giờ:

```python
df['Year']  = np.floor(df['time']).astype(int)       # 1949.083 → 1949
df['Month'] = np.round((df['time'] - df['Year']) * 12 + 1).astype(int)  # 0.083 → 2
df['Date']  = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
```

> Phải dùng `np.round()` khi tính tháng. Do sai số số thực, phép tính có thể trả về `1.9999999`; nếu chuyển thẳng sang số nguyên, kết quả bị cắt xuống tháng 1 thay vì tháng 2.

| Nội dung kiểm tra bằng `assert` | Kết quả mong đợi | Kết quả thực tế |
|---|---|---|
| Ngày bắt đầu | 1949-01-01 | Không sinh lỗi |
| Ngày kết thúc | 1960-12-01 | Không sinh lỗi |
| Ngày trùng lặp | 0 | Không sinh lỗi |
| Tháng bị thiếu | 0 | Không sinh lỗi, đối chiếu với `pd.date_range('1949-01-01', '1960-12-01', freq='MS')` |

### 2.4 Vai trò của các biến sau khi làm sạch

| Cột | Vai trò | Ghi chú |
|---|---|---|
| `Date` | **Mốc thời gian / index** | Đặt làm index và gán tần suất `MS` bằng `asfreq('MS')` |
| `Passengers` | **Biến mục tiêu** | |
| `rownames` | Đã loại bỏ | |
| `time` | Đã loại bỏ | Thay bằng `Date` |

### 2.5 Đánh giá quy mô dữ liệu phục vụ forecastingu



### 2.6 Kiểm tra chất lượng cơ bản

| Nội dung kiểm tra | Kết quả | Cách xử lý |
|---|---|---|
| Giá trị thiếu | 0 | Không cần xử lý |
| Dòng trùng lặp | 0 | Không cần xử lý |
| Ngày trùng lặp | 0 | Không cần xử lý |
| Khoảng giá trị | [104, 622] | Không có giá trị $\leq 0$, có thể áp dụng logarit |
| Đã sắp xếp theo thời gian | Có | Không cần xử lý |
| Tần suất suy ra từ dữ liệu | `MS` | Các mốc thời gian cách đều |

Các kiểm tra này chỉ xác nhận tính toàn vẹn của dữ liệu, không dẫn đến quyết định mô hình hóa nào, nên được chạy trên cả 144 dòng.

### 2.7 Chia tập huấn luyện và tập kiểm tra

| Tập | Giai đoạn | Số quan sát | Vai trò |
|---|---|---|---|
| Train | 1949-01 đến 1959-12 | 132 | Khám phá dữ liệu, chọn biến đổi và bậc sai phân, ước lượng mô hình |
| Test | 1960-01 đến 1960-12 | 12 | Chỉ dùng một lần để chấm điểm dự báo |

Việc chọn logarit, bậc sai phân $d$, $D$ và ngưỡng phát hiện ngoại lệ đều là các quyết định rút ra từ dữ liệu. Nếu chúng dựa trên cả năm 1960 thì điểm đánh giá trên năm 1960 sẽ lạc quan hơn thực tế. Vì vậy các quyết định này chỉ dựa trên train.

---

## GIAI ĐOẠN 3: QUAN SÁT CHUỖI

### 3.1 Chuỗi gốc

- Trend: hướng dịch chuyển dài hạn của dữ liệu xuyên suốt thời gian, bỏ qua các biến động ngắn hạn. Xu hướng có thể là đi lên (tăng trưởng), đi xuống (suy thoái) hoặc đi ngang.

    => Đường đi lên rõ rệt, từ khoảng 110 (đầu 1949) lên khoảng 400 (cuối 1959).

    => Số hành khách nhìn chung tăng từ đầu đến cuối giai đoạn.

- Seasonality: những biến động lặp đi lặp lại mang tính quy luật với một tần suất và chu kỳ thời gian cố định (như theo quý, theo tháng trong năm, hoặc theo ngày trong tuần)

    => Đường đồ thị có cấu trúc "răng cưa" lặp lại với chu kỳ thời gian chính xác là 12 tháng (hàng năm). Mỗi năm đều xuất hiện một đỉnh nhọn vào giữa năm và một đáy thấp vào giai đoạn cuối năm/đầu năm sau. 

    => Đỉnh nằm ở khoảng tháng 7 - 8, kỳ nghỉ hè tại Bắc Bán Cầu (Mỹ và Châu Âu). Đây là thời điểm trường học đóng cửa, các gia đình tổ chức đi nghỉ mát, và thời tiết mùa hè lý tưởng nhất cho các chuyến bay.

    => Đáy thấp cuối và đầu năm, là giai đoạn kết thúc của kỳ nghỉ. Người dân quay trở lại nhịp sống học tập và làm việc bình thường. Thời tiết khắc nghiệt (bão tuyết) cũng làm cản trở việc đi lại. Lượng hành khách trong các tháng này chủ yếu thu hẹp lại ở tệp khách hàng doanh nhân đi công tác, khiến tổng số vé bán ra chạm đáy.

- Biên độ của các dao động này ngày càng tăng theo thời gian.

    => Đặc điểm của cấu trúc mùa vụ nhân (multiplicative seasonality).

- Chưa thấy điểm bất thường rõ ràng

- Chưa thấy sự thay đổi cấu trúc đột ngột, rõ rệt (structural breaks)

- Không có missing periods trong đồ thị (đã kiểm tra timestamp)

- Có dấu hiệu không dừng do xu hướng và mức biến động thay đổi

### 3.2. Trend analysis

- Line Plot đã giúp nhận diện sơ bộ sự tồn tại của xu hướng biến động của chuỗi theo thời gian. Và line plot còn bao gồm các dao động ngắn hạn và yếu tố mùa vụ nên có thể làm  che khuất đặc điểm của xu hướng dài hạn. Vì vậy, cần trend analysis nhằm làm rõ hướng, hình dạng và sự thay đổi của xu hướng theo thời gian. 

    => Sử dụng rolling mean để làm giảm ảnh hưởng của những dao động ngắn hạn, giúp xu hướng dài hạn của chuỗi thời gian trở nên dễ quan sát hơn.

- Nếu cửa sổ không phù hợp với chu kỳ mùa vụ, đường rolling mean vẫn có thể mang dao động mùa vụ. Cửa sổ quá lớn có thể che khuất các điểm đảo chiều hoặc thay đổi cấu trúc quan trọng.

    => Có thể lựa chọn window bằng một hoặc nhiều lần chu kỳ mùa vụ.

    => Với AirPassengers, sử dụng window = 12, vì chuỗi có tần suất tháng và mùa vụ năm. Nên lấy trung bình 12 tháng liên tiếp giúp làm giảm ảnh hưởng của mùa vụ, làm thể hiện xu hướng dài hạn rõ hơn.

- Nhận xét rolling mean plot:

  - Năm 1950, rolling mean ở khoảng 125–130 nghìn hành khách.
  - Đến cuối năm 1959, rolling mean tăng lên khoảng 425–430 nghìn hành khách.
  - Trong phần lớn giai đoạn quan sát, đường rolling mean đi lên.

  => Chuỗi có xu hướng tăng dài hạn (upward trend) rõ rệt.

  => Mức số lượng hành khách bình quân trong các cửa sổ 12 tháng liên tiếp ngày càng cao, bất chấp những dao động theo mùa của chuỗi gốc.
  
### 3.3. Variability analysis

- Line Plot cho thấy biên độ dao động của số lượng hành khách ở những năm cuối lớn hơn so với những năm đầu. Tuy nhiên, biên độ dao động chỉ phản ánh khoảng cách giữa các giá trị cực đại và cực tiểu, chưa thể hiện đầy đủ mức độ phân tán của tất cả các quan sát quanh giá trị trung bình. 

  => Sử dụng Rolling Standard Deviation (Rolling STD) trên chuỗi thô để theo dõi mức độ phân tán của các quan sát trong từng cửa sổ thời gian, qua đó xác định độ phân tán có tăng, giảm hay duy trì tương đối ổn định qua các giai đoạn.

  => Mục đích: Kiểm tra liệu các giai đoạn của chuỗi có mức độ phân tán tuyệt đối tương tự nhau hay không. Nếu độ phân tán thay đổi đáng kể, cần tìm hiểu/giải thích liệu sự thay đổi đó đến từ trend, seasonality hay các biến động do ngẫu nhiên, bởi những nguyên nhân khác nhau có thể đòi hỏi cách xử lý phù hợp và quyết định hướng xây dựng mô hình dự báo.

- Định nghĩa:  Rolling std là chuỗi các độ lệch chuẩn tính trên cửa sổ trượt, cho thấy độ phân tán quanh mean cục bộ thay đổi thế nào theo thời gian.

  Nói đơn giản, STD của một window đo mức độ phân tán của các quan sát quanh Mean của chính window đó. Bất cứ thứ gì làm các điểm trong window cách xa mean cục bộ đều làm std tăng, bất kể nguyên nhân. Mà một chuỗi raw gồm 3 thành phần cơ bản:

  $$Y_t = T_t + S_t + R_t$$

  | Thành phần | Ảnh hưởng đến rolling std trên chuỗi raw |
  |---|---|
  | Trend ($T_t$) | Nếu trend thay đổi đáng kể trong một cửa sổ, các quan sát ở đầu và cuối cửa sổ có thể cách xa mean, làm tăng độ phân tán. |
  | Mùa vụ ($S_t$) | Các đỉnh và đáy mùa vụ tạo ra độ phân tán trong cửa sổ. Biên độ mùa vụ lớn hơn có thể làm std tăng. |
  | Phần dư ($R_t$) | Các biến động còn lại, gồm nhiễu và những bất thường chưa được giải thích, góp phần vào độ phân tán. |

- Cách thực hiện: Với cửa sổ độ dài w, tại thời điểm t ta lấy w quan sát gần nhất (hoặc quanh t) rồi tính độ lệch chuẩn mẫu của chúng. Trượt cửa sổ đi từng bước thì được một chuỗi mới các giá trị std.

- Nhận xét rolling std plot:

  - Quan sát từ biểu đồ:

    - Giai đoạn 1950–1952: Rolling STD tương đối thấp, chủ yếu dao động khoảng 13–27 nghìn hành khách.
    - Giai đoạn 1953–1955: Rolling STD nhìn chung tăng, xuất hiện những đỉnh đạt khoảng 40–50 nghìn hành khách.
    - Giai đoạn 1956–1959: Rolling STD tiếp tục tăng, thường xuyên vượt 50 nghìn hành khách và có thời điểm đạt khoảng 80 nghìn hành khách.
    - Xét toàn bộ chuỗi: Mặc dù xuất hiện những đợt tăng giảm cục bộ, Rolling STD có xu hướng tăng rõ rệt theo thời gian.

  - Các quan sát trong những cửa sổ 12 tháng ở cuối chuỗi nhìn chung phân tán xa giá trị trung bình của chính cửa sổ đó hơn so với những cửa sổ ở đầu chuỗi.
  
    => Độ phân tán tuyệt đối của số lượng hành khách trong các cửa sổ 12 tháng có xu hướng tăng theo thời gian.

  - Đồng thời, Rolling Mean cũng tăng theo thời gian. Điều này cho thấy mức trung bình và độ phân tán tuyệt đối của chuỗi cùng có xu hướng tăng, nhưng chưa chứng minh rằng chúng có quan hệ tỷ lệ hay quan hệ nhân quả.

  - Và rolling STD được tính trên chuỗi thô nên chịu ảnh hưởng của trend, seasonality và các biến động còn lại. Do đó, chưa thể xác định thành phần nào đóng góp chủ yếu vào sự gia tăng độ phân tán, cũng như chưa thể kết luận phương sai của nhiễu có thay đổi hay không.

- Nhận xét kết hợp rolling mean và std plot:

  - Rolling Mean có xu hướng tăng rõ rệt, trong khi Rolling STD cũng tăng theo thời gian. Những quan sát này cho thấy mức trung bình cục bộ và độ phân tán cục bộ của chuỗi không ổn định qua các giai đoạn.

    => Chuỗi AirPassengers có dấu hiệu không dừng (non-stationary), đặc biệt do xu hướng tăng dài hạn và cấu trúc mùa vụ rõ rệt.

    => Tuy nhiên về mặt thống kê, Rolling Mean và Rolling STD là các thống kê mẫu tính trên những cửa sổ liên tiếp, không trực tiếp chứng minh kỳ vọng và phương sai lý thuyết của quá trình thay đổi theo thời gian. Cần kết hợp với các bằng chứng kiểm định ADF/KPSS để kết luận.

### 3.4. Seasonality analysis

#### Seasonal plot

- Quan sát sơ bộ từ Line Plot cho thấy số lượng hành khách có những dao động lặp lại hằng năm, maybe chuỗi tồn tại seasonality với chu kỳ 12 tháng.

  => Thực hiện Seasonality Analysis để xác định những tháng thường có số lượng hành khách cao hoặc thấp, kiểm tra liệu quy luật tăng giảm này có lặp lại nhất quán qua các năm hay không và biên độ mùa vụ có thay đổi theo thời gian hay không.

  => Mục đích đối với forecasting: Xác định các quy luật biến động lặp lại theo thời gian, đánh giá mức độ nhất quán và sự thay đổi của chúng -> để đánh giá khả năng khai thác thông tin mùa vụ khi dự báo -> thiết kế feature mùa vụ hợp lý cho từng loại mô hình. 
  
  => Đồng thời, kiểm tra xem biên độ mùa vụ có phụ thuộc vào level hay không, từ đó cân nhắc mô hình hóa seasonality theo dạng additive hoặc multiplicative.

- Nhận xét:

  - Tất cả 11 đường có cùng hình dạng. 
  
  - Số lượng hành khách thường ở mức thấp vào đầu năm, tăng rõ rệt trong giai đoạn tháng 6–7, đạt đỉnh vào tháng 7–8, sau đó giảm đến tháng 11 và tăng trở lại vào tháng 12.

  - Trong 11 năm quan sát, có 3 năm đạt đỉnh riêng vào tháng 7, 4 năm đạt đỉnh riêng vào tháng 8 và 4 năm có giá trị tháng 7 và tháng 8 bằng nhau ở mức cao nhất.

  - Giá trị thấp nhất trong năm xuất hiện vào tháng 11 (5 năm), tháng 2 (4 năm) và tháng 1 (2 năm).

  => Chuỗi có quy luật mùa vụ hằng năm rõ rệt: Thời điểm cao điểm tập trung vào tháng 7–8, trong khi thấp điểm thường xuất hiện vào tháng 1–2 hoặc tháng 11. Quy luật này lặp lại tương đối nhất quán trong 11 năm quan sát.
  
  => Chuỗi thể hiện seasonality với chu kỳ 12 tháng rõ rệt. Thời điểm cao điểm và thấp điểm tương đối nhất quán giữa các năm, cho thấy tháng trong năm có thể cung cấp thông tin hữu ích cho forecasting.

  - Mức số lượng hành khách nhìn chung tăng qua các năm. Và khoảng cách giữa giá trị cao nhất và thấp nhất trong từng năm cũng có xu hướng gia tăng theo thời gian.

    => Đây là dấu hiệu phù hợp với multiplicative seasonality, nhưng chưa đủ để xác nhận. Cần kiểm tra thêm mối quan hệ giữa seasonal effects và level, đồng thời xét ảnh hưởng của trend trước khi lựa chọn cấu trúc mùa vụ phù hợp.


#### Seasonal heatmap

- Seasonal Plot giúp quan sát hình dạng biến động qua 12 tháng và so sánh quy luật mùa vụ giữa các năm.

- Còn Seasonal Heatmap giúp đối chiếu trực tiếp giá trị của cùng một tháng qua nhiều năm, xác định tháng nào có giá trị cao hoặc thấp trong từng năm.

- Nhận xét: 

  - Lượng hành khách thường đạt mức cao vào tháng 7–8 và thấp hơn vào tháng 1–2 hoặc tháng 11, củng cố nhận định về quy luật mùa vụ hằng năm từ Seasonal Plot.

  - Số lượng hành khách của cùng một tháng nhìn chung tăng qua các năm, phù hợp với xu hướng tăng dài hạn đã quan sát từ Line Plot và Rolling Mean.

  - Đáng chú ý, năm 1958 có tốc độ tăng trưởng chậm lại rõ rệt so với năm 1957. Nhiều tháng ghi nhận mức hành khách gần như không thay đổi so với cùng kỳ, chẳng hạn tháng 4 và tháng 9 có giá trị bằng nhau giữa hai năm. Tuy nhiên, mức trung bình năm 1958 vẫn cao hơn năm 1957, và đến năm 1959, tăng trưởng phục hồi.

    => Sự chậm lại trong một năm chưa đủ để xác định có structural break. Đồng thời, Seasonal Plot và Heatmap vẫn cho thấy thời điểm cao điểm và thấp điểm năm 1958 tương đối giống các năm khác, nên chưa có bằng chứng trực quan rõ ràng về sự thay đổi seasonal pattern.
---

## GIAI ĐOẠN 4: BIẾN ĐỔI DỮ LIỆU ĐỂ ỔN ĐỊNH PHƯƠNG SAI

- Qua trend analysis và seasonality analysis, chuỗi AirPassengers cho thấy xu hướng tăng dài hạn, độ lớn dao động mùa vụ cũng tăng khi mức số lượng hành khách tăng. Đây là dấu hiệu phù hợp với multiplicative seasonality,  trong đó khi mức số lượng hành khách nền (level) tăng, độ lớn dao động do mùa vụ tạo ra cũng tăng tương ứng.

- Khi mức độ dao động tăng theo level, độ phân tán tuyệt đối của các giá trị quan sát có thể lớn hơn ở những giai đoạn có level cao. Điều này đã được thể hiện qua Rolling STD trên chuỗi thô có xu hướng tăng theo thời gian.

  => Tuy nhiên, sự gia tăng độ phân tán của chuỗi thô chưa chứng minh phương sai của thành phần nhiễu ngẫu nhiên tăng, vì độ phân tán đó còn chịu ảnh hưởng của trend và seasonality.

- Trong khi đó, một số mô hình dự báo, chẳng hạn SARIMA có phương sai hằng số, giả định rằng độ phân tán của thành phần sai số không thay đổi theo thời gian. Nếu phương sai sai số thực tế phụ thuộc vào level hoặc thời gian, giả định này có thể không phù hợp, dẫn đến prediction intervals được ước lượng không chính xác.

- Vì vậy mục tiêu là **thử biến đổi giá trị của chuỗi thời gian sang một thang đo khác, mà độ lớn dao động ít phụ thuộc vào level hơn, từ đó giúp phương sai ổn định hơn theo thời gian**.

- Log transformation là một trong những phương án phù hợp để thử nghiệm. Vì tính chất (\log(ab)=\log(a)+\log(b)), cho phép chuyển cấu trúc nhân trên thang gốc thành cấu trúc cộng trên thang log, với điều kiện các giá trị đều dương. 

- Đối với chuỗi có multiplicative seasonality, tác động mùa vụ được biểu diễn bằng một hệ số nhân với level. Nếu hệ số này tương đối ổn định qua các năm, khi level tăng, mức chênh lệch tuyệt đối do mùa vụ tạo ra cũng tăng, nhưng tỷ lệ chênh lệch so với level gần như không đổi.

- Khi áp dụng Log Transformation, những tác động mùa vụ có cùng tỷ lệ sẽ được biểu diễn bằng cùng một chênh lệch trên thang log, bất kể level cao hay thấp.

- Nhờ đó, hiệu ứng mùa vụ dạng nhân trên chuỗi gốc trở thành hiệu ứng mùa vụ dạng cộng trên thang log, giúp các phương pháp phân rã dạng cộng như STL Decomposition mô tả quy luật mùa vụ thuận tiện hơn.

- Tuy nhiên Log Transformation chỉ thay đổi thang đo của các giá trị quan sát. Nếu số lượng hành khách tiếp tục tăng theo thời gian, chuỗi sau log vẫn có thể mang xu hướng tăng. Tương tự, nếu số lượng hành khách thường đạt cao điểm vào tháng 7–8, quy luật mùa vụ này vẫn tồn tại sau log.

- Điểm khác biệt là những dao động mùa vụ có cùng tỷ lệ với level sẽ được biểu diễn bằng những chênh lệch bằng nhau trên thang log, thay vì những chênh lệch tuyệt đối ngày càng lớn.

- Vì vậy, việc log làm giảm độ phân tán giữa các cửa sổ thời gian nhưng không tự động loại bỏ trend, seasonality hoặc đảm bảo phương sai của thành phần nhiễu hằng số.

- Nhận xét std trước và sau log transformation:

  | Chỉ số | Raw | Log |
  |---|---|---|
  | Yearly STD nhỏ nhất | 13.720 | 0.108 |
  | Yearly STD lớn nhất | 69.830 | 0.160 |
  | STD max/min | 5.09 | 1.48 |

  Vì STD của chuỗi gốc và STD trên thang log có đơn vị khác nhau nên không so sánh trực tiếp giá trị tuyệt đối giữa hai cột. Tỷ số max/min không có đơn vị và cho phép đánh giá sự chênh lệch tương đối của STD giữa các năm trên từng thang đo.

  - Trên chuỗi gốc: Yearly STD tăng rõ rệt qua các năm, với giá trị nhỏ nhất là 13.720 và lớn nhất là 69.830. Tỷ số STD max/min đạt 5.09, cho thấy mức độ phân tán tuyệt đối của số lượng hành khách giữa các tháng trong năm tăng đáng kể theo thời gian.

  - Trên chuỗi sau log: Yearly STD nằm trong khoảng 0.108–0.160, với tỷ số STD max/min giảm xuống 1.48. Các giá trị STD giữa các năm trở nên đồng đều hơn đáng kể, mặc dù vẫn có xu hướng cao hơn ở những năm cuối.

  => Kết luận: Log Transformation giúp giảm rõ rệt sự chênh lệch độ phân tán trong năm giữa các giai đoạn. Kết quả phù hợp với giả thuyết rằng độ lớn dao động của AirPassengers có liên hệ với level và có thể được biểu diễn ổn định hơn trên thang log.

- Nhận xét tập dữ liệu trên thang gốc và thang log:

  - Ở cả hai biểu đồ, số lượng hành khách đều có xu hướng tăng theo thời gian.

  - Trên chuỗi raw: Quan sát từ đầu đến cuối giai đoạn, khoảng cách theo chiều dọc giữa các đỉnh và đáy mùa vụ có xu hướng tăng đáng kể. Các dao động đầu chuỗi tương đối nhỏ, trong khi dao động cuối chuỗi lớn hơn nhiều.

  - Trên chuỗi log: các đỉnh và đáy mùa vụ vẫn lặp lại, nhưng khoảng cách giữa chúng trở nên tương đối đồng đều hơn qua các năm.

# GIAI ĐOẠN 5: STL DECOMPOSITION

- STL (Seasonal and Trend decomposition using LOESS) là phương pháp phân rã chuỗi thời gian thành ba thành phần: Trend, Seasonality và Residual/Residual.

- Line Plot và Seasonal Plot đã giúp nhận diện xu hướng tăng và quy luật mùa vụ của dataset nhưng chưa ước lượng riêng mức đóng góp của từng thành phần vào các giá trị quan sát.

  => Sử dụng STL Decomposition để ước lượng và tách riêng trend, seasonality và Residual.

  => Phân tích thành phần Trend để kiểm tra hình dạng và sự thay đổi của xu hướng dài hạn sau khi tách ảnh hưởng mùa vụ.

  => Phân tích thành phần Seasonality để đánh giá độ lớn và sự thay đổi của hiệu ứng mùa vụ qua các năm, sau khi tách ảnh hưởng của trend.

  => Phân tích residual để xác định những biến động chưa được trend và seasonality giải thích, đặc biệt là các quan sát bất thường và những cấu trúc còn sót lại.

- STL là phương pháp phân rã dạng cộng, biểu diễn chuỗi thành tổng của ba thành phần: trend, seasonality và residual. Tuy nhiên, STL vẫn cho phép seasonal component thay đổi theo thời gian. Vì vậy, phương pháp này có thể áp dụng cho cả chuỗi gốc (biên độ mùa vụ thay đổi) và chuỗi sau log transformation.

- Trên chuỗi gốc, seasonal component thể hiện mức chênh lệch tuyệt đối so với trend. Nếu mùa vụ có tính chất nhân, khi level tăng, chênh lệch tuyệt đối do mùa vụ tạo ra cũng tăng. Chẳng hạn, với hệ số mùa vụ là 1,2, level tăng từ 100 lên 400 sẽ khiến hiệu ứng mùa vụ tăng từ +20 lên +80. Do đó, STL trên chuỗi gốc cần ước lượng seasonal component có biên độ tăng theo thời gian để phản ánh sự thay đổi này.

- Trong ví dụ trên, hiệu ứng mùa vụ ở cả hai mức level đều bằng ln(1,2) ≈ 0,1823. Vì vậy, nếu các hệ số mùa vụ tương đối ổn định qua các năm, STL trên chuỗi log có thể biểu diễn chúng bằng seasonal component tương đối ổn định, thay vì phải ước lượng biên độ tăng theo level.

- STL sử dụng các phương pháp làm mượt để ước lượng thành phần mùa vụ. Khi hiệu ứng mùa vụ có độ lớn tương đối ổn định trên thang log, STL không cần điều chỉnh biên độ seasonal component chỉ để phản ánh sự gia tăng của level. Điều này có thể giúp việc ước lượng quy luật mùa vụ đơn giản hơn. Ngược lại, trên chuỗi gốc, STL phải theo dõi cả sự thay đổi biên độ mùa vụ, và mức độ theo kịp sự thay đổi đó phụ thuộc một phần vào tham số làm mượt.

- Đối với AirPassengers, các phân tích trước đó cho thấy biên độ mùa vụ tuyệt đối tăng theo level, trong khi độ phân tán giữa các năm trở nên đồng đều hơn trên thang log. Đây là cơ sở để ưu tiên STL trên chuỗi log nhằm kiểm tra giả thuyết rằng quy luật mùa vụ được biểu diễn ổn định hơn dưới dạng hiệu ứng tỷ lệ.

  => Lựa chọn STL trên chuỗi log không phải vì STL không thể xử lý multiplicative seasonality trên chuỗi gốc, mà vì phép log có thể biến các hiệu ứng mùa vụ mang tính tỷ lệ thành những chênh lệch cộng tương đối ổn định, giúp STL ước lượng quy luật mùa vụ với ít thay đổi biên độ hơn.

  Tuy nhiên, đây là cơ sở để lựa chọn phương pháp phân tích ban đầu, chưa phải bằng chứng STL trên log cho kết quả phân rã tốt hơn. Điều đó cần được đánh giá từ seasonal component và Residual sau khi thực hiện STL.

- Thực hiện: STL được áp dụng cho logarit của train với `period=12`, `robust=True`; các tham số còn lại để mặc định. Biểu đồ phân rã gồm bốn thành phần: chuỗi quan sát, xu hướng, mùa vụ và phần dư.

  - ``period=12``:

  - ``robust=True``:

- Nhận xét sơ đồ phân rã:

  - **Trend**: Quan sát từ biểu đồ, trend trên thang log tăng xuyên suốt giai đoạn 1949–1959. Tuy nhiên, đường trend không phải đường thẳng có độ dốc cố định. 

    - 1949-1956: Trend tăng tương đối đều.

    - 1957-1958: Đường trend thoải hơn, cho thấy tốc độ tăng trên thang log có dấu hiệu chậm lại. Nguyên nhân chưa xác định được vì không có dữ liệu ngoài.

    - 1959: Đường trend dốc lên trở lại.

    => Vì phân rã được thực hiện trên thang log, độ dốc của trend phản ánh gần đúng tốc độ tăng tương đối của mức nền, chứ không phải số nghìn hành khách tăng thêm mỗi tháng.

  - **Seasonal component**: Quan sát từ biểu đồ, seasonal component thể hiện quy luật lặp lại rõ rệt với chu kỳ 12 tháng. Hình dạng mùa vụ tương đối nhất quán: các đỉnh và đáy xuất hiện theo một mẫu hình gần giống nhau giữa các năm.

    - Tuy nhiên, có một chi tiết đáng lưu ý: biên độ của seasonal component trên thang log không hoàn toàn cố định. Các đỉnh dương có xu hướng cao hơn ở cuối giai đoạn so với đầu giai đoạn.

      => Log transformation đã giúp biểu diễn mùa vụ dưới dạng hiệu ứng cộng trên thang log.

      => Nhưng hệ số mùa vụ trên thang gốc có thể vẫn thay đổi qua các năm. Vì vậy, giả thuyết multiplicative seasonality với hệ số hoàn toàn cố định khi dùng log chưa được xác nhận.

  - **Residual**: 

    - Thứ nhất, phần lớn giá trị residual tập trung khá gần 0 trên thang 0.

      => Trend và seasonality ước lượng tái tạo tương đối sát nhiều quan sát trong mẫu. 
      
      => Số lượng hành khách ước tính từ Trend và Seasonality ước lượng thường không chênh lệch nhiều so với số hành khách thực tế tại các tháng trong tập dữ liệu.

    - Thứ hai, vẫn xuất hiện một số thời điểm có chênh lệch đáng kể và có tính tập trung theo giai đoạn. Cụ thể:
      - Khoảng 1951 và 1953, có một số Residual dương khá lớn.
      - Khoảng 1954, xuất hiện một Residual âm xấp xỉ −0.11.
      - Khoảng 1958, có thêm một số Residual âm đáng chú ý.

      => Do đó, cần kiểm tra thêm liệu Residual còn chứa autocorrelation hoặc các quan sát bất thường hay không.

- Sau khi STL phân tách chuỗi thời gian thành trend, seasonality và residual, thành phần Remainder biểu thị phần biến động không được giải thích bởi hai thành phần đã ước lượng. Tuy nhiên, Remainder không nhất thiết chỉ bao gồm nhiễu ngẫu nhiên mà vẫn có thể chứa các đặc điểm phân phối đáng chú ý hoặc cấu trúc phụ thuộc theo thời gian.

- Vì vậy, cần tiếp tục phân tích residual trên hai phương diện:

  - Đặc điểm phân phối: residual phân bố như thế nào? Phân phối có bất đối xứng hoặc xuất hiện những giá trị cực trị đáng chú ý không? Sử dụng thống kê mô tả và Q–Q plot để khảo sát hình dạng phân phối, đặc biệt là sự khác biệt so với phân phối chuẩn.

  - Sự phụ thuộc theo thời gian: residual tại các tháng khác nhau có còn tương quan với nhau sau khi tách Trend và Seasonality không? → Sử dụng ACF để xem mức tương quan theo từng độ trễ và kiểm định Ljung–Box để đánh giá liệu autocorrelation còn tồn tại trong một nhóm độ trễ hay không.

- Nhận xét Q-Q plot của residual:

    - Vùng trung tâm: Phần lớn các điểm quan sát ở vùng trung tâm tương đối gần đường tham chiếu, mặc dù không hoàn toàn trùng khớp với đường thẳng.

    - Đuôi trái: Một số điểm nằm thấp hơn đường tham chiếu, trong đó có một quan sát cực trị với residual xấp xỉ −0.115.

    - Đuôi phải: Nhiều điểm nằm rõ rệt phía trên đường tham chiếu, với giá trị lớn nhất khoảng +0.115.

    => Kết luận: Q-Q plot cho thấy phân phối residual có sự sai lệch so với phân phối chuẩn tham chiếu, đặc biệt ở hai đuôi. Tuy nhiên, sự sai lệch khỏi phân phối chuẩn không đồng nghĩa với việc residual còn chứa autocorrelation hoặc cấu trúc có thể dự báo. Điều này cần được đánh giá riêng thông qua ACF và kiểm định Ljung-Box.

- Nhận xét ACF của residual:

  - Sau khi sử dụng phương pháp STL để phân rã chuỗi AirPassengers thành ba thành phần: xu hướng (Trend), mùa vụ (Seasonality) và phần dư (Residual), chúng ta tiếp tục phân tích biểu đồ ACF của phần dư để đánh giá liệu chuỗi còn tồn tại sự phụ thuộc theo thời gian hay không.

  - Quan sát biểu đồ, phần lớn các hệ số ACF của phần dư đã giảm về gần 0 và nằm trong vùng tin cậy 95%. So với chuỗi dữ liệu gốc, đây là sự thay đổi đáng kể, cho thấy quá trình phân rã STL đã giúp loại bỏ phần lớn cấu trúc phụ thuộc gắn với xu hướng và mùa vụ.

  - Tuy nhiên, tại lag 1, hệ số ACF vẫn đạt khoảng 0.43, vượt rõ rệt giới hạn tin cậy 95%. Điều này cho thấy phần dư của tháng hiện tại vẫn có tương quan dương đáng kể với phần dư của tháng trước.

  - Tương tự, tại lag 2, ACF khoảng 0.24 và cũng vượt nhẹ giới hạn tin cậy, cho thấy sự phụ thuộc vẫn còn tồn tại ở độ trễ hai tháng.

  - Từ các lag tiếp theo, phần lớn hệ số ACF dao động quanh 0 và nằm trong vùng tin cậy. Đồng thời, chúng ta không còn quan sát được những đỉnh tương quan mùa vụ rõ rệt tại lag 12, 24 và 36 như trên biểu đồ ACF của chuỗi gốc. Điều này cho thấy cấu trúc mùa vụ ban đầu đã được giảm đáng kể sau khi phân rã.

  - Như vậy, có thể kết luận rằng STL đã giúp loại bỏ phần lớn xu hướng và mùa vụ của chuỗi AirPassengers. Tuy nhiên, phần dư vẫn còn tự tương quan ngắn hạn, đặc biệt tại lag 1 và lag 2. Vì vậy, phần dư chưa thể được xem là nhiễu trắng (white noise).

  - Kết quả này cho thấy vẫn còn cấu trúc phụ thuộc theo thời gian chưa được giải thích hết. Để đánh giá chính thức mức độ tự tương quan còn lại, chúng ta có thể sử dụng thêm kiểm định Ljung–Box.

  - Kiểm định Ljung–Box cho kết quả $Q(12)=46.31$, $p=0.000006$ và $Q(24)=54.18$, $p=0.000403$. Ở mức ý nghĩa 5%, bác bỏ giả thuyết không rằng các tự tương quan của phần dư đến lag tương ứng đồng thời bằng $0$. Như vậy, phần dư chưa thể xem là nhiễu trắng; kết luận này phù hợp với các đỉnh ACF ở lag 1 và 2.

  => Kết quả phù hợp với nhận xét ở phần phân rã: STL đã mô tả xu hướng và mùa vụ năm, nhưng residual vẫn giữ cấu trúc phụ thuộc ngắn hạn. Nếu tiếp tục xây dựng mô hình dự báo, có thể cân nhắc mô hình hóa phần phụ thuộc này (chẳng hạn bằng thành phần ARMA), rồi kiểm tra lại residual. Mọi lựa chọn chỉ dựa trên train/validation theo thời gian; năm 1960 vẫn được giữ làm test cuối cùng như đã quy định ở mục 2.7.

  - Q–Q plot và ACF trả lời hai câu hỏi khác nhau: Q–Q plot cho thấy residual lệch khỏi phân phối chuẩn ở hai đuôi; ACF và Ljung–Box cho thấy residual còn phụ thuộc theo thời gian. Không nên dùng kết quả của biểu đồ này để thay cho kết luận từ biểu đồ kia.

  - Dải 95% trên đồ thị là dải xấp xỉ cho từng lag riêng lẻ; khi xem nhiều lag, một vài cột có thể vượt dải do ngẫu nhiên. Vì vậy, nên đọc hình cùng kiểm định Ljung–Box, không kết luận chỉ từ một cột đơn lẻ.

## GIAI ĐOẠN 6: ACF, PACF

| Lag | Ý nghĩa |
|---:|---|
| 1 | So sánh tháng hiện tại với tháng trước |
| 2 | So sánh tháng hiện tại với 2 tháng trước |
| 12 | So sánh tháng hiện tại với cùng kỳ năm trước |
| 24 | So sánh tháng hiện tại với cùng kỳ 2 năm trước |
| 36 | So sánh tháng hiện tại với cùng kỳ 3 năm trước |
| 40 | So sánh tháng hiện tại với tháng cách đây 40 tháng |

**Đầu tiên, chúng ta cần xác định liệu chuỗi AirPassengers có tồn tại tự tương quan mạnh hay không.**

Quan sát biểu đồ ACF, có thể thấy hệ số tự tương quan tại lag 1 rất cao, khoảng 0.94. Khi độ trễ tăng lên, hệ số này giảm tương đối chậm: lag 2 vẫn đạt khoảng 0.87, lag 5 khoảng 0.70 và lag 8 khoảng 0.63.

Điều này cho thấy lượng hành khách ở tháng hiện tại có mối tương quan dương mạnh không chỉ với tháng liền trước mà còn với nhiều tháng trong quá khứ. Nói cách khác, chuỗi có **tính phụ thuộc theo thời gian kéo dài**, thay vì các quan sát biến động hoàn toàn độc lập.

Ngoài ra, các hệ số ACF ở những lag đầu, đến khoảng lag 13, đều vượt giới hạn tin cậy 95%. Điều này củng cố nhận định rằng chuỗi có tự tương quan đáng kể. Tuy nhiên, vùng tin cậy trên biểu đồ ACF mở rộng dần theo độ trễ do phương pháp ước lượng sai số chuẩn có tính đến các hệ số tự tương quan ở những lag trước. Vì vậy, ở các lag xa hơn, một số hệ số ACF vẫn đạt khoảng 0.2–0.5 nhưng đã nằm trong vùng tin cậy. Ta cần phân biệt giữa **độ lớn của hệ số tương quan** và **ý nghĩa thống kê của hệ số đó**.

**Từ đặc điểm tự tương quan mạnh và suy giảm chậm này, chúng ta tiếp tục xem xét liệu chuỗi có xu hướng và mùa vụ hay không.**

Trước hết, về xu hướng, ACF duy trì giá trị dương và suy giảm từ từ qua nhiều độ trễ, thay vì nhanh chóng tiến về 0. Đây là mẫu hình thường gặp ở những chuỗi có xu hướng dài hạn. Đối với AirPassengers, khi kết hợp với đặc điểm lượng hành khách tăng theo thời gian trong dữ liệu gốc, kết quả này cho thấy chuỗi có xu hướng tăng và chưa đáp ứng tốt giả định tính dừng. Tuy nhiên, riêng ACF giảm chậm chỉ là dấu hiệu nghi ngờ tính không dừng, chứ chưa phải bằng chứng chứng minh tuyệt đối.

Tiếp theo, về mùa vụ, mặc dù ACF nhìn chung đang giảm, chúng ta vẫn quan sát được những đỉnh tăng trở lại tại các độ trễ cách nhau 12 tháng. Cụ thể, ACF tại lag 12 đạt khoảng 0.75, lag 24 khoảng 0.51 và lag 36 khoảng 0.26.

Do dữ liệu AirPassengers được ghi nhận hằng tháng, các độ trễ 12, 24 và 36 lần lượt tương ứng với một, hai và ba năm. Việc những đỉnh tương quan xuất hiện lặp lại tại các khoảng cách này cho thấy **dấu hiệu của cấu trúc mùa vụ có chu kỳ 12 tháng**. Điều này phù hợp với việc hoạt động vận tải hàng không thường có những biến động theo mùa trong năm.

Như vậy, biểu đồ ACF gợi ý chuỗi chịu ảnh hưởng đồng thời của xu hướng dài hạn và mùa vụ hằng năm. Hai thành phần này góp phần giải thích tại sao các giá trị trong chuỗi có thể duy trì tương quan dương qua nhiều độ trễ, đồng thời xuất hiện những đỉnh tương quan ở các lag mùa vụ.

**Để tìm hiểu sâu hơn các mối liên hệ này, chúng ta chuyển sang biểu đồ PACF và xác định những lag quan trọng nhất.**

Khác với ACF đo tương quan tổng thể, PACF đo tương quan riêng phần giữa hai thời điểm sau khi đã kiểm soát ảnh hưởng tuyến tính của các độ trễ trung gian.

Trên biểu đồ PACF, lag 1 tiếp tục nổi bật với hệ số khoảng 0.94, gần bằng giá trị ACF tại cùng độ trễ. Điều này cho thấy tháng liền trước mang thông tin rất lớn về tháng hiện tại, ngay cả khi xem xét tương quan riêng phần.

Trong khi đó, tại các lag tiếp theo, phần lớn hệ số PACF giảm xuống đáng kể. Điều này cho thấy nhiều mối tương quan được quan sát trong ACF có thể được giải thích bởi sự phụ thuộc thông qua các độ trễ gần hơn, đặc biệt là lag 1.

Tuy nhiên, PACF không hoàn toàn biến mất sau lag 1. Đáng chú ý nhất là lag 13 có hệ số âm rất lớn, khoảng −0.51, vượt xa vùng tin cậy 95%. Điều này cho thấy sau khi kiểm soát ảnh hưởng tuyến tính của 12 độ trễ trước đó, vẫn tồn tại một mối tương quan riêng phần âm đáng kể giữa tháng hiện tại và thời điểm cách đó 13 tháng.

Kết hợp với những đỉnh tại lag 12, 24 và 36 trên ACF, kết quả này gợi ý cấu trúc phụ thuộc của chuỗi có thể liên quan đến cả thành phần ngắn hạn và mùa vụ. Tuy nhiên, **không nên xem lag 13 là chu kỳ mùa vụ mới**, vì PACF âm tại lag này không trực tiếp đồng nghĩa với chu kỳ 13 tháng.

Ngoài ra, PACF còn có các đỉnh dương ở lag 9 và lag 11, khoảng 0.25, vượt giới hạn tin cậy. Ngược lại, lag 2 khoảng −0.21 và lag 25 khoảng −0.19 chỉ vượt nhẹ giới hạn, nên cần thận trọng khi diễn giải vì chúng cũng có thể xuất hiện do biến động lấy mẫu và việc kiểm tra đồng thời nhiều lag.

Từ lag 14 trở đi, phần lớn các hệ số PACF nằm trong vùng tin cậy, ngoại trừ một số đỉnh riêng lẻ. Điều này cho thấy không phải mọi độ trễ xa đều mang thêm thông tin tuyến tính riêng phần đáng kể sau khi đã kiểm soát các lag trung gian.

**Tổng hợp cả ba câu hỏi, chúng ta rút ra những kết luận chính sau đây.**

Thứ nhất, AirPassengers có tự tương quan dương rất mạnh, đặc biệt tại các độ trễ gần, thể hiện qua ACF lag 1 khoảng 0.94 và mẫu hình suy giảm chậm.

Thứ hai, chuỗi có dấu hiệu rõ rệt của xu hướng tăng và mùa vụ 12 tháng. Điều này phù hợp với đặc điểm của dữ liệu AirPassengers và gợi ý chuỗi gốc chưa dừng.

Thứ ba, xét các độ trễ nổi bật, lag 1 quan trọng nhất đối với sự phụ thuộc ngắn hạn; lag 12 thể hiện rõ dấu hiệu mùa vụ trên ACF; còn lag 13 nổi bật trên PACF, cho thấy một mối tương quan riêng phần âm mạnh sau khi đã kiểm soát các độ trễ trung gian.

**Về ý nghĩa đối với mô hình dự báo**, các kết quả này cho thấy chúng ta không nên lựa chọn trực tiếp mô hình ARIMA hoặc SARIMA chỉ dựa trên ACF và PACF của chuỗi gốc. Thay vào đó, cần tiếp tục xem xét biến đổi log để ổn định phương sai, sai phân thông thường để xử lý xu hướng và sai phân mùa vụ với chu kỳ 12 tháng. Sau những bước này, chúng ta sẽ phân tích lại ACF và PACF để xác định các thành phần phù hợp cho mô hình SARIMA.

---
