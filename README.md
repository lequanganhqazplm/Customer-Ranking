# Dự Án Phân Hạng Khách Hàng (Customer Ranking)

Dự án ứng dụng Trí tuệ nhân tạo và Khai phá dữ liệu để **dự đoán phân hạng giá trị của khách hàng trong 90 ngày tiếp theo** dựa trên lịch sử mua sắm từ bộ dữ liệu thương mại điện tử **UCI Online Retail**. 

Dự án được trang bị **giao diện trực quan Dashboard (Streamlit)** cho phép theo dõi kết quả, đối chiếu mô hình và **kích hoạt huấn luyện mô hình hoàn toàn bằng giao diện (UI) mà không cần dùng dòng lệnh**.

---

## 📌 1. Tổng quan bài toán & Dữ liệu

### 1.1. Bài toán nghiệp vụ
Khác với phương pháp phân cụm tĩnh truyền thống, dự án giải quyết bài toán **dự đoán tương lai (Forward-looking prediction)**:
* **Mốc cắt thời gian (Cutoff date):** Lấy ngày giao dịch cuối cùng trừ đi **90 ngày**.
  * Dữ liệu trước mốc cắt: Tính toán 6 đặc trưng hành vi mua sắm lịch sử.
  * Dữ liệu trong 90 ngày sau mốc cắt: Quan sát mức chi tiêu thực tế (`FutureSpend`) để phân thành 4 hạng.
* **4 phân hạng khách hàng (Customer Ranks):**
  1. 💎 **Diamond (Kim cương):** Chi tiêu tương lai $\ge$ phân vị $P85$ của nhóm chi tiêu dương ở tập Train.
  2. 🥇 **Gold (Vàng):** Chi tiêu tương lai từ $P50$ đến dưới $P85$.
  3. 🥈 **Silver (Bạc):** Có phát sinh chi tiêu dương nhưng dưới $P50$.
  4. 🥉 **Bronze (Đồng):** Không phát sinh chi tiêu hợp lệ nào trong 90 ngày tiếp theo (chi tiêu $\le 0$).

### 1.2. Đặc trưng lịch sử (RFM mở rộng)
Được trích xuất từ các đơn hàng trước mốc cutoff:
* `Recency`: Số ngày kể từ lần mua gần nhất tới mốc dự đoán.
* `Frequency`: Số lượng hóa đơn mua hàng riêng biệt.
* `Monetary`: Tổng số tiền đã chi tiêu (£).
* `TotalQuantity`: Tổng số lượng sản phẩm đã mua.
* `UniqueProducts`: Số loại sản phẩm (mã SKU) khác nhau đã mua.
* `AvgOrderValue`: Giá trị trung bình mỗi đơn hàng (`Monetary / Frequency`).

### 1.3. Chống rò rỉ dữ liệu (Anti-Data Leakage)
* Chia tập Train/Test theo khách hàng (`CustomerID`), không để trùng khách hàng giữa 2 tập.
* Bộ chuẩn hóa (`StandardScaler`) và ngưỡng phân vị ($P50, P85$) **chỉ được học trên tập Train**, sau đó áp dụng cố định sang tập Test.
* Nghiêm cấm đưa các trường tương lai hoặc định danh vào ma trận đặc trưng $X$.

---

## 🤖 2. Các mô hình Machine Learning

Dự án quy hoạch khảo sát 10 mô hình, hiện đã triển khai hoàn chỉnh 2 mô hình đầu tiên:

1. **K-Means (Phân cụm không giám sát kết hợp ánh xạ hạng):**
   * Phân chia tập khách hàng thành các cụm dựa trên khoảng cách đặc trưng RFM.
   * Sử dụng thuật toán `ClusterRankMapper` tính điểm tổng hợp *RFM Composite Percentile* để sắp xếp các cụm từ thấp đến cao và ánh xạ tương ứng vào 4 hạng (*Bronze $\rightarrow$ Silver $\rightarrow$ Gold $\rightarrow$ Diamond*).
2. **Hồi quy tuyến tính - Linear Regression (Học có giám sát):**
   * Huấn luyện mô hình hồi quy để dự đoán trực tiếp số tiền chi tiêu tương lai (`FutureSpend`).
   * Chặn dưới chi tiêu không âm ($\ge 0$) và quy đổi sang 4 hạng dựa vào ngưỡng $P50, P85$ học từ tập Train.
   * Cung cấp bảng trọng số $\beta$ (Coefficients) giải thích mức độ tác động của từng đặc trưng tới chi tiêu tương lai.

---

## 📊 3. Hệ thống độ đo đánh giá (Evaluation Metrics)

Mọi mô hình đều được đánh giá đồng nhất trên **5 chỉ số vĩ mô (Macro metrics)**:
* **Accuracy:** Tỷ lệ dự đoán đúng chính xác cả 4 hạng trên tập kiểm tra.
* **Precision (Macro):** Độ chuẩn xác trung bình khi gán từng hạng.
* **Recall (Macro):** Khả năng nhận diện và bao phủ trung bình của các hạng.
* **F1-Score (Macro):** Điểm cân bằng trung hòa giữa Precision và Recall.
* **Balanced Accuracy:** Độ chính xác cân bằng giữa các hạng (chỉ số quan trọng nhất vì dữ liệu thực tế bị lệch nhiều về nhóm Bronze).

---

## 🚀 4. Hướng dẫn khởi chạy & Sử dụng qua Giao diện (UI)

Dự án được tối ưu để bạn **chỉ cần gõ duy nhất một lệnh khởi động giao diện Streamlit**. Mọi thao tác huấn luyện, tùy chỉnh tham số và xem so sánh đều thực hiện trực tiếp trên màn hình web.

### Bước 1: Chuẩn bị môi trường & Dữ liệu
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Đặt file dữ liệu `Online Retail.xlsx` vào thư mục:
   ```text
   data/raw/Online Retail.xlsx
   ```

### Bước 2: Khởi động giao diện Web (Lệnh duy nhất)
Tại thư mục gốc của dự án, mở Terminal và chạy lệnh:
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở giao diện tại địa chỉ `http://localhost:8501`.

---

### Bước 3: Thao tác & Huấn luyện mô hình hoàn toàn qua UI

#### 1. Huấn luyện mô hình mới ngay trên giao diện:
1. Nhìn vào thanh bên trái (**Sidebar**), bấm mở mục **"🚀 Huấn luyện mô hình từ UI"**.
2. **Chọn mô hình muốn chạy:** 
   * `Hồi quy tuyến tính (Linear Regression)` hoặc
   * `K-Means (phân cụm)`.
3. **Tùy chỉnh tham số (nếu muốn):** 
   * Kéo thanh trượt để chọn tỷ lệ tập kiểm tra (*Test size*, mặc định 0.2 tức 20%).
   * Chọn số cụm nếu chạy K-Means (mặc định 4).
4. Bấm nút **"Bắt đầu huấn luyện"**:
   * Hệ thống sẽ tự động đọc dữ liệu, làm sạch, huấn luyện và đánh giá trên tập test.
   * Sau khi hoàn tất, giao diện sẽ tự động làm mới và cập nhật ngay kết quả mới nhất lên Dashboard!

#### 2. Xem chi tiết kết quả từng mô hình:
1. Tại thanh điều hướng, chọn mục **"Mô hình"**.
2. Chọn mô hình bạn muốn xem trong menu thả xuống (*Dropdown*):
   * **Với K-Means:** Xem 5 thẻ điểm số, các chỉ số cụm (Silhouette, Inertia), bảng đặc điểm các cụm và biểu đồ phân tán cụm.
   * **Với Hồi quy tuyến tính:** Xem 5 thẻ điểm số, ngưỡng $P50, P85$, biểu đồ cột hệ số trọng số $\beta$ của các đặc trưng, và biểu đồ so sánh phân bố hạng thực tế vs dự đoán.

#### 3. So sánh đối đầu giữa các mô hình:
1. Tại thanh điều hướng, chọn mục **"So sánh mô hình"**.
2. Quan sát bảng tổng hợp đối đầu: Hệ thống tự động so sánh K-Means và Linear Regression trên cùng tập kiểm tra holdout.
3. Xem mục **"Kết quả cao nhất theo từng chỉ số"** để biết mô hình nào tối ưu hơn ở Accuracy, F1-score hay Balanced Accuracy.

#### 4. Tra cứu thông tin khách hàng:
1. Chọn mục **"Dữ liệu khách hàng"**.
2. Nhập mã khách hàng (`CustomerID`, ví dụ: `14646`, `17850`...) để tra cứu ngay hạng khách hàng dự đoán cùng các thông số RFM chi tiết.

---

## 📁 5. Cấu trúc thư mục dự án

```text
Customer-Ranking/
├── app.py                           # Ứng dụng Dashboard Streamlit (Giao diện chính)
├── README.md                        # Hướng dẫn tổng quan & sử dụng dự án
├── HUONG_DAN_TEAM.md                # Tài liệu phân công và quy chuẩn kỹ thuật của nhóm
├── requirements.txt                 # Các thư viện phụ thuộc
├── data/
│   ├── raw/Online Retail.xlsx       # File dữ liệu thô gốc (tải từ UCI)
│   └── processed/                   # Báo cáo đặc trưng khách hàng đã làm sạch
├── outputs/
│   └── runs/                        # Lưu trữ độc lập kết quả các lần chạy (runs)
├── src/                             # Mã nguồn chính của hệ thống
│   ├── data_loader.py               # Module đọc dữ liệu Excel
│   ├── preprocessing.py             # Làm sạch hóa đơn, loại bỏ đơn hủy
│   ├── feature_engineering.py       # Tạo đặc trưng RFM lịch sử & gán nhãn 4 hạng
│   ├── evaluate.py                  # Module tính 5 độ đo xếp hạng chuẩn
│   ├── predict.py                   # Adapter dự đoán chung cho các loại mô hình
│   ├── dashboard_helpers.py         # Hàm hỗ trợ hiển thị và kiểm tra hợp lệ dữ liệu
│   ├── train.py                     # Pipeline huấn luyện chuẩn (hỗ trợ CLI & UI)
│   ├── train_kmeans.py              # Script huấn luyện riêng cho K-Means
│   ├── train_linear_regression.py   # Script huấn luyện riêng cho Hồi quy tuyến tính
│   └── models/
│       ├── kmeans_model.py          # Thuật toán K-Means & ClusterRankMapper
│       └── linear_regression_model.py # Thuật toán Hồi quy tuyến tính & ánh xạ phân hạng
└── tests/                           # Hệ thống 60 kiểm thử tự động (Unit tests)
```

