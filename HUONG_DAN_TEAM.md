# HƯỚNG DẪN TEAM — CUSTOMER RANKING

> **Mục tiêu:** Hướng dẫn cả nhóm phát triển project bằng AI coding assistant (vibe coding), nhưng vẫn kiểm soát được kiến trúc, dữ liệu, kết quả và việc ghép code.
>
> **Đề tài:** Xây dựng hệ thống phân tích và phân nhóm khách hàng dựa trên dữ liệu hành vi mua hàng bằng học máy.

---

## 1. Mục tiêu và chức năng cần có

Ứng dụng dùng dataset **UCI Online Retail** để phân tích lịch sử mua hàng và hiển thị kết quả trên dashboard.

Các chức năng cuối cùng:

- Đọc và làm sạch dữ liệu giao dịch.
- Tạo đặc trưng hành vi khách hàng theo RFM và một số đặc trưng bổ sung.
- Xây dựng 3 mô hình:
  1. **K-Means:** phân nhóm khách hàng dựa trên hành vi mua hàng.
  2. **Linear Regression (hồi quy tuyến tính):** dự đoán tổng chi tiêu trong kỳ tiếp theo.
  3. **Decision Tree (cây quyết định):** dự đoán khách hàng có mua hàng trở lại trong kỳ tiếp theo hay không.
- Tạo bảng đánh giá mô hình, biểu đồ trực quan và nhận xét về khách hàng giá trị cao, khách hàng mua thường xuyên, khách hàng có nguy cơ không quay lại.
- Có giao diện web bằng **Streamlit** để xem dữ liệu, kết quả, biểu đồ và danh sách khách hàng.
- Có hướng dẫn cài đặt/chạy project và nguồn dữ liệu trong `README.md`.

**Không được bịa số liệu kết quả.** Bảng, biểu đồ và nhận xét phải lấy từ dữ liệu/mô hình đã chạy thực tế.

---

## 2. Dataset và quy tắc xử lý dữ liệu

### Nguồn dữ liệu

- Trang chính thức: <https://archive.ics.uci.edu/dataset/352/online+retail>
- Tải file `Online Retail.xlsx` và đặt tại:

```text
data/raw/Online Retail.xlsx
```

Dataset có dữ liệu giao dịch của cửa hàng bán lẻ trực tuyến tại Anh trong giai đoạn 01/12/2010–09/12/2011. Các cột chính gồm `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`. Nguồn UCI ghi nhận 541.909 dòng và giấy phép CC BY 4.0. Khi làm báo cáo cần ghi nguồn dataset và trích dẫn theo trang UCI.

### Quy tắc làm sạch bắt buộc

1. Đọc file Excel bằng `pandas.read_excel()`.
2. Chuyển `InvoiceDate` sang kiểu ngày giờ; chuyển `Quantity` và `UnitPrice` sang dạng số.
3. Loại các dòng thiếu `CustomerID` vì không thể gắn giao dịch với khách hàng cụ thể.
4. Loại hóa đơn hủy có `InvoiceNo` bắt đầu bằng chữ `C`.
5. Với bài toán phân tích **mua hàng**, chỉ giữ giao dịch có `Quantity > 0` và `UnitPrice > 0`. Ghi rõ trong báo cáo rằng cách làm này bỏ qua giao dịch hủy/trả hàng.
6. Tạo cột `TotalPrice = Quantity * UnitPrice`.
7. Không ghi đường dẫn cố định của máy cá nhân vào code. Dùng đường dẫn tương đối từ thư mục gốc project hoặc `pathlib.Path`.
8. Kiểm tra dữ liệu sau làm sạch: số dòng, số khách hàng, ngày nhỏ nhất/lớn nhất, giá trị thiếu và giá trị bất thường.

Nếu dữ liệu đã tải về không tồn tại, chương trình phải báo rõ đường dẫn cần đặt file, không tự tạo dữ liệu giả để thay thế.

---

## 3. Cấu trúc thư mục thống nhất

Giữ lại các file đang có trong project. Bổ sung hoặc sắp xếp theo cấu trúc sau:

```text
CustomerRanking/
├── app.py                         # Giao diện Streamlit
├── requirements.txt               # Danh sách thư viện
├── README.md                      # Giới thiệu và hướng dẫn chạy
├── HUONG_DAN_TEAM.md              # File hướng dẫn này
├── .gitignore                     # File không đưa lên Git
│
├── data/
│   ├── raw/
│   │   └── .gitkeep               # Dataset tải riêng, không commit file Excel
│   └── processed/
│       └── .gitkeep               # Dữ liệu sau xử lý, được tạo khi chạy
│
├── models/                        # Chỉ lưu mô hình đã huấn luyện (.pkl/.joblib)
│   └── .gitkeep
│
├── outputs/
│   ├── metrics/
│   │   └── .gitkeep               # Bảng đánh giá CSV
│   ├── figures/
│   │   └── .gitkeep               # Biểu đồ PNG
│   └── reports/
│       └── .gitkeep               # Bảng kết quả/nhận xét xuất ra
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Hằng số, đường dẫn, tên cột dùng chung
│   ├── data_loader.py              # Đọc dữ liệu nguồn
│   ├── preprocessing.py            # Làm sạch giao dịch
│   ├── feature_engineering.py      # Tạo đặc trưng khách hàng và nhãn tương lai
│   ├── train.py                    # Điều phối việc huấn luyện cả 3 mô hình
│   ├── evaluate.py                 # Tính độ đo và lưu bảng kết quả
│   ├── predict.py                  # Ghép dự đoán/phân nhóm thành bảng khách hàng
│   ├── visualization.py            # Tạo biểu đồ và lưu file
│   ├── insights.py                 # Nhận xét nhóm khách hàng
│   │
│   └── models/                     # Chứa CODE của 3 thuật toán
│       ├── __init__.py
│       ├── kmeans_model.py
│       ├── linear_regression_model.py
│       └── decision_tree_model.py
│
└── tests/
    ├── test_preprocessing.py
    └── test_feature_engineering.py
```

### Phân biệt hai thư mục `models`

- `src/models/`: code Python do nhóm viết để tạo/huấn luyện các thuật toán.
- `models/` ở thư mục gốc: file mô hình đã huấn luyện được lưu bằng `joblib`, ví dụ `kmeans.pkl`, `linear_regression.pkl`, `decision_tree.pkl`.

Không để toàn bộ project trong một file. `app.py` chỉ làm giao diện và gọi các hàm đã viết ở `src/`, không viết lại thuật toán bên trong giao diện.

---

## 4. Quy ước dữ liệu và bài toán

### 4.1. Đặc trưng khách hàng

Mỗi dòng trong bảng mô hình tương ứng với **một khách hàng**. Nên tạo ít nhất các cột:

| Cột | Ý nghĩa |
|---|---|
| `CustomerID` | Mã khách hàng, chỉ dùng để nhận diện; không đưa vào đầu vào ML |
| `Recency` | Số ngày từ lần mua gần nhất trong giai đoạn lịch sử đến ngày chốt dữ liệu |
| `Frequency` | Số hóa đơn khác nhau khách hàng đã mua trong giai đoạn lịch sử |
| `Monetary` | Tổng tiền mua hàng trong giai đoạn lịch sử (`TotalPrice` cộng dồn) |
| `TotalQuantity` | Tổng số sản phẩm đã mua |
| `UniqueProducts` | Số mã sản phẩm khác nhau đã mua |
| `AvgOrderValue` | Giá trị trung bình mỗi hóa đơn |
| `FutureSpend` | Tổng tiền mua trong kỳ dự đoán; **nhãn mục tiêu** cho hồi quy |
| `ReturnedNext90Days` | `1` nếu có mua hàng trong kỳ dự đoán, ngược lại `0`; **nhãn mục tiêu** cho cây quyết định |

`FutureSpend` và `ReturnedNext90Days` là nhãn mục tiêu, **không được đưa vào đặc trưng đầu vào** của mô hình.

### 4.2. Tách thời gian để tránh rò rỉ dữ liệu

Đề xuất dùng khoảng **90 ngày cuối của dữ liệu** làm kỳ dự đoán:

- **Giai đoạn lịch sử:** giao dịch trước ngày bắt đầu kỳ dự đoán. Chỉ giai đoạn này được dùng để tính RFM/đặc trưng đầu vào.
- **Giai đoạn tương lai:** khoảng 90 ngày cuối. Dùng giai đoạn này để tính `FutureSpend` và `ReturnedNext90Days`.
- Mốc thời gian phải được xác định từ dữ liệu (ví dụ dựa vào ngày giao dịch lớn nhất), không nhập một ngày cứng không liên quan đến dataset.
- Chỉ đưa vào bảng mô hình những khách hàng xuất hiện trong giai đoạn lịch sử. Khách hàng không có giao dịch trong kỳ tương lai vẫn giữ lại và có `FutureSpend = 0`, `ReturnedNext90Days = 0`.
- Sau khi đã tạo **một dòng trên mỗi khách hàng**, mới chia train/test cho Linear Regression và Decision Tree. Dùng `random_state=42` để kết quả dễ tái lập.

Cần ghi rõ đây là dự đoán khả năng mua lại trong kỳ quan sát tiếp theo, không phải lời khẳng định chắc chắn về hành vi tương lai.

### 4.3. Cách hiểu 3 mô hình

**K-Means**
- Đầu vào chính: `Recency`, `Frequency`, `Monetary`.
- Chuẩn hóa đặc trưng trước khi phân cụm, ví dụ dùng `StandardScaler`.
- Mặc định thử 3 cụm, nhưng có thể cho phép cấu hình số cụm.
- Lưu mã cụm và bảng thống kê trung bình/trung vị của mỗi cụm.
- Không mặc định cụm `0` luôn là khách tốt nhất. Số cụm chỉ là mã; phải xem đặc điểm từng cụm rồi mới viết nhận xét.

**Linear Regression**
- Đầu vào: đặc trưng lịch sử, không gồm `CustomerID`, `FutureSpend`, `ReturnedNext90Days`.
- Nhãn cần dự đoán: `FutureSpend`.
- Kết quả dự đoán không được nhỏ hơn 0 khi hiển thị chi tiêu; nếu cần, chặn giá trị âm về 0 ở bước tạo kết quả đầu ra và ghi chú cách xử lý.

**Decision Tree Classifier**
- Đầu vào: đặc trưng lịch sử.
- Nhãn cần dự đoán: `ReturnedNext90Days`.
- Kết quả: `1` = dự đoán có mua hàng trở lại trong kỳ; `0` = dự đoán không mua lại trong kỳ.
- Có thể dùng `predict_proba()` để hiển thị xác suất dự đoán nếu mô hình hỗ trợ.

---

## 5. Độ đo đánh giá: không dùng sai loại

Ảnh yêu cầu có 5 độ đo: Accuracy, Precision, Recall, F1-score và MAE. Có thể đưa chúng vào **một bảng tổng hợp**, nhưng phải để `N/A` ở mô hình mà độ đo đó không áp dụng. Không được tự tính Accuracy/Precision/Recall/F1 cho K-Means hoặc MAE cho bài toán phân loại.

| Mô hình | Độ đo chính | Độ đo bổ sung |
|---|---|---|
| K-Means | Silhouette Score (cao thường tốt hơn) | Inertia (thấp hơn; dùng hỗ trợ chọn số cụm) |
| Linear Regression | MAE (thấp hơn tốt hơn) | R² (cao hơn thường tốt hơn) |
| Decision Tree | Accuracy, Precision, Recall, F1-score (thường cao hơn tốt hơn) | Confusion matrix |

Trong `outputs/metrics/model_metrics.csv`, tạo bảng có cột như:

```text
Model,Task,Accuracy,Precision,Recall,F1_score,MAE,R2,Silhouette,Inertia
```

Các ô không phù hợp ghi `N/A` hoặc để trống có quy ước rõ ràng. Trên giao diện, trình bày độ đo theo từng loại bài toán và giải thích ngắn gọn. Không xếp hạng trực tiếp cả ba mô hình bằng một con số duy nhất vì chúng làm ba nhiệm vụ khác nhau.

---

## 6. Yêu cầu đầu ra

Khi chạy huấn luyện thành công, tối thiểu phải tạo được:

### File dữ liệu/kết quả

- `data/processed/customer_dataset.csv`: đặc trưng và nhãn mục tiêu theo khách hàng.
- `outputs/metrics/model_metrics.csv`: bảng độ đo của 3 mô hình.
- `outputs/reports/customer_results.csv`: mã khách hàng, RFM, cụm K-Means, chi tiêu dự đoán, dự đoán mua lại và nhóm nhận xét.
- `models/kmeans.pkl`, `models/linear_regression.pkl`, `models/decision_tree.pkl`: mô hình đã huấn luyện.

### Biểu đồ tối thiểu

- Số lượng khách hàng trong từng cụm K-Means.
- Biểu đồ phân bố khách hàng theo Recency/Frequency/Monetary; nếu dùng scatter plot thì chú thích rõ trục và đơn vị.
- Biểu đồ so sánh độ đo phù hợp cho từng mô hình (không gom các độ đo khác đơn vị lên cùng một trục nếu gây hiểu nhầm).
- Biểu đồ Actual vs Predicted cho hồi quy.
- Confusion matrix cho Decision Tree.

Các biểu đồ phải được hiển thị trên dashboard và có thể lưu vào `outputs/figures/`.

### Nhận xét khách hàng

Phải có phần nhận xét được sinh từ số liệu thực tế, gồm:

- **Khách hàng giá trị cao:** chi tiêu lịch sử cao, có thể kết hợp với tần suất mua cao.
- **Khách hàng mua thường xuyên:** số hóa đơn khác nhau cao.
- **Khách hàng có nguy cơ không quay lại:** dựa vào dự đoán `ReturnedNext90Days = 0` của Decision Tree, có thể kết hợp Recency cao để ưu tiên xem xét.

Các nhóm nhận xét có thể **chồng lấn**. Ví dụ một khách hàng vừa mua thường xuyên vừa có giá trị cao. Quy tắc xác định nhóm phải được ghi trong code và giao diện; nên dùng ngưỡng phân vị (ví dụ phân vị 75%) hoặc quy tắc giải thích được, không tự đặt ngưỡng tùy ý mà không nói rõ.

Với K-Means, hiển thị thêm hồ sơ mỗi cụm: số khách hàng, Recency trung vị, Frequency trung vị, Monetary trung vị. Nhận xét cụm phải được tạo dựa trên các giá trị đó, không gán tên cụm cố định chỉ theo số `0`, `1`, `2`.

---

## 7. Yêu cầu giao diện `app.py`

Dùng Streamlit để làm dashboard. Giao diện tối thiểu có các khu vực/tab sau:

1. **Tổng quan:** số khách hàng, số giao dịch sau làm sạch, tổng doanh thu trong dữ liệu, ngày bắt đầu/kết thúc.
2. **Phân nhóm khách hàng:** số lượng từng cụm, bảng thống kê cụm và biểu đồ.
3. **So sánh mô hình:** bảng metric, mô tả nhiệm vụ và giải thích tại sao mỗi mô hình có độ đo khác nhau.
4. **Danh sách khách hàng:** bảng có thể lọc theo nhóm, cụm, tần suất, chi tiêu và dự đoán mua lại.
5. **Nhận xét:** số lượng/tỷ lệ khách giá trị cao, khách mua thường xuyên, khách dự đoán không quay lại; giải thích quy tắc xác định.

Yêu cầu giao diện:

- Tên, nhãn, chú thích bằng tiếng Việt.
- Hiển thị tiền theo đơn vị GBP (£) vì dataset dùng bảng giá bằng bảng Anh; không tự đổi thành VND nếu không có tỷ giá và yêu cầu rõ ràng.
- Hiển thị thông báo dễ hiểu khi chưa tải dataset hoặc chưa chạy huấn luyện.
- Không huấn luyện lại tất cả mô hình mỗi lần người dùng đổi tab. Ưu tiên đọc kết quả/mô hình đã lưu; có nút **Chạy huấn luyện** riêng nếu cần.
- Không để logic tiền xử lý và thuật toán bị lặp lại trong `app.py`.

---

## 8. Nội dung từng file và giao kèo hàm

Các thành viên phải thống nhất tên hàm/đầu ra trước khi code. Có thể dùng giao kèo sau để tránh lỗi khi ghép file:

| File | Trách nhiệm chính |
|---|---|
| `src/config.py` | Đường dẫn, tên file, cột đặc trưng và cấu hình kỳ dự đoán |
| `src/data_loader.py` | `load_data(path=None)` đọc Excel và trả về `DataFrame` |
| `src/preprocessing.py` | `clean_transactions(df)` làm sạch và tạo `TotalPrice` |
| `src/feature_engineering.py` | `build_customer_dataset(df, prediction_days=90)` tạo một dòng/khách hàng, đặc trưng và nhãn tương lai |
| `src/models/kmeans_model.py` | Hàm tạo/huấn luyện K-Means và trả lại mô hình, nhãn cụm, độ đo phân cụm |
| `src/models/linear_regression_model.py` | Hàm tạo/huấn luyện Linear Regression |
| `src/models/decision_tree_model.py` | Hàm tạo/huấn luyện Decision Tree Classifier |
| `src/train.py` | Điều phối toàn quy trình; lưu mô hình và kết quả |
| `src/evaluate.py` | Tính metric từ nhãn thật và dự đoán; lưu `model_metrics.csv` |
| `src/predict.py` | Tạo bảng kết quả khách hàng từ mô hình đã huấn luyện |
| `src/visualization.py` | Hàm vẽ biểu đồ; không làm sạch lại dữ liệu |
| `src/insights.py` | Tạo nhóm nhận xét và tóm tắt bằng số liệu thật |
| `app.py` | Đọc kết quả và hiển thị dashboard |

Nếu thay đổi chữ ký hàm hoặc tên cột, phải báo cho người phụ trách file liên quan và cập nhật nơi gọi hàm. Tránh mỗi thành viên tự đặt tên cột khác nhau, ví dụ `Frequency`, `frequency`, `SoLanMua` cùng xuất hiện trong project.

---

## 9. Quy trình vibe coding — làm từng bước với AI

**Nguyên tắc:** không yêu cầu AI tạo toàn bộ project trong một prompt lớn. Làm theo từng giai đoạn, chạy thử sau mỗi giai đoạn và yêu cầu AI chỉ sửa lỗi liên quan. Mỗi thành viên phải đọc diff trước khi chấp nhận thay đổi.

### Bước 1 — Nhờ AI kiểm tra cấu trúc hiện tại

Prompt có thể dùng:

```text
Bạn là lập trình viên Python hỗ trợ project CustomerRanking.
Hãy kiểm tra cấu trúc thư mục và các file hiện có trước khi sửa.
Project dùng dataset UCI Online Retail, Streamlit và scikit-learn.
Tôi cần 3 mô hình: K-Means phân cụm, Linear Regression dự đoán chi tiêu 90 ngày tiếp theo,
Decision Tree dự đoán có mua lại trong 90 ngày tiếp theo hay không.
Hãy đề xuất các file thiếu và giao kèo hàm giữa các file.
Không xóa hoặc viết lại các file không liên quan. Trước tiên chỉ báo cáo kế hoạch,
chưa tạo code hàng loạt.
```

### Bước 2 — Làm phần dữ liệu trước

Prompt:

```text
Hãy triển khai data_loader.py, preprocessing.py và feature_engineering.py theo kiến trúc đã thống nhất.
Đọc data/raw/Online Retail.xlsx; xử lý CustomerID thiếu, hóa đơn hủy InvoiceNo bắt đầu bằng C,
Quantity <= 0, UnitPrice <= 0; tạo TotalPrice = Quantity * UnitPrice.
Tạo đặc trưng lịch sử Recency, Frequency (đếm InvoiceNo duy nhất), Monetary,
TotalQuantity, UniqueProducts, AvgOrderValue.
Dùng 90 ngày cuối làm kỳ dự đoán, chỉ dùng dữ liệu trước kỳ đó để tạo X.
Tạo FutureSpend và ReturnedNext90Days từ dữ liệu trong kỳ dự đoán.
Mỗi khách hàng chỉ có một dòng. Không đưa CustomerID hoặc các cột mục tiêu vào X.
Dùng đường dẫn tương đối, kiểm tra cột bắt buộc và báo lỗi dễ hiểu nếu thiếu file/cột.
Viết thêm kiểm tra cơ bản cho đầu ra. Chỉ sửa các file thuộc giai đoạn dữ liệu này.
```

Kiểm tra trước khi tiếp tục:

- Số dòng khách hàng bằng số `CustomerID` duy nhất sau xử lý trong giai đoạn lịch sử.
- `Recency`, `Frequency`, `Monetary` không bị thiếu; `Frequency > 0`.
- `FutureSpend >= 0`; `ReturnedNext90Days` chỉ có `0` hoặc `1`.
- Dữ liệu đầu vào của mô hình không có `FutureSpend` hoặc `ReturnedNext90Days`.

### Bước 3 — Viết K-Means

Prompt:

```text
Chỉ triển khai src/models/kmeans_model.py.
Dùng Recency, Frequency, Monetary; chuẩn hóa đặc trưng bằng StandardScaler trước K-Means.
Mặc định n_clusters=3 và random_state=42, n_init=10.
Trả về mô hình có thể dùng lại để predict cụm cho khách hàng mới, nhãn cụm,
Silhouette Score và Inertia. Có kiểm tra trường hợp dữ liệu quá ít hoặc chỉ tạo được một cụm.
Không tự gán Cluster 0 là khách tốt nhất; ý nghĩa cụm sẽ được phân tích từ thống kê cụm.
Không sửa app.py hoặc các mô hình khác.
```

### Bước 4 — Viết Linear Regression

Prompt:

```text
Chỉ triển khai src/models/linear_regression_model.py.
Huấn luyện LinearRegression để dự đoán FutureSpend từ đặc trưng lịch sử đã thống nhất.
Hàm huấn luyện nhận X_train, y_train và trả về model đã fit.
Không nhận CustomerID, FutureSpend hoặc ReturnedNext90Days làm đặc trưng.
Không xử lý/đọc file dữ liệu bên trong module mô hình.
Giữ code đơn giản, có kiểm tra đầu vào và không sửa các file khác.
```

### Bước 5 — Viết Decision Tree

Prompt:

```text
Chỉ triển khai src/models/decision_tree_model.py.
Dùng DecisionTreeClassifier để dự đoán ReturnedNext90Days (0/1) từ đặc trưng lịch sử.
Đặt random_state=42; có thể giới hạn max_depth để tránh cây quá phức tạp.
Hàm huấn luyện nhận X_train, y_train và trả về model đã fit.
Không dùng cột mục tiêu hoặc CustomerID trong X. Không sửa file khác.
```

### Bước 6 — Ghép huấn luyện và đánh giá

Prompt:

```text
Hãy hoàn thiện src/train.py và src/evaluate.py để gọi 3 module mô hình hiện có.
Chia train/test theo dòng khách hàng sau khi đã tạo dataset, dùng random_state=42;
Decision Tree dùng stratify nếu số lượng lớp cho phép.
K-Means đánh giá bằng Silhouette Score và Inertia.
Linear Regression đánh giá bằng MAE và R2.
Decision Tree đánh giá bằng Accuracy, Precision, Recall, F1-score và confusion matrix.
Tạo một bảng tổng hợp model_metrics.csv; metric không áp dụng ghi N/A.
Lưu model bằng joblib vào thư mục models/ ở gốc project.
Lưu kết quả khách hàng vào outputs/reports/. Không bịa số liệu và không đưa cột mục tiêu vào X.
Chương trình phải chạy từ thư mục gốc bằng lệnh python -m src.train.
Chỉ sửa các file điều phối/đánh giá cần thiết và báo rõ lệnh chạy thử.
```

### Bước 7 — Viết dự đoán, nhận xét và biểu đồ

Prompt:

```text
Hãy triển khai src/predict.py, src/insights.py và src/visualization.py theo giao kèo hiện tại.
Kết quả khách hàng cần có CustomerID, RFM, KMeans cluster, dự đoán FutureSpend,
dự đoán ReturnedNext90Days và các nhãn nhận xét.
Tạo các nhãn giải thích được: giá trị cao (Monetary cao và Frequency đủ cao),
mua thường xuyên (Frequency cao), nguy cơ không quay lại (Decision Tree dự đoán 0).
Ngưỡng cần được tính từ phân vị dữ liệu hoặc cấu hình tập trung; ghi rõ trong code.
Các nhóm nhận xét có thể chồng lấn. Tạo biểu đồ số khách hàng mỗi cụm,
Actual vs Predicted cho hồi quy và confusion matrix cho cây quyết định.
Lưu bảng vào outputs/reports/ và metric/biểu đồ vào thư mục outputs phù hợp.
Không tạo dữ liệu giả, không sửa lại thuật toán trong các file model.
```

### Bước 8 — Viết giao diện Streamlit

Prompt:

```text
Hoàn thiện app.py bằng Streamlit, ưu tiên giao diện tiếng Việt dễ demo.
Có các phần: Tổng quan, Phân nhóm K-Means, So sánh mô hình, Danh sách khách hàng, Nhận xét.
Đọc kết quả và mô hình đã lưu; không huấn luyện lại mỗi lần đổi tab.
Có thông báo nếu chưa có data/raw/Online Retail.xlsx hoặc chưa chạy python -m src.train.
Hiển thị bảng metrics đúng loại bài toán, các biểu đồ đã tạo và bộ lọc khách hàng.
Số liệu, biểu đồ và nhận xét phải lấy từ file kết quả thực tế, không viết cứng ví dụ.
Giữ code xử lý AI ở src/, không sao chép thuật toán vào app.py.
Cuối cùng hướng dẫn lệnh chạy streamlit run app.py.
```

### Bước 9 — Kiểm thử và sửa lỗi

Prompt:

```text
Hãy kiểm tra project CustomerRanking từ thư mục gốc.
Chạy qua các bước đọc dữ liệu, làm sạch, tạo đặc trưng, huấn luyện 3 mô hình,
lưu metric/kết quả và mở dashboard. Tìm lỗi import, đường dẫn, tên cột không thống nhất,
rò rỉ nhãn mục tiêu và các trường hợp file chưa tồn tại.
Chỉ sửa lỗi có liên quan; không thay đổi cấu trúc toàn bộ project nếu không cần.
Sau mỗi thay đổi, nêu file đã sửa và lệnh kiểm thử lại.
Không được bịa kết quả chạy nếu không thực sự chạy được.
```

---

## 10. Cài đặt và chạy project

Mở terminal tại thư mục gốc `CustomerRanking/`.

### Tạo môi trường ảo (khuyến nghị)

Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

Nếu máy không có lệnh `py`, thử `python`.

### Cài thư viện

Đảm bảo `requirements.txt` có tối thiểu:

```text
pandas
numpy
scikit-learn
streamlit
matplotlib
openpyxl
joblib
```

Sau đó chạy:

```powershell
python -m pip install -r requirements.txt
```

### Huấn luyện và tạo kết quả

Tải `Online Retail.xlsx` từ trang UCI rồi đặt đúng vào `data/raw/`.

```powershell
python -m src.train
```

Nếu thành công, kiểm tra thư mục `models/`, `outputs/metrics/` và `outputs/reports/`.

### Mở giao diện

```powershell
streamlit run app.py
```

Streamlit sẽ hiển thị địa chỉ local trong terminal; mở địa chỉ đó bằng trình duyệt.

---

## 11. Quy tắc làm việc chung trên GitHub

### Chia việc gợi ý

- **Thành viên 1:** `data_loader.py`, `preprocessing.py`, `feature_engineering.py`.
- **Thành viên 2:** `src/models/kmeans_model.py` và phần phân tích cụm.
- **Thành viên 3:** `src/models/linear_regression_model.py`, đánh giá hồi quy.
- **Thành viên 4:** `src/models/decision_tree_model.py`, đánh giá phân loại.
- **Thành viên 5 (nếu có):** `app.py`, `visualization.py`, `insights.py`.
- **Người tích hợp:** `config.py`, `train.py`, kiểm thử tích hợp, `README.md`.

Nếu ít người hơn, có thể gộp phần việc nhưng vẫn giữ đúng ranh giới file.

### Quy tắc Git

1. Không commit trực tiếp vào nhánh `main` nếu nhóm đang làm theo Pull Request.
2. Mỗi người tạo nhánh riêng, ví dụ `feature/preprocessing`, `feature/kmeans`, `feature/dashboard`.
3. Chỉ sửa các file thuộc phần được giao; không tự ý đổi tên cột/hàm dùng chung.
4. Trước khi ghép, chạy thử phần của mình và ghi rõ lệnh đã chạy.
5. Tạo Pull Request để người tích hợp xem diff và kiểm tra.
6. Không commit `.venv/`, `__pycache__/`, file dữ liệu Excel gốc, file bí mật hoặc file mô hình lớn.
7. Nếu cần đổi giao kèo hàm, thông báo trong nhóm trước khi sửa.

### `.gitignore` nên có

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.env
.streamlit/secrets.toml

# Dataset và dữ liệu sinh ra: mỗi thành viên tự tải/tạo
data/raw/*.xlsx
data/processed/*.csv
models/*.pkl
models/*.joblib
outputs/metrics/*.csv
outputs/figures/*.png
outputs/reports/*.csv
```

> Nếu nhóm muốn lưu kết quả minh họa cuối cùng lên GitHub, hãy chọn riêng file báo cáo cần nộp thay vì commit tất cả file phát sinh.

---

## 12. Checklist nghiệm thu cuối project

### Dữ liệu

- [ ] Có nguồn dataset được ghi trong README và báo cáo.
- [ ] Chạy được tiền xử lý bằng đường dẫn tương đối.
- [ ] Một khách hàng chỉ có một dòng trong bảng đặc trưng.
- [ ] Đặc trưng chỉ dùng dữ liệu lịch sử; nhãn lấy từ kỳ dự đoán sau đó.

### Mô hình

- [ ] K-Means được chuẩn hóa dữ liệu trước khi phân cụm.
- [ ] Linear Regression dự đoán `FutureSpend`.
- [ ] Decision Tree dự đoán `ReturnedNext90Days`.
- [ ] Lưu/đọc lại được ba model bằng `joblib`.
- [ ] Có metric phù hợp với từng bài toán, không gán nhầm metric.

### Kết quả/giao diện

- [ ] Có bảng so sánh metric và giải thích `N/A`.
- [ ] Có các biểu đồ yêu cầu.
- [ ] Có danh sách khách hàng và các bộ lọc cơ bản.
- [ ] Có nhận xét dựa trên kết quả thực tế về khách hàng giá trị cao, mua thường xuyên và nguy cơ không quay lại.
- [ ] Dashboard báo lỗi rõ ràng nếu thiếu file hoặc chưa huấn luyện.

### GitHub/ báo cáo

- [ ] `requirements.txt` đầy đủ.
- [ ] `README.md` có cách cài đặt, cách chạy, nguồn dữ liệu và phân công.
- [ ] Không có đường dẫn tuyệt đối của máy cá nhân.
- [ ] Không có dữ liệu giả hoặc chỉ số được điền thủ công.
- [ ] Project chạy được từ đầu theo đúng thứ tự hướng dẫn.

---

## 13. Nguồn tham khảo

- UCI Machine Learning Repository — Online Retail: <https://archive.ics.uci.edu/dataset/352/online+retail>
- Scikit-learn — Model evaluation: <https://scikit-learn.org/stable/modules/model_evaluation.html>
- Scikit-learn — Clustering evaluation/Silhouette Score: <https://scikit-learn.org/stable/modules/clustering.html>
- Streamlit documentation: <https://docs.streamlit.io/>

**Thông tin trích dẫn dataset gợi ý:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5BW33>. Dataset sử dụng giấy phép CC BY 4.0; hãy ghi nguồn đúng theo yêu cầu giấy phép.
