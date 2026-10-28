# Hướng dẫn nhóm — Customer Ranking

## 1. Mục tiêu hiện tại

Hệ thống dự đoán hạng chi tiêu của khách hàng trong 90 ngày quan sát tiếp theo, dựa trên lịch sử trước mốc dự đoán:

- **Diamond:** chi tiêu tương lai từ phân vị 85 trở lên.
- **Gold:** chi tiêu từ phân vị 50 đến dưới phân vị 85.
- **Silver:** có chi tiêu dương nhưng dưới phân vị 50.
- **Bronze:** không có chi tiêu hợp lệ trong kỳ quan sát.

Bronze mô tả riêng kỳ 90 ngày đang dự đoán, không kết luận khách hàng mất giá trị lâu dài.

## 2. Cửa sổ thời gian và nhãn

src/feature_engineering.py đặt mốc dự đoán tại ngày giao dịch cuối cùng trừ 90 ngày. Giao dịch có InvoiceDate < prediction_start chỉ dùng để tạo đặc trưng lịch sử. Giao dịch từ mốc đó đến ngày cuối dữ liệu dùng để tính FutureSpend và ReturnedNext90Days. Chỉ khách hàng có lịch sử được đưa vào cohort; khách chỉ xuất hiện trong tương lai bị loại khỏi tập dự đoán.

assign_customer_rank(future_spend, q50, q85) áp dụng ranh giới theo thứ tự:

1. Giá trị thiếu, không hữu hạn, âm hoặc bằng 0 được xếp Bronze. Pipeline dữ liệu làm sạch các giao dịch không hợp lệ trước đó.
2. Chi tiêu dương >= q85 là Diamond.
3. Còn lại, chi tiêu >= q50 là Gold.
4. Chi tiêu dương thấp hơn q50 là Silver.

P50/P85 được tính trên **chi tiêu dương của khách hàng train**; mức 0 được dành riêng cho Bronze nên không kéo hai ngưỡng về 0 khi nhiều khách không quay lại. Khi q50 == q85, điều kiện Diamond được xét trước; Gold có thể không xuất hiện. Hạng không xuất hiện trong một tập vẫn được giữ trong metric macro và được ghi nhận là thiếu, không tự tạo mẫu cho hạng đó. Nếu train không có chi tiêu dương, ngưỡng là 0/0: nhãn 0 vẫn là Bronze, còn chi tiêu dương mới ở test theo quy tắc biên sẽ thành Diamond; cần xem đây là trường hợp không có tham chiếu positive spend, không phải khẳng định mô hình đã học được hạng chi tiêu đó.

Các hàm tạo dữ liệu chỉ trả DataFrame theo mặc định. Nếu truyền output_path, ghi file sẽ từ chối ghi đè file có sẵn, trừ khi caller chủ động đặt overwrite=True.

## 3. Đặc trưng và chống leakage

Đầu vào chuẩn trong FEATURE_COLUMNS:

| Cột | Nguồn |
|---|---|
| Recency | Số ngày từ giao dịch lịch sử gần nhất tới mốc dự đoán |
| Frequency | Số hóa đơn lịch sử riêng biệt |
| Monetary | Tổng TotalPrice lịch sử |
| TotalQuantity | Tổng số lượng mua lịch sử |
| UniqueProducts | Số mã sản phẩm lịch sử khác nhau |
| AvgOrderValue | Monetary lịch sử chia Frequency |

CustomerID chỉ định danh. FutureSpend, FutureFrequency, ReturnedNext90Days, CustomerRank và Cluster không được đưa vào X.

Quy trình trong src.train:

1. Tạo một dòng cho mỗi khách hàng lịch sử.
2. Chia khách hàng train/test với random_state=42; cùng khách hàng không xuất hiện ở cả hai tập.
3. Tính ngưỡng hạng từ FutureSpend của train rồi áp dụng y nguyên cho test.
4. Fit StandardScaler, K-Means và bảng cluster_id -> CustomerRank chỉ trên train.
5. Chỉ sau đó mới dùng nhãn test để tính metric cuối.

Đây là một mốc cắt thời gian duy nhất trên Online Retail, không phải đánh giá cuốn chiếu hoặc kiểm chứng qua nhiều giai đoạn. Kết quả vì vậy mô tả một kỳ quan sát lịch sử.

Lần kiểm tra dữ liệu hiện tại: cutoff `2011-09-10 12:50:00`, observation end `2011-12-09 12:50:00`; P50/P85 chi tiêu dương của 1,531 khách train lần lượt là `591.35` và `1707.955`. Có 4,338 CustomerID trong các giao dịch hợp lệ toàn kỳ; cohort có giao dịch lịch sử trước cutoff gồm 3,370 khách (2,696 train và 674 test). 968 khách không có giao dịch hợp lệ trước cutoff nên không nằm trong cohort dự đoán. Train có Diamond 230, Gold 536, Silver 765, Bronze 1,165; test có Diamond 55, Gold 143, Silver 192, Bronze 284. Tỷ lệ được in ra bởi lệnh huấn luyện và lưu trong run metadata.

Audit Excel gốc: 541,909 dòng raw, 397,884 dòng sau quy tắc làm sạch và 18,532 invoice duy nhất trong dữ liệu hợp lệ. CustomerID được chuẩn hóa về số trước khi group; ID thiếu, không hữu hạn, không phải số hoặc không nguyên bị loại vì không thể định danh khách nhất quán. Hiện tại không có bước loại exact duplicate; còn 5,192 dòng trùng chính xác sau lọc. Do chưa có quyết định nghiệp vụ khẳng định đó là bản ghi nhập trùng, không tự loại chúng. `TotalPrice` được tính đúng một lần trong preprocessing là `Quantity * UnitPrice`; feature engineering cộng cột này theo kỳ lịch sử.

## 4. Huấn luyện K-Means và lưu kết quả

K-Means là mô hình phân cụm không giám sát. Mã cụm là số nguyên gốc từ mô hình, không có nghĩa Diamond/Gold/Silver/Bronze. Không dùng đa số CustomerRank để ánh xạ vì nhiều cụm có thể cùng đa số một lớp, làm mất Gold/Silver hoặc các hạng khác. `ClusterRankMapper` chỉ nhận `cluster_ids` và `X_train`, không nhận nhãn; nó tính điểm RFM trên train bằng trung bình percentile của Monetary, Frequency và `1 - percentile(Recency)`. Các cụm được sắp từ điểm thấp tới cao và ánh xạ theo thứ tự Bronze, Silver, Gold, Diamond. Với nhiều hơn bốn cụm, các cụm liền kề trong thứ tự điểm có thể cùng hạng. Cách này tạo ra quy tắc xếp hạng nhất quán dựa trên đặc trưng kinh doanh, nhưng không biến cluster thành nhãn thật và không bảo đảm mọi hạng sẽ xuất hiện trong mọi test split. Nếu test có cluster không nằm trong mapping, dự đoán báo lỗi; không dùng nhãn test hoặc fallback theo đa số để bịa hạng.

Mặc định pipeline rank dùng 4 cụm để khởi đầu việc ánh xạ bốn hạng; đây chỉ là cấu hình thử nghiệm, không chứng minh dữ liệu có đúng bốn cụm. Silhouette và Inertia được báo cáo riêng trên cụm train. Chúng không nằm trong năm metric chung và không đánh giá độ chính xác CustomerRank.

Lệnh huấn luyện yêu cầu một thư mục run **chưa tồn tại**:

    python -m src.train --output-dir outputs/runs/rank-kmeans-run-01

Có thể truyền --data, --clusters, --test-size và --random-state. Run lưu model.joblib, ranking_metrics.csv, test_predictions.csv và run_metadata.json trong thư mục mới. Lệnh không ghi đè artifact cũ. Không chạy lệnh trên dữ liệu thật khi chưa chọn một output path mới.

Pipeline mô tả cụm cũ python -m src.train_kmeans nay dừng trước khi đọc dữ liệu nếu phát hiện bất kỳ đầu ra cố định nào đã tồn tại. Nó không ghi đè models/, outputs/ hoặc file processed cũ.

### Huấn luyện Decision Tree

Decision Tree phân loại trực tiếp CustomerRank bằng đúng FEATURE_COLUMNS, không dùng scaler. Mặc định cây được giới hạn với max_depth=5, min_samples_split=20, min_samples_leaf=10 và random_state=42. Pipeline dùng split_customer_dataset với test_size=0.2 và random_state=42; P50/P85 và nhãn train/test được tạo theo quy tắc train-only hiện hành. Test chỉ dùng để đánh giá, không dùng huấn luyện hoặc chọn tham số.

Chọn một thư mục run mới trước khi chạy:

    python -m src.train_decision_tree --output-dir outputs/runs/rank-decision-tree-run-01

Có thể truyền --data, --test-size, --random-state, --max-depth, --min-samples-split và --min-samples-leaf. src.data_loader.load_data đọc Excel, kiểm tra file và các cột giao dịch bắt buộc, không sửa dữ liệu nguồn. Run lưu model.joblib, ranking_metrics.csv, test_predictions.csv và run_metadata.json; thư mục đã tồn tại sẽ bị từ chối. Evaluation_Set_ID dùng cùng phép băm CustomerID của K-Means để xác nhận cùng test split. Dashboard đọc metadata và dự đoán từ run Decision Tree, hiển thị metric, cấu hình mô hình, phân bố nhãn và bảng dự đoán; run thiếu hoặc sai schema không được tính vào so sánh.

## 5. API đánh giá chung

Trong src/evaluate.py:

    evaluate_classification(y_true_rank, y_pred_rank)
    evaluate_regression(y_true_future_spend, y_pred_future_spend, train_thresholds)
    evaluate_clustering(X_train_scaled, cluster_ids, inertia=None)
    make_model_result(model, model_type, metrics, ...)
    write_model_results(records, path, overwrite=False)

Hai hàm classification và regression đều trả cùng năm metric trên **hạng cuối cùng**:

- Accuracy
- Precision_Macro
- Recall_Macro
- F1_Macro
- Balanced_Accuracy

Precision/Recall/F1 macro cố định đủ bốn lớp và dùng zero_division=0. Kết quả cũng chứa số lượng hạng thật/dự đoán và các lớp vắng mặt. Regression phải nhận ngưỡng đã fit trên train; dự đoán chi tiêu âm được chặn về 0 trước khi đổi hạng. API clustering chỉ trả Silhouette/Inertia và số cụm. Metric CSV dùng MODEL_RESULT_COLUMNS, có thêm loại mô hình, thời gian, trạng thái và Evaluation_Set_ID; ghi CSV mặc định không ghi đè.

Ví dụ tích hợp mô hình mới:

1. Fit mô hình trên train[FEATURE_COLUMNS] và nhãn train phù hợp.
2. Classifier trả một trong bốn nhãn trực tiếp; regressor trả spend rồi dùng ngưỡng train; clusterer phải có bảng ánh xạ học trên train.
3. Dùng cùng test customer IDs và cùng ngưỡng train.
4. Gọi evaluate_classification hoặc evaluate_regression rồi make_model_result.
5. Chỉ thêm dòng vào bảng so sánh nếu test split dùng chung; nếu không, giữ Evaluation_Set_ID khác nhau để dashboard cảnh báo.
6. Thêm estimator thật vào MODEL_CATALOG sau khi có huấn luyện, dự đoán và test phù hợp. Không đánh dấu mô hình chỉ vì nó xuất hiện trong danh sách UI.

## 6. Trạng thái 10 mô hình

| Mô hình | Loại | Trạng thái |
|---|---|---|
| K-Means | Clustering, map sang rank | Đã chạy trên Excel gốc, holdout 674 khách, mapping train-only; kết quả nằm trong run hiện hành |
| Logistic Regression | Classification | Chưa triển khai |
| Decision Tree | Classification | Đã triển khai và có kết quả holdout trong `outputs/runs/rank-decision-tree-run-01/` |
| Random Forest | Classification | Chưa triển khai |
| Gradient Boosting | Classification | Chưa triển khai |
| KNN | Classification | Chưa triển khai |
| SVC | Classification | Chưa triển khai |
| Linear Regression | Regression | Chưa triển khai |
| Ridge Regression | Regression | Chưa triển khai |
| Random Forest Regressor | Regression | Chưa triển khai |

src.predict.predict_customer_rank chuẩn bị giao kèo đầu ra chung cho ba loại mô hình. Nó không thay cho việc triển khai hoặc đánh giá các estimator còn lại.

## 7. Dashboard Streamlit

Chạy:

    streamlit run app.py

Sidebar có bốn mục: **Tổng quan**, **Mô hình**, **So sánh mô hình**, **Dữ liệu khách hàng**. Trong Mô hình, chọn một trong 10 tên. K-Means đọc mapping và dự đoán test từ run có metadata hiện hành; Decision Tree hiển thị kết quả và dự đoán test riêng từ run có metadata tương ứng. Báo cáo feature lịch sử được dùng riêng để tra cứu và không được hiểu là dự đoán cả cohort. CustomerRank dự đoán và ClusterID trong trang tra cứu chỉ có cho các khách ở holdout K-Means, không được thay bằng đầu ra Decision Tree. Mô hình chưa triển khai hiển thị trạng thái riêng. Bảng so sánh chỉ lấy các dòng completed có cùng Evaluation_Set_ID; dòng thiếu ID hoặc dùng split khác bị loại khỏi so sánh và được ghi rõ. `model_metrics.csv` cũ có Silhouette/Inertia không được coi là bảng năm metric.

`customer_clusters.csv` và `cluster_summary.csv` là báo cáo phân cụm lịch sử; ClusterID cũ không phải nhãn CustomerRank tương lai. Dashboard dùng `data/processed/customer_features.csv` cho tra cứu, không đọc/hiển thị cluster cũ như cluster của run phân hạng. CustomerRank_predicted và ClusterID được ghép từ test_predictions.csv của run hiện hành; khách không nằm trong holdout sẽ để trống hai cột đó. Nguồn và cutoff từ metadata được hiển thị. Khi file thiếu hoặc sai schema, dashboard hiển thị trạng thái thay vì crash hoặc tạo số mặc định.

Run dữ liệu thật K-Means và Decision Tree được lưu riêng trong `outputs/runs/`. Dashboard đọc metric từ CSV đã lưu, không hardcode chỉ số; chỉ so sánh các mô hình có cùng Evaluation_Set_ID. Chỉ số K-Means train-only (Silhouette/Inertia) là chẩn đoán phân cụm, tách biệt với năm chỉ số phân hạng.

## 8. Kiểm thử và an toàn artifact

Chạy bộ test mà không dùng dataset thật:

    python -m pytest -q -p no:cacheprovider --basetemp <thu_muc_tam_duoc_phep>

Test tạo giao dịch tổng hợp và workbook Excel tổng hợp, chỉ ghi vào thư mục tạm. Có thể chạy riêng các test pipeline mới:

    python -m pytest -q -p no:cacheprovider --basetemp <thu_muc_tam_duoc_phep> tests/test_decision_tree_pipeline.py tests/test_data_loader.py

Không chạy huấn luyện trên dữ liệu thật trong kiểm thử mặc định. Trước khi dùng CLI trên dữ liệu thật, chọn --output-dir mới. Không xóa hoặc thay thế data/raw/Online Retail.xlsx, model, CSV, hình hoặc báo cáo hiện có.

## 9. Artifact lịch sử và giới hạn hiện tại

Các file models/kmeans.pkl, models/scaler.pkl, báo cáo khách hàng/cụm, biểu đồ và model_metrics.csv đang có là artifact phân cụm lịch sử. Chúng được giữ nguyên. Pipeline cũ đã fit trên toàn bộ đặc trưng trước đây, không có nhãn tương lai và không có mapping cụm sang CustomerRank; do vậy chỉ dùng chúng cho trang xem phân cụm, không dùng để tuyên bố năm metric phân hạng hoặc chọn mô hình tốt nhất.

K-Means nén hành vi thành cụm và ánh xạ từng cụm sang một hạng duy nhất; trong một cụm có thể có nhiều hạng thật. Mapping vì thế làm mất thông tin và có thể kém hơn classifier giám sát. Cần đánh giá trên test và thêm đánh giá theo nhiều cutoff thời gian trước khi khái quát hiệu năng theo thời gian.

Nguồn dữ liệu: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), Chen (2015), DOI: 10.24432/C5BW33, giấy phép CC BY 4.0. Tiền tệ trong dữ liệu là Bảng Anh (£); quy tắc làm sạch bỏ qua hủy/trả hàng, chỉ giữ giao dịch mua dương.
