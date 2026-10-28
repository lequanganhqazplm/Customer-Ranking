"""Streamlit dashboard for customer ranks, model results, and customer lookup."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard_helpers import (
    FEATURE_REPORT_COLUMNS,
    MODEL_CATALOG,
    MODEL_RESULT_COLUMNS,
    RANK_DISPLAY_NAMES,
    RANK_ORDER,
    available_model_results,
    comparable_model_results,
    read_csv_report,
    validate_feature_report,
    validate_model_results,
    validate_test_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("CUSTOMER_RANKING_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUTS_ROOT = Path(os.environ.get("CUSTOMER_RANKING_OUTPUTS_ROOT", PROJECT_ROOT / "outputs"))
PROCESSED_DIR = DATA_ROOT / "processed"
METRICS_DIR = OUTPUTS_ROOT / "metrics"
RUNS_DIR = OUTPUTS_ROOT / "runs"

METRICS = (
    ("Accuracy", "Độ chính xác (Accuracy)"),
    ("Precision_Macro", "Độ chính xác trung bình các hạng (Precision Macro)"),
    ("Recall_Macro", "Khả năng nhận diện trung bình các hạng (Recall Macro)"),
    ("F1_Macro", "Điểm F1 trung bình các hạng (F1-score Macro)"),
    ("Balanced_Accuracy", "Độ chính xác cân bằng (Balanced Accuracy)"),
)
MODEL_DISPLAY_NAMES = {
    "K-Means": "K-Means (phân cụm)",
    "Logistic Regression": "Hồi quy logistic (Logistic Regression)",
    "Decision Tree": "Cây quyết định (Decision Tree)",
    "Random Forest": "Rừng ngẫu nhiên (Random Forest)",
    "Gradient Boosting": "Tăng cường gradient (Gradient Boosting)",
    "KNN": "Láng giềng gần nhất (KNN)",
    "SVC": "Máy véc-tơ hỗ trợ (SVC)",
    "Linear Regression": "Hồi quy tuyến tính (Linear Regression)",
    "Ridge Regression": "Hồi quy Ridge (Ridge Regression)",
    "Random Forest Regressor": "Rừng ngẫu nhiên hồi quy (Random Forest Regressor)",
}
MODEL_TYPE_NAMES = {
    "Clustering": "Phân cụm",
    "Classification": "Phân loại",
    "Regression": "Hồi quy",
}
FEATURE_NAMES = {
    "CustomerID": "Mã khách hàng",
    "CustomerRank_predicted": "Hạng khách hàng dự đoán",
    "ClusterID": "Mã cụm",
    "Recency": "Số ngày từ lần mua gần nhất",
    "Frequency": "Số hóa đơn",
    "Monetary": "Tổng tiền mua hàng (£)",
    "TotalQuantity": "Tổng số lượng sản phẩm",
    "UniqueProducts": "Số loại sản phẩm khác nhau",
    "AvgOrderValue": "Giá trị trung bình mỗi hóa đơn (£)",
}

st.set_page_config(page_title="Phân hạng khách hàng", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .block-container {max-width: 1200px; padding-top: 1.8rem; padding-bottom: 2.5rem;}
      h1, h2, h3 {color: #17324d;}
      [data-testid="stMetric"] {background: #f5f8fa; border: 1px solid #e1e8ee;
        padding: 10px 14px; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_report(file_path: str, required_columns: tuple = ()):
    return read_csv_report(file_path, required_columns)


def load_ranking_results():
    """Load actual results and use only the newest saved result per model."""
    paths = []
    direct_path = METRICS_DIR / "ranking_metrics.csv"
    if direct_path.is_file():
        paths.append(direct_path)
    if RUNS_DIR.is_dir():
        paths.extend(
            sorted(
                RUNS_DIR.glob("*/ranking_metrics.csv"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    frames, warnings = [], []
    for path in paths:
        report, error = load_report(str(path), MODEL_RESULT_COLUMNS)
        if error:
            warnings.append("Một bảng kết quả không đọc được và đã được bỏ qua.")
            continue
        validated, message = validate_model_results(report)
        if validated is None:
            warnings.append("Một bảng kết quả thiếu dữ liệu cần thiết và đã được bỏ qua.")
            continue
        frames.append(validated)
        if message:
            warnings.append("Có kết quả không xác định được cùng tập kiểm tra.")

    if not frames:
        return None, warnings
    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.loc[combined["Model"].duplicated(keep=False), "Model"].unique().tolist()
    if duplicates:
        warnings.append("Có nhiều lần chạy cho cùng mô hình; chỉ dùng kết quả mới nhất.")
        combined = combined.drop_duplicates("Model", keep="first")
    validated, message = validate_model_results(combined)
    if message:
        warnings.append("Một số mô hình không thể xác nhận dùng chung tập kiểm tra.")
    return validated, list(dict.fromkeys(warnings))


def load_latest_rank_run():
    """Load the latest K-Means run that follows the current rank-mapping rule."""
    if not RUNS_DIR.is_dir():
        return None, None, None, None
    candidates = sorted(
        RUNS_DIR.glob("*/ranking_metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for metrics_path in candidates:
        metadata_path = metrics_path.parent / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, None, None, "Không đọc được thông tin kết quả K-Means."
        if not isinstance(metadata, dict) or metadata.get("model") != "K-Means":
            continue
        if metadata.get("cluster_mapping_strategy") != "train_rfm_percentile_composite":
            return None, None, None, "Kết quả K-Means cũ chưa có quy tắc ánh xạ hạng hiện hành."
        predictions, error = load_report(str(metrics_path.parent / "test_predictions.csv"))
        if error:
            return metadata, None, metrics_path.parent, "Chưa có bảng dự đoán để hiển thị."
        predictions, error = validate_test_predictions(predictions)
        return metadata, predictions, metrics_path.parent, error
    return None, None, None, None


customers, customers_error = load_report(str(PROCESSED_DIR / "customer_features.csv"), FEATURE_REPORT_COLUMNS)
if customers is not None:
    customers, customers_error = validate_feature_report(customers)

rank_results, rank_result_warnings = load_ranking_results()
rank_metadata, rank_predictions, rank_run_path, rank_run_error = load_latest_rank_run()
if rank_results is not None and not rank_results.empty and (
    rank_metadata is None or rank_predictions is None or rank_run_error
):
    stale_rank = rank_results["Model"].astype(str).eq("K-Means")
    if stale_rank.any():
        rank_results = rank_results.loc[~stale_rank].reset_index(drop=True)
        rank_result_warnings.append("K-Means chưa có kết quả ánh xạ và dự đoán hợp lệ nên đã được loại khỏi so sánh.")
completed_results = available_model_results(rank_results)
comparison_results, excluded_models, comparison_warning = comparable_model_results(rank_results)

if customers is not None and rank_predictions is not None:
    feature_frame = customers.copy()
    feature_frame["CustomerID"] = feature_frame["CustomerID"].astype(str)
    predictions_for_join = rank_predictions[["CustomerID", "CustomerRank_predicted", "ClusterID"]].copy()
    customers_for_view = feature_frame.merge(
        predictions_for_join, on="CustomerID", how="left", validate="one_to_one"
    )
else:
    customers_for_view = customers.copy() if customers is not None else None
    if customers_for_view is not None:
        customers_for_view["CustomerRank_predicted"] = pd.NA
        customers_for_view["ClusterID"] = pd.NA


def rank_name(value):
    return RANK_DISPLAY_NAMES.get(value, "Chưa có kết quả dự đoán")


def model_display_name(value):
    return MODEL_DISPLAY_NAMES.get(value, value)


def model_status_rows():
    saved = {}
    if rank_results is not None and not rank_results.empty:
        saved = {row["Model"]: row for _, row in rank_results.iterrows()}
    rows = []
    for model in MODEL_CATALOG:
        result = saved.get(model["name"])
        if result is None:
            status = "Đã triển khai, chưa có kết quả" if model["implemented"] else "Chưa triển khai"
        elif str(result["Status"]).lower() == "completed":
            status = "Đã đánh giá"
        else:
            status = "Đã chạy nhưng chưa có kết quả hợp lệ"
        rows.append(
            {
                "Mô hình": model_display_name(model["name"]),
                "Loại mô hình": MODEL_TYPE_NAMES.get(model["type"], model["type"]),
                "Trạng thái": status,
            }
        )
    return pd.DataFrame(rows)


def render_metric_cards(row):
    columns = st.columns(5)
    short_labels = (
        "Độ chính xác",
        "Độ chính xác TB các hạng",
        "Nhận diện TB các hạng",
        "Điểm F1 TB các hạng",
        "Độ chính xác cân bằng",
    )
    for column, (key, label), short_label in zip(columns, METRICS, short_labels):
        column.metric(short_label, f"{float(row[key]):.3f}", help=label)


def render_kmeans_model():
    st.subheader("K-Means (thuật toán phân cụm)")
    st.write(
        "K-Means tự chia khách hàng thành các cụm dựa trên đặc trưng mua hàng. "
        "Hạng trong bảng được gán thêm theo tổng tiền, số hóa đơn và số ngày từ lần mua gần nhất; "
        "đây là quy tắc bổ sung, không phải nhãn mà K-Means tự học."
    )

    result = completed_results.get("K-Means")
    if result is not None:
        st.success("Đã có kết quả phân hạng trên tập kiểm tra.")
        render_metric_cards(result)
    elif rank_run_error:
        st.warning("Chưa thể hiển thị kết quả K-Means vì bảng dự đoán chưa đầy đủ hoặc không hợp lệ.")
    else:
        st.info("Chưa có kết quả đánh giá K-Means.")

    if rank_metadata is None:
        st.info("Chưa có thông tin cụm từ lần đánh giá hiện hành.")
        return

    diagnostics = rank_metadata.get("train_only_clustering_diagnostics", {})
    silhouette = diagnostics.get("silhouette_score")
    inertia = diagnostics.get("inertia")
    profiles = pd.DataFrame(rank_metadata.get("cluster_profiles_train", []))
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Số cụm được tạo", str(len(profiles)))
    kpi2.metric(
        "Độ tách biệt của cụm (Silhouette Score)",
        f"{float(silhouette):.3f}" if silhouette is not None else "Chưa có dữ liệu",
    )
    kpi3.metric(
        "Tổng bình phương khoảng cách trong cụm (Inertia)",
        f"{float(inertia):,.2f}" if inertia is not None else "Chưa có dữ liệu",
    )
    st.caption("Hai chỉ số cụm được tính trên tập huấn luyện; chúng không đo độ chính xác của hạng khách hàng.")
    st.caption("Kết quả dựa trên một mốc thời gian, chưa được kiểm tra qua nhiều giai đoạn.")

    mapping = rank_metadata.get("cluster_to_rank", {})
    if mapping and not profiles.empty:
        profiles["Hạng được gán"] = profiles["ClusterID"].map(
            lambda cluster_id: mapping.get(str(int(cluster_id)), mapping.get(int(cluster_id)))
        )
        profiles["Hạng được gán"] = profiles["Hạng được gán"].map(rank_name)
        profile_columns = {
            "ClusterID": "Mã cụm",
            "n_train": "Số khách hàng trong tập huấn luyện",
            "BusinessValueScore": "Điểm đặc trưng",
            "Monetary_median": "Tổng tiền mua hàng trung vị (£)",
            "Frequency_median": "Số hóa đơn trung vị",
            "Recency_median": "Số ngày từ lần mua gần nhất trung vị",
        }
        profile_view = profiles.rename(columns=profile_columns)
        st.subheader("Đặc điểm cụm và hạng được gán")
        st.dataframe(profile_view, width="stretch", hide_index=True)
        st.caption(
            "Điểm đặc trưng kết hợp thứ hạng tương đối của tổng tiền, số hóa đơn và số ngày từ lần mua gần nhất "
            "trong tập huấn luyện; mua gần đây hơn được xem là tích cực hơn."
        )

    rank_distribution = []
    for split_key, split_label in (("train", "Huấn luyện"), ("test", "Kiểm tra")):
        for rank in RANK_ORDER:
            values = rank_metadata.get(f"{split_key}_rank_distribution", {}).get(rank, {})
            rank_distribution.append(
                {
                    "Tập dữ liệu": split_label,
                    "Hạng khách hàng": RANK_DISPLAY_NAMES[rank],
                    "Số khách hàng": values.get("count", 0),
                    "Tỷ lệ (%)": values.get("percent", 0.0),
                }
            )
    if rank_distribution:
        with st.expander("Phân bố hạng thực tế trong hai tập dữ liệu"):
            st.dataframe(pd.DataFrame(rank_distribution), width="stretch", hide_index=True)

    if rank_predictions is not None and not rank_predictions.empty:
        cluster_counts = rank_predictions.groupby("ClusterID", sort=True).size().rename("Số khách hàng").reset_index()
        cluster_counts = cluster_counts.rename(columns={"ClusterID": "Mã cụm"})
        st.subheader("Số khách hàng theo cụm trong tập kiểm tra")
        st.dataframe(cluster_counts, width="stretch", hide_index=True)
        test_features = (
            customers_for_view.dropna(subset=["ClusterID"]).copy()
            if customers_for_view is not None
            else pd.DataFrame()
        )
        if not test_features.empty:
            chart_data = test_features[["Frequency", "Monetary", "ClusterID"]].copy()
            chart_data = chart_data.rename(columns={"Frequency": "Số hóa đơn", "Monetary": "Tổng tiền mua hàng (£)"})
            chart_data["Mã cụm"] = chart_data.pop("ClusterID").astype(int).map(lambda value: f"Cụm {value}")
            st.subheader("Phân bố khách theo số hóa đơn và tổng tiền mua hàng")
            st.scatter_chart(
                chart_data,
                x="Số hóa đơn",
                y="Tổng tiền mua hàng (£)",
                color="Mã cụm",
                height=330,
            )


with st.sidebar:
    st.title("Phân hạng khách hàng")
    page = st.radio("Điều hướng", ["Tổng quan", "Mô hình", "So sánh mô hình", "Dữ liệu khách hàng"])


if page == "Tổng quan":
    st.title("Tổng quan")
    if customers is None:
        st.warning("Chưa có dữ liệu đặc trưng khách hàng để hiển thị.")
    else:
        total_customers = len(customers)
        total_historical_spend = float(customers["Monetary"].sum())
        profile_count = len(rank_metadata.get("cluster_profiles_train", [])) if rank_metadata else 0
        model_status = "Đã đánh giá" if "K-Means" in completed_results else "Chưa có kết quả đánh giá"
        first, second, third, fourth = st.columns(4)
        first.metric("Khách hàng có dữ liệu lịch sử", f"{total_customers:,}")
        second.metric("Số cụm K-Means (phân cụm)", str(profile_count) if profile_count else "Chưa có dữ liệu")
        third.metric("Tổng tiền mua hàng lịch sử", f"£{total_historical_spend:,.0f}")
        fourth.metric("Trạng thái đánh giá", model_status)

    if rank_predictions is None or rank_predictions.empty:
        st.info("Chưa có kết quả để hiển thị phân bố hạng khách hàng.")
    else:
        actual = rank_predictions["CustomerRank_true"].value_counts().reindex(RANK_ORDER, fill_value=0)
        predicted = rank_predictions["CustomerRank_predicted"].value_counts().reindex(RANK_ORDER, fill_value=0)
        distribution = pd.DataFrame(
            {
                "Hạng khách hàng": [RANK_DISPLAY_NAMES[rank] for rank in RANK_ORDER],
                "Thực tế (tập kiểm tra)": actual.values,
                "Dự đoán (tập kiểm tra)": predicted.values,
            }
        )
        st.subheader("Phân bố hạng khách hàng")
        st.dataframe(distribution, width="stretch", hide_index=True)
        st.bar_chart(distribution.set_index("Hạng khách hàng"), color=["#93a7b8", "#137c8b"])

        most_predicted = predicted.idxmax()
        st.write(
            f"Hạng được dự đoán nhiều nhất là {RANK_DISPLAY_NAMES[most_predicted]} "
            f"({int(predicted[most_predicted]):,}/{len(rank_predictions):,} khách kiểm tra)."
        )
        if "K-Means" in completed_results:
            result = completed_results["K-Means"]
            st.write(
                f"Trên tập kiểm tra, độ chính xác là {result['Accuracy']:.1%}; "
                f"độ chính xác cân bằng là {result['Balanced_Accuracy']:.1%}."
            )

    st.caption(
        "Các hạng phản ánh chi tiêu trong kỳ quan sát dùng để đánh giá, không phải giá trị khách hàng lâu dài. "
        "Bronze (Đồng) biểu thị khách hàng không có chi tiêu dương trong kỳ tương lai được dùng để xây dựng nhãn đánh giá. "
        "Điều này không có nghĩa khách hàng không có giá trị lâu dài hoặc chắc chắn sẽ không mua hàng trong tương lai."
    )


elif page == "Mô hình":
    st.title("Mô hình")
    selected = st.selectbox(
        "Chọn mô hình",
        [item["name"] for item in MODEL_CATALOG],
        format_func=model_display_name,
    )
    model_info = next(item for item in MODEL_CATALOG if item["name"] == selected)
    st.caption(f"Loại mô hình: {MODEL_TYPE_NAMES.get(model_info['type'], model_info['type'])}")
    if selected == "K-Means":
        render_kmeans_model()
    elif not model_info["implemented"]:
        st.info("Chưa triển khai")
    else:
        result = completed_results.get(selected)
        if result is None:
            st.info("Chưa có kết quả đánh giá hợp lệ.")
        else:
            st.success("Đã có kết quả phân hạng trên tập kiểm tra.")
            render_metric_cards(result)


elif page == "So sánh mô hình":
    st.title("So sánh mô hình")
    st.write("Bảng chỉ gồm các mô hình đã đánh giá trên cùng tập kiểm tra và cùng quy ước phân hạng.")
    for warning in rank_result_warnings:
        st.warning(warning)

    if comparison_results is None or comparison_results.empty:
        st.info("Chưa có kết quả đánh giá hợp lệ để so sánh.")
    else:
        if comparison_warning:
            st.warning("Một số mô hình dùng tập kiểm tra khác hoặc thiếu thông tin nên không được đưa vào bảng.")
        display = comparison_results[["Model", *[key for key, _ in METRICS]]].copy()
        display["Model"] = display["Model"].map(model_display_name)
        display = display.rename(columns={"Model": "Mô hình", **{key: label for key, label in METRICS}})
        st.dataframe(display, width="stretch", hide_index=True)

        if len(comparison_results) == 1:
            st.info("Hiện mới có một mô hình được đánh giá. Chưa đủ dữ liệu để kết luận mô hình nào tốt nhất.")
        else:
            st.subheader("Kết quả cao nhất theo từng chỉ số")
            for metric, label in METRICS:
                highest = comparison_results[metric].max()
                names = comparison_results.loc[comparison_results[metric].eq(highest), "Model"].map(model_display_name)
                st.write(f"{label}: {highest:.3f} — {', '.join(names)}.")

    st.subheader("Trạng thái triển khai")
    st.dataframe(model_status_rows(), width="stretch", hide_index=True)


elif page == "Dữ liệu khách hàng":
    st.title("Dữ liệu khách hàng")
    if customers_for_view is None:
        st.warning("Chưa có dữ liệu đặc trưng khách hàng để tra cứu.")
    else:
        if rank_metadata:
            start = pd.to_datetime(rank_metadata.get("prediction_window_start"), errors="coerce")
            end = pd.to_datetime(rank_metadata.get("observation_end"), errors="coerce")
            start_text = start.strftime("%d/%m/%Y %H:%M") if pd.notna(start) else "chưa rõ"
            end_text = end.strftime("%d/%m/%Y %H:%M") if pd.notna(end) else "chưa rõ"
            st.caption(
                f"Nguồn: Online Retail (file Excel gốc). Đặc trưng được tính từ giao dịch trước {start_text}; "
                f"kỳ đánh giá kết thúc {end_text}. "
                "Hạng dự đoán và mã cụm chỉ có cho khách thuộc tập kiểm tra."
            )
        else:
            st.caption("Đặc trưng phản ánh lịch sử mua hàng; chưa có hạng dự đoán hợp lệ để hiển thị.")

        frame = customers_for_view.copy()
        rank_options = ["Tất cả hạng", *[RANK_DISPLAY_NAMES[rank] for rank in RANK_ORDER]]
        cluster_ids = sorted(frame["ClusterID"].dropna().astype(int).unique().tolist())
        cluster_options = ["Tất cả cụm", *[f"Cụm {value}" for value in cluster_ids]]
        sort_labels = {
            "Mã khách hàng": "CustomerID",
            "Số ngày từ lần mua gần nhất": "Recency",
            "Số hóa đơn": "Frequency",
            "Tổng tiền mua hàng (£)": "Monetary",
            "Giá trị trung bình mỗi hóa đơn (£)": "AvgOrderValue",
        }
        with st.expander("Lọc và sắp xếp", expanded=True):
            first, second, third = st.columns(3)
            selected_rank = first.selectbox("Hạng khách hàng dự đoán", rank_options)
            selected_cluster = second.selectbox("Mã cụm", cluster_options)
            customer_query = third.text_input("Mã khách hàng")
            fourth, fifth = st.columns(2)
            sort_label = fourth.selectbox("Sắp xếp theo", list(sort_labels))
            ascending = fifth.radio("Thứ tự", ["Giảm dần", "Tăng dần"], horizontal=True) == "Tăng dần"

        if selected_rank != "Tất cả hạng":
            rank_value = next(rank for rank, label in RANK_DISPLAY_NAMES.items() if label == selected_rank)
            frame = frame[frame["CustomerRank_predicted"] == rank_value]
        if selected_cluster != "Tất cả cụm":
            cluster_value = int(selected_cluster.split()[-1])
            frame = frame[frame["ClusterID"] == cluster_value]
        if customer_query.strip():
            frame = frame[frame["CustomerID"].astype(str).str.contains(customer_query.strip(), regex=False, na=False)]
        frame = frame.sort_values(sort_labels[sort_label], ascending=ascending, na_position="last")

        all_columns = [
            "CustomerID", "CustomerRank_predicted", "ClusterID", "Recency", "Frequency", "Monetary",
            "TotalQuantity", "UniqueProducts", "AvgOrderValue",
        ]
        main_columns = ["CustomerID", "CustomerRank_predicted", "ClusterID", "Recency", "Frequency", "Monetary"]

        def localized_customer_table(data, columns):
            localized = data[columns].rename(columns=FEATURE_NAMES).copy()
            localized["Hạng khách hàng dự đoán"] = localized["Hạng khách hàng dự đoán"].map(rank_name)
            localized["Mã cụm"] = localized["Mã cụm"].map(
                lambda value: f"Cụm {int(value)}" if pd.notna(value) else "Chưa có kết quả dự đoán"
            )
            return localized

        st.caption(f"Hiển thị {len(frame):,} trên {len(customers_for_view):,} khách hàng.")
        st.dataframe(localized_customer_table(frame, main_columns), width="stretch", hide_index=True, height=420)
        st.download_button(
            "Tải danh sách đang lọc (CSV)",
            localized_customer_table(frame, all_columns).to_csv(index=False).encode("utf-8-sig"),
            file_name="danh_sach_khach_hang.csv",
            mime="text/csv",
        )

        if frame.empty:
            st.info("Không tìm thấy khách hàng phù hợp với bộ lọc.")
        else:
            selected_id = st.selectbox("Xem chi tiết mã khách hàng", frame["CustomerID"].tolist())
            matches = frame[frame["CustomerID"] == selected_id]
            if matches.empty:
                st.info("Không tìm thấy khách hàng trong danh sách đang lọc.")
            else:
                customer = matches.iloc[0]
                st.subheader(f"Chi tiết khách hàng {selected_id}")
                first, second, third = st.columns(3)
                first.metric("Hạng khách hàng dự đoán", rank_name(customer["CustomerRank_predicted"]))
                cluster = customer["ClusterID"]
                second.metric("Mã cụm", f"Cụm {int(cluster)}" if pd.notna(cluster) else "Chưa có kết quả dự đoán")
                third.metric("Tổng tiền mua hàng (£)", f"£{customer['Monetary']:,.2f}")
                detail_columns = ["Recency", "Frequency", "TotalQuantity", "UniqueProducts", "AvgOrderValue"]
                detail = customer[detail_columns].to_frame("Giá trị").rename_axis("Đặc trưng").reset_index()
                detail["Đặc trưng"] = detail["Đặc trưng"].map(FEATURE_NAMES)
                detail.loc[detail["Đặc trưng"].eq("Giá trị trung bình mỗi hóa đơn (£)"), "Giá trị"] = customer["AvgOrderValue"]
                st.dataframe(detail, width="stretch", hide_index=True)
