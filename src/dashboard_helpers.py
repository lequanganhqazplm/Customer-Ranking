"""Validation and presentation helpers for the report-driven dashboard."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


CUSTOMER_COLUMNS = (
    "CustomerID",
    "Cluster",
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "AvgOrderValue",
)
FEATURE_REPORT_COLUMNS = (
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "AvgOrderValue",
)
RFM_COLUMNS = ("Recency", "Frequency", "Monetary")
RANK_ORDER = ("Diamond", "Gold", "Silver", "Bronze")
RANK_DISPLAY_NAMES = {
    "Diamond": "Diamond (Kim cương)",
    "Gold": "Gold (Vàng)",
    "Silver": "Silver (Bạc)",
    "Bronze": "Bronze (Đồng)",
}
SUMMARY_COLUMNS = (
    "Cluster",
    "n_customers",
    "Recency_mean",
    "Recency_median",
    "Frequency_mean",
    "Frequency_median",
    "Monetary_mean",
    "Monetary_median",
)
EVALUATION_COLUMNS = ("k", "inertia", "silhouette_score")
COMMON_METRIC_COLUMNS = (
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "Balanced_Accuracy",
)
MODEL_RESULT_COLUMNS = (
    "Model",
    "ModelType",
    *COMMON_METRIC_COLUMNS,
    "Training_Time",
    "Prediction_Time",
    "Status",
    "Evaluation_Set_ID",
)
MODEL_CATALOG = (
    {"name": "K-Means", "type": "Clustering", "implemented": True},
    {"name": "Logistic Regression", "type": "Classification", "implemented": False},
    {"name": "Decision Tree", "type": "Classification", "implemented": True},
    {"name": "Random Forest", "type": "Classification", "implemented": False},
    {"name": "Gradient Boosting", "type": "Classification", "implemented": False},
    {"name": "KNN", "type": "Classification", "implemented": False},
    {"name": "SVC", "type": "Classification", "implemented": False},
    {"name": "Linear Regression", "type": "Regression", "implemented": True},
    {"name": "Ridge Regression", "type": "Regression", "implemented": False},
    {"name": "Random Forest Regressor", "type": "Regression", "implemented": False},
)


def read_csv_report(path, required_columns=()):
    """Read a report CSV, returning ``(data, error)`` instead of raising."""
    path = Path(path)
    if not path.is_file():
        return None, f"Không tìm thấy tệp {path.name}."

    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None, f"Tệp {path.name} không có dữ liệu."
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        return None, f"Không thể đọc {path.name}: {exc}"

    if data.empty:
        return None, f"Tệp {path.name} không có dữ liệu."

    # cluster_summary.csv is written with the cluster ID as the CSV index.
    if "Cluster" in required_columns and "Cluster" not in data.columns:
        unnamed = [column for column in data.columns if str(column).startswith("Unnamed:")]
        if unnamed:
            data = data.rename(columns={unnamed[0]: "Cluster"})

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        return None, f"Tệp {path.name} thiếu cột bắt buộc: {', '.join(missing)}."
    return data, None


def validate_customer_report(data):
    """Validate dashboard inputs while preserving the model's numeric IDs."""
    if data is None:
        return None, "Không có báo cáo khách hàng để hiển thị."
    if data.empty:
        return None, "Báo cáo khách hàng không có dòng dữ liệu."

    missing = [column for column in CUSTOMER_COLUMNS if column not in data.columns]
    if missing:
        return None, f"Báo cáo khách hàng thiếu cột bắt buộc: {', '.join(missing)}."

    result = data.copy()
    ids = result["CustomerID"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        return None, "Báo cáo khách hàng có CustomerID bị thiếu."
    result["CustomerID"] = ids.map(_normalize_customer_id)
    if result["CustomerID"].duplicated().any():
        return None, "Báo cáo khách hàng có CustomerID trùng lặp."

    clusters = pd.to_numeric(result["Cluster"], errors="coerce")
    if clusters.isna().any() or not np.isfinite(clusters).all() or not np.equal(clusters, np.floor(clusters)).all():
        return None, "Cột Cluster phải chứa các mã cụm số nguyên hợp lệ."
    result["Cluster"] = clusters.astype(int)

    for column in CUSTOMER_COLUMNS[2:]:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            return None, f"Cột {column} có giá trị thiếu hoặc không phải số hợp lệ."
        result[column] = values
    return result, None


def validate_feature_report(data):
    """Validate historical feature rows independently from legacy clusters."""
    if data is None:
        return None, "Không có báo cáo đặc trưng khách hàng để hiển thị."
    if data.empty:
        return None, "Báo cáo đặc trưng khách hàng không có dòng dữ liệu."
    missing = [column for column in FEATURE_REPORT_COLUMNS if column not in data.columns]
    if missing:
        return None, f"Báo cáo đặc trưng thiếu cột bắt buộc: {', '.join(missing)}."
    result = data[list(FEATURE_REPORT_COLUMNS)].copy()
    ids = result["CustomerID"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        return None, "Báo cáo đặc trưng có CustomerID bị thiếu."
    result["CustomerID"] = ids.map(_normalize_customer_id)
    if result["CustomerID"].duplicated().any():
        return None, "Báo cáo đặc trưng có CustomerID trùng lặp."
    for column in FEATURE_REPORT_COLUMNS[1:]:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            return None, f"Cột {column} có giá trị thiếu hoặc không phải số hợp lệ."
        result[column] = values
    return result, None


def build_cluster_summary(customers):
    """Calculate complete cluster statistics from validated customer rows."""
    grouped = customers.groupby("Cluster", sort=True)[list(RFM_COLUMNS)].agg(["mean", "median"])
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
    grouped.insert(0, "n_customers", customers.groupby("Cluster", sort=True).size())
    return grouped.reset_index()[list(SUMMARY_COLUMNS)]


def reconcile_cluster_summary(report, customers):
    """Use summary values when valid and fill absent clusters from customer data."""
    calculated = build_cluster_summary(customers)
    if report is None:
        return calculated, "Không dùng được cluster_summary.csv; thống kê được tính từ customer_clusters.csv."

    report = report.copy()
    if "Cluster" not in report.columns:
        unnamed = [column for column in report.columns if str(column).startswith("Unnamed:")]
        if unnamed:
            report = report.rename(columns={unnamed[0]: "Cluster"})
    missing = [column for column in SUMMARY_COLUMNS if column not in report.columns]
    if missing or report.empty:
        return calculated, "cluster_summary.csv thiếu dữ liệu hoặc cột cần thiết; thống kê được tính từ customer_clusters.csv."

    report = report[list(SUMMARY_COLUMNS)].copy()
    report["Cluster"] = pd.to_numeric(report["Cluster"], errors="coerce")
    if report["Cluster"].isna().any() or not np.isfinite(report["Cluster"]).all():
        return calculated, "cluster_summary.csv có mã cụm không hợp lệ; thống kê được tính từ customer_clusters.csv."
    if not np.equal(report["Cluster"], np.floor(report["Cluster"])).all() or report["Cluster"].duplicated().any():
        return calculated, "cluster_summary.csv có mã cụm trùng lặp hoặc không nguyên; thống kê được tính từ customer_clusters.csv."
    report["Cluster"] = report["Cluster"].astype(int)

    value_columns = [column for column in SUMMARY_COLUMNS if column != "Cluster"]
    for column in value_columns:
        report[column] = pd.to_numeric(report[column], errors="coerce")
    report_values = report[value_columns].to_numpy(dtype=float)
    if not np.isfinite(report_values).all():
        return calculated, "cluster_summary.csv có thống kê không hợp lệ; thống kê được tính từ customer_clusters.csv."

    actual_ids = set(calculated["Cluster"])
    report = report[report["Cluster"].isin(actual_ids)].set_index("Cluster")
    calculated = calculated.set_index("Cluster")
    missing_ids = sorted(actual_ids - set(report.index))
    for cluster_id in missing_ids:
        report.loc[cluster_id, :] = calculated.loc[cluster_id, value_columns]

    # The customer assignments are the source of truth for cluster size.
    report["n_customers"] = calculated["n_customers"]
    result = report.reset_index().sort_values("Cluster").reset_index(drop=True)
    if missing_ids:
        return result, "Một số cụm bị thiếu trong cluster_summary.csv; đã bổ sung thống kê từ customer_clusters.csv."
    return result[list(SUMMARY_COLUMNS)], None


def build_cluster_labels(summary):
    """Name clusters by relative median spend, without inferring VIP status."""
    if summary is None or summary.empty:
        return {}

    ordered = summary.sort_values(
        ["Monetary_median", "Frequency_median", "Recency_median", "Cluster"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    count = len(ordered)
    labels = {}
    if count == 1:
        row = ordered.iloc[0]
        labels[int(row["Cluster"])] = f"Cụm {int(row['Cluster'])} ({int(row['n_customers']):,} khách)"
        return labels

    for index, row in ordered.iterrows():
        if index == 0:
            level = "Chi tiêu trung vị cao nhất"
        elif index == count - 1:
            level = "Chi tiêu trung vị thấp nhất"
        else:
            level = f"Mức chi tiêu trung vị thứ {index + 1}"
        cluster_id = int(row["Cluster"])
        labels[cluster_id] = f"{level} (Cụm {cluster_id}, {int(row['n_customers']):,} khách)"
    return labels


def validate_evaluation_report(data):
    """Return a numeric evaluation table or an explanatory validation error."""
    if data is None:
        return None, "Không có báo cáo đánh giá K-Means."
    missing = [column for column in EVALUATION_COLUMNS if column not in data.columns]
    if missing:
        return None, f"Báo cáo đánh giá thiếu cột bắt buộc: {', '.join(missing)}."
    if data.empty:
        return None, "Báo cáo đánh giá K-Means không có dòng dữ liệu."
    result = data[list(EVALUATION_COLUMNS)].copy()
    for column in EVALUATION_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        return None, "Báo cáo đánh giá K-Means có giá trị không hợp lệ."
    return result, None


def validate_model_results(data):
    """Validate the common rank-metric comparison CSV.

    Legacy ``model_metrics.csv`` uses different metrics and is intentionally
    not accepted as a four-rank model-comparison table.
    """
    if data is None:
        return None, "Chưa có bảng đánh giá chung cho CustomerRank."
    if data.empty:
        return None, "Bảng đánh giá chung không có dòng kết quả."
    missing = [column for column in MODEL_RESULT_COLUMNS if column not in data.columns]
    if missing:
        return None, f"Bảng đánh giá chung thiếu cột bắt buộc: {', '.join(missing)}."
    result = data[list(MODEL_RESULT_COLUMNS)].copy()
    if result["Model"].isna().any() or result["Model"].duplicated().any():
        return None, "Bảng đánh giá chung có tên mô hình thiếu hoặc trùng lặp."
    result["Model"] = result["Model"].astype(str)
    result["Status"] = result["Status"].fillna("").astype(str)
    for column in (*COMMON_METRIC_COLUMNS, "Training_Time", "Prediction_Time"):
        original = result[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_text = original.notna() & original.astype(str).str.strip().ne("") & numeric.isna()
        if invalid_text.any():
            return None, f"Bảng đánh giá chung có giá trị không phải số trong cột {column}."
        result[column] = numeric
    completed = result["Status"].str.lower().eq("completed")
    for column in COMMON_METRIC_COLUMNS:
        values = result.loc[completed, column]
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            return None, f"Mô hình completed thiếu chỉ số hợp lệ trong cột {column}."
        if ((values < 0) | (values > 1)).any():
            return None, f"Chỉ số {column} phải nằm trong khoảng từ 0 đến 1."
    evaluation_ids = result.loc[completed, "Evaluation_Set_ID"].dropna().astype(str).unique()
    warning = None
    if len(evaluation_ids) > 1:
        warning = "Các kết quả dùng nhiều tập test khác nhau; không nên so sánh trực tiếp."
    elif completed.any() and result.loc[completed, "Evaluation_Set_ID"].fillna("").astype(str).str.strip().eq("").any():
        warning = "Thiếu Evaluation_Set_ID ở một số kết quả; không thể xác nhận chúng dùng chung tập test."
    return result, warning


def available_model_results(data):
    """Return completed ranking results keyed by the catalog display name."""
    if data is None or data.empty:
        return {}
    completed = data[data["Status"].str.lower() == "completed"]
    return {row["Model"]: row for _, row in completed.iterrows()}


def validate_test_predictions(data):
    """Validate saved holdout predictions without inferring rank from ClusterID."""
    required = {"CustomerID", "CustomerRank_true", "CustomerRank_predicted", "Cluster"}
    if data is None:
        return None, "Run hiện tại chưa lưu dự đoán test."
    if data.empty:
        return None, "Tệp dự đoán test không có dòng dữ liệu."
    missing = sorted(required - set(data.columns))
    if missing:
        return None, f"Tệp dự đoán test thiếu cột: {', '.join(missing)}."
    result = data.copy()
    if result["CustomerID"].isna().any():
        return None, "Tệp dự đoán có CustomerID bị thiếu."
    result["CustomerID"] = result["CustomerID"].map(_normalize_customer_id)
    if result["CustomerID"].duplicated().any():
        return None, "Tệp dự đoán có CustomerID trùng lặp."
    for column in ("CustomerRank_true", "CustomerRank_predicted"):
        result[column] = result[column].astype("string").str.strip()
        invalid = result[column].isna() | ~result[column].isin(RANK_ORDER)
        if invalid.any():
            return None, f"Tệp dự đoán có nhãn không hợp lệ trong {column}."
    clusters = pd.to_numeric(result["Cluster"], errors="coerce")
    if clusters.isna().any() or not np.isfinite(clusters).all() or not np.equal(clusters, np.floor(clusters)).all():
        return None, "Cột Cluster phải chứa mã cụm số nguyên hợp lệ."
    result["Cluster"] = clusters.astype(int)
    result = result.rename(columns={"Cluster": "ClusterID"})
    return result, None


def comparable_model_results(data):
    """Return completed rows on one verifiable evaluation set plus exclusions."""
    if data is None or data.empty:
        return data, [], "Chưa có kết quả đánh giá CustomerRank."
    completed = data[data["Status"].astype(str).str.lower().eq("completed")].copy()
    if completed.empty:
        return completed, [], "Chưa có mô hình completed để so sánh."
    ids = completed["Evaluation_Set_ID"].fillna("").astype(str).str.strip()
    if ids.eq("").any():
        excluded = completed.loc[ids.eq(""), "Model"].astype(str).tolist()
        completed = completed.loc[~ids.eq("")].copy()
    else:
        excluded = []
    if completed.empty:
        return completed, excluded, "Không có Evaluation_Set_ID hợp lệ để xác nhận cùng tập test."
    # Results are already ordered newest-first by load_ranking_results.
    reference_id = str(completed.iloc[0]["Evaluation_Set_ID"])
    same_set = completed[completed["Evaluation_Set_ID"].astype(str).eq(reference_id)].copy()
    other_models = completed.loc[~completed["Evaluation_Set_ID"].astype(str).eq(reference_id), "Model"].astype(str).tolist()
    excluded.extend(other_models)
    warning = None
    if excluded:
        warning = "Không đưa vào so sánh các mô hình thiếu mã tập test hoặc dùng tập test khác: " + ", ".join(dict.fromkeys(excluded)) + "."
    return same_set, list(dict.fromkeys(excluded)), warning


def validate_dt_predictions(data):
    """Validate Decision Tree holdout predictions."""
    required = {"CustomerID", "CustomerRank_true", "CustomerRank_predicted"}
    if data is None:
        return None, "Run Decision Tree chưa lưu dự đoán test."
    if data.empty:
        return None, "Tệp dự đoán Decision Tree không có dòng dữ liệu."
    missing = sorted(required - set(data.columns))
    if missing:
        return None, f"Tệp dự đoán Decision Tree thiếu cột: {', '.join(missing)}."
    result = data.copy()
    if result["CustomerID"].isna().any():
        return None, "Tệp dự đoán Decision Tree có CustomerID bị thiếu."
    result["CustomerID"] = result["CustomerID"].map(_normalize_customer_id)
    if result["CustomerID"].duplicated().any():
        return None, "Tệp dự đoán Decision Tree có CustomerID trùng lặp."
    for column in ("CustomerRank_true", "CustomerRank_predicted"):
        result[column] = result[column].astype("string").str.strip()
        invalid = result[column].isna() | ~result[column].isin(RANK_ORDER)
        if invalid.any():
            return None, f"Tệp dự đoán Decision Tree có nhãn không hợp lệ trong {column}."
    return result, None


def load_dt_run(runs_dir):
    """Load and validate the latest Decision Tree run artifacts."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None, None, None, None
    candidates = sorted(
        runs_dir.glob("*/ranking_metrics.csv"),
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
            return None, None, None, "Không đọc được thông tin kết quả Decision Tree."
        if not isinstance(metadata, dict) or metadata.get("model") != "Decision Tree":
            continue
        predictions_path = metrics_path.parent / "test_predictions.csv"
        if not predictions_path.is_file():
            return metadata, None, metrics_path.parent, "Chưa có bảng dự đoán Decision Tree để hiển thị."
        try:
            raw = pd.read_csv(predictions_path)
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
            return metadata, None, metrics_path.parent, "Không đọc được bảng dự đoán Decision Tree."
        predictions, error = validate_dt_predictions(raw)
        return metadata, predictions, metrics_path.parent, error
    return None, None, None, None


def _normalize_customer_id(value):
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text
