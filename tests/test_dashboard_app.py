"""Streamlit AppTest smoke checks against the current local reports."""

from streamlit.testing.v1 import AppTest
import pytest

from src.dashboard_helpers import MODEL_CATALOG
from pathlib import Path


def test_navigation_and_every_model_choice_run_without_streamlit_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run()
    assert not app.exception

    navigation = next(widget for widget in app.radio if widget.label == "Điều hướng")
    navigation.set_value("Mô hình").run()
    assert not app.exception
    model_selector = app.selectbox[0]
    displayed_names = [
        "K-Means (phân cụm)", "Hồi quy logistic (Logistic Regression)", "Cây quyết định (Decision Tree)",
        "Rừng ngẫu nhiên (Random Forest)", "Tăng cường gradient (Gradient Boosting)",
        "Láng giềng gần nhất (KNN)", "Máy véc-tơ hỗ trợ (SVC)",
        "Hồi quy tuyến tính (Linear Regression)", "Hồi quy Ridge (Ridge Regression)",
        "Rừng ngẫu nhiên hồi quy (Random Forest Regressor)",
    ]
    assert model_selector.options == displayed_names

    for model_info, display_name in zip(MODEL_CATALOG, displayed_names):
        app.selectbox[0].set_value(display_name).run()
        assert not app.exception
        if not model_info["implemented"]:
            assert any("Chưa triển khai" in item.value for item in app.info)
            assert len(app.metric) == 0

    app.selectbox[0].set_value("K-Means (phân cụm)").run()
    assert not app.exception
    if (Path(__file__).resolve().parents[1] / "outputs" / "runs" / "rank-kmeans-rfm-map-20261028" / "test_predictions.csv").is_file():
        assert any("Silhouette" in metric.label for metric in app.metric)
        assert any("Đặc điểm cụm và hạng được gán" in element.value for element in app.subheader)
        assert all("Evaluation_Set_ID" not in item.value for item in app.caption)
        profile = next(
            table.value for table in app.dataframe
            if {"Mã cụm", "Hạng được gán", "Số khách hàng trong tập huấn luyện"}.issubset(set(table.value.columns))
        )
        assert len(profile) == 4

    for page_name in ("So sánh mô hình", "Dữ liệu khách hàng", "Tổng quan"):
        navigation = next(widget for widget in app.radio if widget.label == "Điều hướng")
        navigation.set_value(page_name).run()
        assert not app.exception
        if page_name == "So sánh mô hình" and (
            Path(__file__).resolve().parents[1] / "outputs" / "runs" / "rank-kmeans-rfm-map-20261028" / "ranking_metrics.csv"
        ).is_file():
            status_table = next(
                table.value for table in app.dataframe
                if {"Mô hình", "Trạng thái"}.issubset(set(table.value.columns))
            )
            assert len(status_table) == 10
            assert (status_table["Trạng thái"] == "Chưa triển khai").sum() == 9
            assert any(
                "Hiện mới có một mô hình được đánh giá. Chưa đủ dữ liệu để kết luận mô hình nào tốt nhất." in item.value
                for item in app.info
            )
            visible_captions = [item.value for item in app.caption]
            assert all("Evaluation_Set_ID" not in caption for caption in visible_captions)


def test_empty_data_and_evaluation_states_are_clear_and_do_not_crash(tmp_path, monkeypatch):
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    monkeypatch.setenv("CUSTOMER_RANKING_DATA_ROOT", str(tmp_path / "empty-data"))
    monkeypatch.setenv("CUSTOMER_RANKING_OUTPUTS_ROOT", str(tmp_path / "empty-outputs"))

    app = AppTest.from_file(app_path).run()
    assert not app.exception
    assert any("Chưa có dữ liệu đặc trưng" in item.value for item in app.warning)

    navigation = next(widget for widget in app.radio if widget.label == "Điều hướng")
    navigation.set_value("Mô hình").run()
    app.selectbox[0].set_value("Logistic Regression").run()
    assert not app.exception
    assert any("Chưa triển khai" in item.value for item in app.info)
    assert len(app.metric) == 0

    navigation = next(widget for widget in app.radio if widget.label == "Điều hướng")
    navigation.set_value("So sánh mô hình").run()
    assert not app.exception
    assert any("Chưa có kết quả đánh giá hợp lệ để so sánh" in item.value for item in app.info)


def test_customer_lookup_distinguishes_predicted_and_unpredicted_customers():
    project_root = Path(__file__).resolve().parents[1]
    predictions = project_root / "outputs" / "runs" / "rank-kmeans-rfm-map-20261028" / "test_predictions.csv"
    if not predictions.is_file():
        pytest.skip("Current real-data run is not available in this checkout.")

    app = AppTest.from_file(project_root / "app.py").run()
    navigation = next(widget for widget in app.radio if widget.label == "Điều hướng")
    navigation.set_value("Dữ liệu khách hàng").run()
    app.text_input[0].set_value("14646").run()
    assert not app.exception
    assert app.metric[0].value == "Diamond (Kim cương)"
    assert app.metric[1].value == "Cụm 2"

    app.text_input[0].set_value("12347").run()
    assert not app.exception
    assert app.metric[0].value == "Chưa có kết quả dự đoán"
    assert app.metric[1].value == "Chưa có kết quả dự đoán"
