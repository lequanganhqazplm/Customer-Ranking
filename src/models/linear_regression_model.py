"""Linear Regression model for predicting FutureSpend and mapping to CustomerRank."""

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import (
    FEATURE_COLUMNS,
    RANK_ORDER,
    RankThresholds,
    assign_customer_rank,
    fit_rank_thresholds,
)

FORBIDDEN_FEATURES = {
    "customerid",
    "futurespend",
    "futurefrequency",
    "returnednext90days",
    "customerrank",
    "cluster",
    "clusterid",
}


class LinearRegressionRankModel:
    """Predict future spend using Linear Regression and map to the four CustomerRank tiers.

    The model fits a StandardScaler and LinearRegression on historical features to
    predict continuous ``FutureSpend``. Rank boundaries (q50, q85) are learned from
    the positive training spends and used to map predicted spend into Diamond, Gold,
    Silver, or Bronze.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.model_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.scaler_ = StandardScaler()
        self.thresholds_: RankThresholds | None = None
        self.feature_names_in_: tuple[str, ...] | None = None
        self.n_features_in_: int = 0

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[float] | np.ndarray | pd.Series,
        thresholds: RankThresholds | None = None,
    ):
        """Fit scaler and linear regression model on training data.

        Args:
            X: Training features (e.g. historical RFM metrics).
            y: Continuous target representing future spend (FutureSpend).
            thresholds: Optional pre-fitted RankThresholds. If None, thresholds
                are automatically learned from positive training spend values.
        """
        values, columns = _validate_feature_matrix(X)
        target_spend = _validate_target_vector(y)

        if len(values) != len(target_spend):
            raise ValueError(
                f"Features count ({len(values)}) does not match target count ({len(target_spend)})."
            )
        if len(values) == 0:
            raise ValueError("Training dataset cannot be empty.")

        self.feature_names_in_ = tuple(columns) if columns is not None else tuple(FEATURE_COLUMNS)
        self.n_features_in_ = values.shape[1]

        # Learn rank thresholds from positive spend on train split
        if thresholds is not None:
            self.thresholds_ = thresholds
        else:
            self.thresholds_ = fit_rank_thresholds(target_spend)

        # Scale features and fit linear regression
        scaled_features = self.scaler_.fit_transform(values)
        self.model_.fit(scaled_features, target_spend)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict continuous future spend, clipped at zero (no negative spend).

        Compatible with scikit-learn regressor protocol and ``predict_customer_rank``.
        """
        self._check_is_fitted()
        values, columns = _validate_feature_matrix(X)
        self._check_columns(values, columns)

        scaled_features = self.scaler_.transform(values)
        raw_pred = self.model_.predict(scaled_features)
        # Spend cannot be negative in retail purchase domain
        return np.maximum(raw_pred, 0.0)

    def predict_rank(
        self,
        X: pd.DataFrame | np.ndarray,
        thresholds: RankThresholds | None = None,
    ) -> np.ndarray:
        """Predict the final four-tier CustomerRank (Diamond, Gold, Silver, Bronze)."""
        active_thresholds = thresholds or self.thresholds_
        if active_thresholds is None:
            raise ValueError("Thresholds are required to convert spend predictions to ranks.")

        spend_predictions = self.predict(X)
        return np.asarray(
            assign_customer_rank(
                spend_predictions,
                active_thresholds.q50,
                active_thresholds.q85,
            ),
            dtype=object,
        )

    def get_coefficients(self) -> pd.DataFrame:
        """Return regression coefficients associated with each feature name."""
        self._check_is_fitted()
        coefs = pd.DataFrame(
            {
                "Feature": list(self.feature_names_in_),
                "Coefficient": self.model_.coef_,
            }
        ).sort_values("Coefficient", ascending=False, key=abs).reset_index(drop=True)
        return coefs

    def _check_is_fitted(self):
        if not hasattr(self.model_, "coef_"):
            raise ValueError("LinearRegressionRankModel is not fitted yet. Call 'fit' before predicting.")

    def _check_columns(self, values, columns):
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {values.shape[1]}."
            )
        if self.feature_names_in_ is not None and columns is not None:
            if tuple(columns) != self.feature_names_in_:
                raise ValueError(
                    f"Feature columns {tuple(columns)} do not match training features {self.feature_names_in_}."
                )


def _validate_feature_matrix(X):
    """Validate feature DataFrame or array and reject forbidden target/ID columns."""
    if isinstance(X, pd.DataFrame):
        if X.empty or X.shape[1] == 0:
            raise ValueError("X must be a non-empty feature table.")
        forbidden = sorted(
            column for column in X.columns if str(column).strip().lower() in FORBIDDEN_FEATURES
        )
        if forbidden:
            raise ValueError(f"Target, identifier, and cluster columns cannot be model inputs: {forbidden}")
        values = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        columns = list(X.columns)
    else:
        values = np.asarray(X, dtype=float)
        columns = None

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix.")
    if not np.isfinite(values).all():
        raise ValueError("X must contain finite numeric values only.")
    return values, columns


def _validate_target_vector(y):
    """Validate target FutureSpend vector."""
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Target y must be a non-empty one-dimensional vector.")
    if not np.isfinite(values).all():
        raise ValueError("Target y must contain finite numeric values.")
    if (values < 0).any():
        raise ValueError("Target FutureSpend cannot contain negative actual values.")
    return values


def train_linear_regression(
    X_train: pd.DataFrame | np.ndarray,
    y_train: Sequence[float] | np.ndarray,
    thresholds: RankThresholds | None = None,
) -> LinearRegressionRankModel:
    """Convenience helper to fit and return a LinearRegressionRankModel."""
    model = LinearRegressionRankModel()
    model.fit(X_train, y_train, thresholds=thresholds)
    return model


if __name__ == "__main__":
    import io
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

    print("=" * 60)
    print("DEMO: Huấn luyện và dự đoán với LinearRegressionRankModel")
    print("=" * 60)

    # 1. Tạo tập dữ liệu mẫu mô phỏng hành vi khách hàng
    np.random.seed(42)
    sample_customers = pd.DataFrame(
        {
            "Recency": [5, 45, 120, 15, 80],
            "Frequency": [12, 3, 1, 8, 2],
            "Monetary": [2500.0, 350.0, 45.0, 1200.0, 180.0],
            "TotalQuantity": [150, 20, 2, 70, 10],
            "UniqueProducts": [25, 5, 1, 14, 3],
            "AvgOrderValue": [208.3, 116.7, 45.0, 150.0, 90.0],
        }
    )
    # Chi tiêu thực tế 90 ngày tới
    actual_future_spend = np.array([2100.0, 280.0, 0.0, 950.0, 50.0])

    print("\n[1] Dữ liệu khách hàng mẫu (X):")
    print(sample_customers)
    print("\n[2] Chi tiêu thực tế 90 ngày tới (y):", actual_future_spend)

    # 2. Khởi tạo và huấn luyện mô hình
    model = LinearRegressionRankModel()
    model.fit(sample_customers, actual_future_spend)

    print(f"\n[3] Ngưỡng phân hạng học được từ Train:")
    print(f"    - P50 (Silver -> Gold):    {model.thresholds_.q50:.2f} £")
    print(f"    - P85 (Gold -> Diamond):   {model.thresholds_.q85:.2f} £")

    # 3. Xem hệ số hồi quy (Coefficients)
    print("\n[4] Trọng số các đặc trưng (Feature Coefficients):")
    print(model.get_coefficients().to_string(index=False))

    # 4. Dự đoán chi tiêu và phân hạng
    predicted_spend = model.predict(sample_customers)
    predicted_ranks = model.predict_rank(sample_customers)

    results = sample_customers.copy()
    results["Spend_Actual"] = actual_future_spend
    results["Spend_Predicted"] = np.round(predicted_spend, 2)
    results["Rank_Predicted"] = predicted_ranks

    print("\n[5] Kết quả dự đoán chi tiêu và hạng:")
    print(results[["Monetary", "Recency", "Spend_Actual", "Spend_Predicted", "Rank_Predicted"]].to_string())
    print("\nChạy thử nghiệm hoàn tất thành công!")

