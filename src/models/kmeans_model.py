"""K-Means clustering utilities and train-only mapping to CustomerRank."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import RANK_ORDER


FORBIDDEN_FEATURES = {
    "customerid",
    "futurespend",
    "futurefrequency",
    "returnednext90days",
    "customerrank",
    "cluster",
    "clusterid",
}


class ClusterRankMapper:
    """Map ordered training clusters to business tiers using train RFM only.

    Per-customer value is the equal-weight mean of train-set percentile ranks
    for Monetary, Frequency, and (1 - Recency percentile). Cluster median value
    orders clusters from Bronze to Diamond. Target labels are not accepted.
    """

    def fit(self, cluster_ids, X_train):
        clusters = _cluster_id_vector(cluster_ids)
        if not isinstance(X_train, pd.DataFrame) or X_train.empty:
            raise ValueError("X_train must be a non-empty DataFrame with RFM columns.")
        if len(clusters) != len(X_train):
            raise ValueError("cluster_ids and X_train must have the same number of rows.")
        rfm_columns = ("Monetary", "Frequency", "Recency")
        missing = [column for column in rfm_columns if column not in X_train.columns]
        if missing:
            raise KeyError(f"Missing business-value features: {', '.join(missing)}")
        rfm = X_train.loc[:, list(rfm_columns)].apply(pd.to_numeric, errors="coerce")
        if rfm.isna().any().any() or not np.isfinite(rfm.to_numpy(dtype=float)).all():
            raise ValueError("RFM values must be finite and numeric.")

        percentile = pd.DataFrame(index=rfm.index)
        percentile["Monetary_pct"] = rfm["Monetary"].rank(pct=True, method="average")
        percentile["Frequency_pct"] = rfm["Frequency"].rank(pct=True, method="average")
        percentile["Recency_value_pct"] = 1.0 - rfm["Recency"].rank(pct=True, method="average")
        percentile["BusinessValueScore"] = percentile[
            ["Monetary_pct", "Frequency_pct", "Recency_value_pct"]
        ].mean(axis=1)

        profiles = []
        for cluster_id in sorted(np.unique(clusters)):
            positions = np.flatnonzero(clusters == cluster_id)
            profile = rfm.iloc[positions]
            score = percentile.iloc[positions]["BusinessValueScore"]
            profiles.append(
                {
                    "ClusterID": int(cluster_id),
                    "n_train": int(len(positions)),
                    "BusinessValueScore": float(score.median()),
                    "Monetary_median": float(profile["Monetary"].median()),
                    "Frequency_median": float(profile["Frequency"].median()),
                    "Recency_median": float(profile["Recency"].median()),
                }
            )
        profile_frame = pd.DataFrame(profiles).sort_values(
            ["BusinessValueScore", "Monetary_median", "Frequency_median", "Recency_median", "ClusterID"],
            ascending=[True, True, True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        cluster_count = len(profile_frame)
        if cluster_count < 4:
            raise ValueError(
                f"Only {cluster_count} non-empty training clusters; at least four are required "
                "to assign four customer ranks without inventing cluster labels."
            )

        low_to_high = ("Bronze", "Silver", "Gold", "Diamond")
        mapping = {}
        for position, row in profile_frame.iterrows():
            tier_index = min(3, int(np.floor((position + 0.5) * 4 / cluster_count)))
            mapping[int(row["ClusterID"])] = low_to_high[tier_index]
        if set(mapping.values()) != set(low_to_high):
            raise ValueError("Training clusters could not represent all four business-value tiers.")

        self.cluster_to_rank_ = mapping
        self.cluster_profiles_ = profile_frame
        self.mapping_strategy_ = "train_rfm_percentile_composite"
        self.mapping_formula_ = "mean(percentile(Monetary), percentile(Frequency), 1 - percentile(Recency))"
        self.classes_ = np.asarray(RANK_ORDER, dtype=object)
        return self

    def map_with_status(self, cluster_ids):
        if not hasattr(self, "cluster_to_rank_"):
            raise ValueError("ClusterRankMapper must be fitted before mapping clusters.")
        values = _cluster_id_vector(cluster_ids)
        unknown = [int(value) for value in np.unique(values) if int(value) not in self.cluster_to_rank_]
        ranks = np.asarray(
            [self.cluster_to_rank_.get(int(value)) for value in values],
            dtype=object,
        )
        return ranks, unknown

    def predict(self, cluster_ids):
        ranks, unknown = self.map_with_status(cluster_ids)
        if unknown:
            raise ValueError(f"No train-derived business mapping for cluster IDs: {unknown}")
        return ranks


class KMeansRankClassifier:
    """Scale and cluster train features, then order clusters by train RFM."""

    def __init__(self, n_clusters=4, random_state=42, n_init=10):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)

    def fit(self, X):
        values, columns = _feature_matrix(X)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("KMeansRankClassifier.fit requires named historical feature columns.")
        if len(values) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} training customers for K-Means.")
        if self.n_clusters < 4:
            raise ValueError("At least four clusters are required to map to all four customer ranks.")

        self.feature_names_in_ = tuple(columns) if columns is not None else None
        self.scaler_ = StandardScaler().fit(values)
        scaled = self.scaler_.transform(values)
        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        cluster_ids = self.model_.fit_predict(scaled)
        if len(np.unique(cluster_ids)) < 4:
            raise ValueError("K-Means produced fewer than four non-empty training clusters.")
        self.rank_mapper_ = ClusterRankMapper().fit(cluster_ids, X)
        self.training_cluster_ids_ = cluster_ids.copy()
        self.training_scaled_ = scaled.copy()
        self.inertia_ = float(self.model_.inertia_)
        self.n_features_in_ = values.shape[1]
        return self

    def predict_clusters(self, X):
        values, columns = _feature_matrix(X)
        self._check_columns(values, columns)
        return self.model_.predict(self.scaler_.transform(values))

    def predict(self, X):
        return self.rank_mapper_.predict(self.predict_clusters(X))

    def predict_with_status(self, X):
        clusters = self.predict_clusters(X)
        ranks, unknown = self.rank_mapper_.map_with_status(clusters)
        return {"clusters": clusters, "ranks": ranks, "unmapped_clusters": unknown}

    def cluster_diagnostics(self):
        """Return unsupervised diagnostics for the training clusters only."""
        scaled = self.training_scaled_
        labels = self.model_.labels_
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
            silhouette = None
        else:
            silhouette = float(silhouette_score(scaled, labels))
        return {"silhouette_score": silhouette, "inertia": self.inertia_}

    def _check_columns(self, values, columns):
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features do not match the fitted feature count.")
        if self.feature_names_in_ is not None and tuple(columns or ()) != self.feature_names_in_:
            raise ValueError("Prediction feature columns must match the training feature columns and order.")


def _feature_matrix(X):
    if isinstance(X, pd.DataFrame):
        if X.empty or X.shape[1] == 0:
            raise ValueError("X must be a non-empty feature table.")
        forbidden = sorted(column for column in X.columns if str(column).strip().lower() in FORBIDDEN_FEATURES)
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


def _cluster_id_vector(cluster_ids):
    values = np.asarray(cluster_ids)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("cluster_ids must be a non-empty one-dimensional vector.")
    if pd.isna(values).any():
        raise ValueError("cluster_ids cannot contain missing values.")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("cluster_ids must be finite integer IDs.") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("cluster_ids must be finite integer IDs.")
    return numeric.astype(int)


def train_kmeans(X: np.ndarray, n_clusters: int = 3,
                 random_state: int = 42, n_init: int = 10) -> dict:
    """Train a K-Means model on the (already-scaled) feature matrix *X*.

    Args:
        X: 2-D array of scaled features (e.g. Recency, Frequency, Monetary).
        n_clusters: Number of clusters to form.
        random_state: Random seed for reproducibility.
        n_init: Number of times the algorithm is run with different centroid seeds.

    Returns:
        Dictionary with keys:
        - ``model``: the fitted ``KMeans`` instance.
        - ``labels``: cluster labels for each sample.
        - ``silhouette``: Silhouette Score (``float``).
        - ``inertia``: within-cluster sum of squares (``float``).
    """
    if X.shape[0] < n_clusters:
        raise ValueError(
            f"Not enough samples ({X.shape[0]}) for {n_clusters} clusters."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(X)

    # Silhouette score requires at least 2 distinct labels
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        sil = float("nan")
    else:
        sil = silhouette_score(X, labels)

    return {
        "model": model,
        "labels": labels,
        "silhouette": sil,
        "inertia": model.inertia_,
    }


def predict_clusters(model: KMeans, X: np.ndarray) -> np.ndarray:
    """Predict cluster labels for new (scaled) data.

    Args:
        model: A fitted ``KMeans`` instance.
        X: 2-D array of scaled features.

    Returns:
        1-D array of predicted cluster labels.
    """
    return model.predict(X)


def evaluate_k_range(X: np.ndarray, k_min: int = 2, k_max: int = 10,
                     random_state: int = 42, n_init: int = 10) -> dict:
    """Run K-Means for a range of *k* values and collect evaluation metrics.

    This is used to produce the Elbow curve (inertia vs k) and
    Silhouette curve (silhouette score vs k).

    Returns:
        Dictionary with keys:
        - ``k_values``: list of k values tried.
        - ``inertias``: list of inertia values.
        - ``silhouettes``: list of silhouette scores.
    """
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        result = train_kmeans(X, n_clusters=k,
                              random_state=random_state, n_init=n_init)
        inertias.append(result["inertia"])
        silhouettes.append(result["silhouette"])

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
    }
