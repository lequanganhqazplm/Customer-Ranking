"""Module for CustomerRanking project.

K-Means clustering model: training, prediction, and evaluation utilities.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
