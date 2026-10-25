"""Orchestrate the full K-Means pipeline.

Usage (from the project root):
    python -m src.train_kmeans

Steps:
1. Load raw data.
2. Clean transactions.
3. Build per-customer RFM features.
4. Standardise the RFM columns.
5. Evaluate K from 2 to 10 (Elbow + Silhouette).
6. Train final K-Means model (best K chosen by Silhouette).
7. Save model, scaler, metrics, cluster report, and figures.
"""
import io
import os
import sys
from pathlib import Path

# Ensure stdout can handle UTF-8 on Windows consoles (cp1252 etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so that relative imports work when executed with
# ``python -m src.train_kmeans`` from the project directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.data_loader import load_data
from src.preprocessing import clean_transactions, fit_and_save_scaler
from src.feature_engineering import build_customer_features
from src.models.kmeans_model import train_kmeans, evaluate_k_range

# ── Configurable constants ──────────────────────────────────────────────────
RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]
K_MIN, K_MAX = 2, 10
DEFAULT_K = 3          # fallback if best-silhouette search fails
RANDOM_STATE = 42


def _ensure_dirs():
    """Create output directories if they do not exist."""
    for sub in ("models",
                "outputs/metrics",
                "outputs/figures",
                "outputs/reports",
                "data/processed"):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def _save_elbow_figure(k_values, inertias, path):
    """Save the Elbow Method chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, "bo-", linewidth=2)
    ax.set_xlabel("Số cụm (K)")
    ax.set_ylabel("Inertia (tổng bình phương khoảng cách trong cụm)")
    ax.set_title("Elbow Method – Lựa chọn số cụm K")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


def _save_silhouette_figure(k_values, silhouettes, path):
    """Save the Silhouette Score chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, silhouettes, "rs-", linewidth=2)
    ax.set_xlabel("Số cụm (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score theo số cụm K")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


def _save_cluster_bar(labels, path):
    """Save a bar chart showing the number of customers per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"Cụm {c}" for c in unique], counts, color="steelblue")
    ax.set_xlabel("Cụm K-Means")
    ax.set_ylabel("Số lượng khách hàng")
    ax.set_title("Số lượng khách hàng trong từng cụm K-Means")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(cnt), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


def _save_rfm_scatter(features, labels, path):
    """Save an RFM scatter plot coloured by cluster."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [("Recency", "Frequency"),
             ("Recency", "Monetary"),
             ("Frequency", "Monetary")]
    for ax, (x_col, y_col) in zip(axes, pairs):
        scatter = ax.scatter(features[x_col], features[y_col],
                             c=labels, cmap="viridis", alpha=0.6, s=10)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
    fig.suptitle("Phân bố khách hàng theo RFM (màu = cụm K-Means)", fontsize=13)
    fig.colorbar(scatter, ax=axes, label="Cụm")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Main entry point ────────────────────────────────────────────────────────
def main():
    _ensure_dirs()

    # 1. Load ----------------------------------------------------------------
    print("[1/7] Đọc dữ liệu …")
    raw = load_data()
    print(f"  Số dòng thô: {len(raw):,}")

    # 2. Clean ---------------------------------------------------------------
    print("[2/7] Làm sạch dữ liệu …")
    clean = clean_transactions(raw)
    print(f"  Số dòng sau làm sạch: {len(clean):,}")

    # 3. Features ------------------------------------------------------------
    print("[3/7] Tạo đặc trưng RFM …")
    features = build_customer_features(clean)
    print(f"  Số khách hàng: {len(features):,}")
    print(f"  Cột: {list(features.columns)}")

    # 4. Scale ---------------------------------------------------------------
    print("[4/7] Chuẩn hóa RFM …")
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    scaler = fit_and_save_scaler(features, RFM_COLUMNS, scaler_path)
    X_scaled = scaler.transform(features[RFM_COLUMNS])
    print(f"  Scaler đã lưu: {scaler_path}")

    # 5. Evaluate K range ----------------------------------------------------
    print(f"[5/7] Đánh giá K từ {K_MIN} đến {K_MAX} …")
    eval_results = evaluate_k_range(X_scaled, k_min=K_MIN, k_max=K_MAX,
                                    random_state=RANDOM_STATE)
    k_values = eval_results["k_values"]
    inertias = eval_results["inertias"]
    silhouettes = eval_results["silhouettes"]

    # Print table
    print(f"  {'K':>3}  {'Inertia':>12}  {'Silhouette':>12}")
    for k, iner, sil in zip(k_values, inertias, silhouettes):
        print(f"  {k:>3}  {iner:>12.2f}  {sil:>12.4f}")

    # Save CSVs
    eval_df = pd.DataFrame({
        "k": k_values,
        "inertia": inertias,
        "silhouette_score": silhouettes,
    })
    eval_csv = PROJECT_ROOT / "outputs" / "metrics" / "kmeans_eval.csv"
    eval_df.to_csv(eval_csv, index=False)
    print(f"  [saved] {eval_csv}")

    # Choose best K by highest silhouette
    best_idx = int(np.argmax(silhouettes))
    best_k = k_values[best_idx]
    print(f"  → K tốt nhất theo Silhouette: {best_k} "
          f"(score = {silhouettes[best_idx]:.4f})")

    # 6. Train final model ---------------------------------------------------
    print(f"[6/7] Huấn luyện K-Means với K = {best_k} …")
    result = train_kmeans(X_scaled, n_clusters=best_k,
                          random_state=RANDOM_STATE)
    model = result["model"]
    labels = result["labels"]

    # Save model
    model_path = PROJECT_ROOT / "models" / "kmeans.pkl"
    joblib.dump(model, model_path)
    print(f"  Model đã lưu: {model_path}")
    print(f"  Silhouette Score : {result['silhouette']:.4f}")
    print(f"  Inertia          : {result['inertia']:.2f}")

    # 7. Reports + figures ---------------------------------------------------
    print("[7/7] Xuất báo cáo và biểu đồ …")

    # -- Cluster assignment --
    features = features.copy()
    features["Cluster"] = labels
    report_path = PROJECT_ROOT / "outputs" / "reports" / "customer_clusters.csv"
    features.to_csv(report_path, index=False)
    print(f"  [saved] {report_path}")

    # -- Cluster summary (statistics per cluster) --
    summary = features.groupby("Cluster")[RFM_COLUMNS].agg(["mean", "median", "count"])
    # Flatten multi-level columns
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    # Keep only one count column (they are identical across R/F/M)
    count_col = [c for c in summary.columns if c.endswith("_count")][0]
    summary = summary.rename(columns={count_col: "n_customers"})
    summary = summary[[c for c in summary.columns
                        if not c.endswith("_count")]]
    summary_path = PROJECT_ROOT / "outputs" / "reports" / "cluster_summary.csv"
    summary.to_csv(summary_path)
    print(f"  [saved] {summary_path}")
    print()
    print("── Thống kê các cụm ──")
    print(summary.to_string())

    # -- Metrics row for the unified table (K-Means only) --
    metrics_row = pd.DataFrame([{
        "Model": "K-Means",
        "Task": "Clustering",
        "Accuracy": "N/A",
        "Precision": "N/A",
        "Recall": "N/A",
        "F1_score": "N/A",
        "MAE": "N/A",
        "R2": "N/A",
        "Silhouette": round(result["silhouette"], 4),
        "Inertia": round(result["inertia"], 2),
    }])
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "model_metrics.csv"
    metrics_row.to_csv(metrics_path, index=False)
    print(f"\n  [saved] {metrics_path}")

    # -- Figures --
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    _save_elbow_figure(k_values, inertias, fig_dir / "elbow.png")
    _save_silhouette_figure(k_values, silhouettes, fig_dir / "silhouette.png")
    _save_cluster_bar(labels, fig_dir / "cluster_bar.png")
    _save_rfm_scatter(features, labels, fig_dir / "rfm_scatter.png")

    print("\n✅ Pipeline K-Means hoàn tất.")


if __name__ == "__main__":
    main()
