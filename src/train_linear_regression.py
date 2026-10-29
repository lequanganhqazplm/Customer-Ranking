"""Orchestrate the full Linear Regression training pipeline for customer ranking.

Usage (from the project root):
    python -m src.train_linear_regression
    hoặc:
    python -m src.train_linear_regression --output-dir outputs/runs/rank-linear-regression-run-custom

Steps:
1. Load raw transaction data.
2. Clean transactions and create the 90-day forward-looking dataset.
3. Split train/test (80/20) and fit the LinearRegressionRankModel.
4. Evaluate ranking metrics on the holdout test set.
5. Save model, ranking metrics, test predictions, and run metadata.
"""

import argparse
import datetime
import io
import sys
from pathlib import Path

# Ensure stdout can handle UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.data_loader import load_data
from src.train import run_training, save_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and evaluate Linear Regression for CustomerRank prediction."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Target run directory. If omitted, a timestamped directory inside outputs/runs/ is created.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to Online Retail.xlsx (default: data/raw/Online Retail.xlsx).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42).",
    )
    args = parser.parse_args(argv)

    # Resolve output directory
    runs_root = (PROJECT_ROOT / "outputs" / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = runs_root / f"rank-linear-regression-{timestamp}"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        if output_dir.exists():
            parser.error(f"Output directory already exists; choose a new directory: {output_dir}")
        try:
            output_dir.resolve().relative_to(runs_root)
        except ValueError:
            parser.error(f"Output directory must be inside {runs_root}")

    print("=" * 70)
    print("PIPELINE HUẤN LUYỆN HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)")
    print("=" * 70)
    print(f"Đọc dữ liệu từ: {args.data or (PROJECT_ROOT / 'data' / 'raw' / 'Online Retail.xlsx')}")

    transactions = load_data(args.data)
    print(f"Tổng số bản ghi giao dịch thô: {len(transactions):,}")

    print("\nĐang xử lý đặc trưng và huấn luyện mô hình...")
    result = run_training(
        transactions,
        model_name="linear_regression",
        test_size=args.test_size,
        random_state=args.random_state,
    )

    saved_to = save_experiment(result, output_dir)
    print(f"\n[OK] Đã lưu kết quả tại: {saved_to}")

    print("\n" + "-" * 70)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ (TEST SET METRICS):")
    print("-" * 70)
    print(pd.DataFrame([result["result_row"]]).to_string(index=False))

    print("\n" + "-" * 70)
    print("PHÂN BỐ HẠNG KHÁCH HÀNG (TRAIN & TEST):")
    print("-" * 70)
    for split_name in ("train", "test"):
        dist = result["metadata"][f"{split_name}_rank_distribution"]
        summary = " | ".join(
            f"{rank}: {dist[rank]['count']} ({dist[rank]['percent']:.2f}%)"
            for rank in ("Diamond", "Gold", "Silver", "Bronze")
        )
        print(f"- Tập {split_name.title()}: {summary}")

    print("\n" + "-" * 70)
    print("HỆ SỐ HỒI QUY CÁC ĐẶC TRƯNG (FEATURE COEFFICIENTS):")
    print("-" * 70)
    coefs = pd.DataFrame(result["metadata"]["coefficients"])
    print(coefs.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()

