"""Print the MLflow runs as a Markdown table for the README / report.

    python scripts/export_mlflow_results.py
"""
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
runs = mlflow.search_runs(experiment_names=["malicious-url-detector"], order_by=["start_time ASC"])
if runs.empty:
    sys.exit("No runs found. Run train.py first.")

SKIP = {"algorithm", "sample", "test_size", "seed"}
print("| # | Run | Algorithm | Hyperparameters | Accuracy | Macro-F1 | Weighted-F1 | Malicious recall |")
print("|---|-----|-----------|-----------------|----------|----------|-------------|------------------|")
for i, (_, r) in enumerate(runs.iterrows(), 1):
    params = ", ".join(
        f"{c.split('.', 1)[1]}={r[c]}" for c in runs.columns
        if c.startswith("params.") and c.split(".", 1)[1] not in SKIP and str(r[c]) not in ("None", "nan")
    )
    print(f"| {i} | {r.get('tags.mlflow.runName', '')} | {r['params.algorithm']} | {params} | "
          f"{r['metrics.accuracy']:.4f} | {r['metrics.macro_f1']:.4f} | {r['metrics.weighted_f1']:.4f} | "
          f"{r['metrics.malicious_recall']:.4f} |")
