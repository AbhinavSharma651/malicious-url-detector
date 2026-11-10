"""Train a malicious-URL classifier and track the run with MLflow.

Examples
    python train.py --algorithm xgboost --n_estimators 400 --max_depth 8 --learning_rate 0.1
    python train.py --algorithm random_forest --n_estimators 200 --max_depth 20
    python train.py --algorithm logistic_regression --C 1.0
    mlflow ui --backend-store-uri sqlite:///mlflow.db

The model/model.joblib file is only overwritten when the new run beats the best
macro-F1 stored in model/best_metrics.json (or when --force is given), so
"the best model" is whatever scored highest across all your runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import FEATURE_COLUMNS, extract_features

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "malicious_phish.csv"
CACHE_PATH = ROOT / "data" / ".features_cache.joblib"
MODEL_PATH = ROOT / "model" / "model.joblib"
BEST_PATH = ROOT / "model" / "best_metrics.json"
EXPERIMENT = "malicious-url-detector"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algorithm", choices=["xgboost", "random_forest", "logistic_regression"], default="xgboost")
    p.add_argument("--n_estimators", type=int, default=400, help="trees (xgboost / random_forest)")
    p.add_argument("--max_depth", type=int, default=8, help="tree depth (xgboost / random_forest)")
    p.add_argument("--learning_rate", type=float, default=0.1, help="xgboost only")
    p.add_argument("--subsample", type=float, default=0.9, help="xgboost only")
    p.add_argument("--colsample_bytree", type=float, default=0.9, help="xgboost only")
    p.add_argument("--C", type=float, default=1.0, help="logistic_regression inverse regularisation")
    p.add_argument("--sample", type=int, default=0, help="use only N rows (fast baseline for CI); 0 = all")
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default=None)
    p.add_argument("--force", action="store_true", help="save model.joblib even if it is not the best so far")
    p.add_argument("--no_mlflow", action="store_true", help="skip MLflow logging (debugging only)")
    return p.parse_args(argv)


def load_features(sample: int, seed: int):
    """Read the CSV, drop duplicate URLs, build the 33 features (cached on disk)."""
    if not DATA_PATH.exists():
        sys.exit(f"Dataset not found at {DATA_PATH}. See data/README.md.")
    df = pd.read_csv(DATA_PATH).dropna().drop_duplicates(subset="url").reset_index(drop=True)
    if sample and sample < len(df):
        df = df.sample(sample, random_state=seed).reset_index(drop=True)
        X = pd.DataFrame([extract_features(u) for u in df["url"]])[FEATURE_COLUMNS]
    elif CACHE_PATH.exists():
        X = joblib.load(CACHE_PATH)
    else:
        X = pd.DataFrame([extract_features(u) for u in df["url"]])[FEATURE_COLUMNS]
        joblib.dump(X, CACHE_PATH)
    return X, df["type"]


def build_model(args):
    if args.algorithm == "xgboost":
        from xgboost import XGBClassifier  # imported lazily so other algorithms work without it

        return XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
            subsample=args.subsample, colsample_bytree=args.colsample_bytree, tree_method="hist",
            objective="multi:softprob", eval_metric="mlogloss", random_state=args.seed, n_jobs=-1,
        )
    if args.algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, n_jobs=-1, random_state=args.seed,
        )
    return LogisticRegression(C=args.C, max_iter=1000, random_state=args.seed)


def used_params(args) -> dict:
    """Only log the hyperparameters that matter for the chosen algorithm."""
    common = {"algorithm": args.algorithm, "sample": args.sample, "test_size": args.test_size, "seed": args.seed}
    if args.algorithm == "xgboost":
        keys = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]
    elif args.algorithm == "random_forest":
        keys = ["n_estimators", "max_depth"]
    else:
        keys = ["C"]
    return {**common, **{k: getattr(args, k) for k in keys}}


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    X, y_text = load_features(args.sample, args.seed)

    encoder = LabelEncoder().fit(y_text)
    y = encoder.transform(y_text)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values.astype(float), y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    scaler = StandardScaler().fit(X_tr)  # fitted on the training split only (no leakage)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    model = build_model(args)
    fit_start = time.time()
    model.fit(X_tr_s, y_tr)
    fit_seconds = time.time() - fit_start

    pred = model.predict(X_te_s)
    benign_idx = int(np.where(encoder.classes_ == "benign")[0][0])
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "macro_f1": f1_score(y_te, pred, average="macro"),
        "weighted_f1": f1_score(y_te, pred, average="weighted"),
        # binary view: how often is a malicious URL (any non-benign class) caught / a benign one flagged?
        "malicious_recall": float(np.mean(pred[y_te != benign_idx] != benign_idx)),
        "benign_false_alarm_rate": float(np.mean(pred[y_te == benign_idx] != benign_idx)),
        "train_seconds": fit_seconds,
    }
    report = classification_report(y_te, pred, target_names=encoder.classes_, digits=4)
    cm = confusion_matrix(y_te, pred)

    print(f"\n{args.algorithm}  rows={len(X):,}  fit={fit_seconds:.1f}s")
    print(report)
    print({k: round(v, 4) for k, v in metrics.items()})

    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": encoder,
        "feature_columns": FEATURE_COLUMNS,
        "algorithm": args.algorithm,
        "params": used_params(args),
        "metrics": metrics,
        # unscaled training mean/std, used by check_drift.py
        "train_stats": {c: {"mean": float(m), "std": float(s)} for c, m, s in
                        zip(FEATURE_COLUMNS, X_tr.mean(axis=0), X_tr.std(axis=0))},
    }

    previous_best = json.loads(BEST_PATH.read_text())["macro_f1"] if BEST_PATH.exists() else -1.0
    is_best = metrics["macro_f1"] > previous_best
    if is_best or args.force:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, MODEL_PATH, compress=3)
        BEST_PATH.write_text(json.dumps({"macro_f1": metrics["macro_f1"], "algorithm": args.algorithm,
                                         "params": used_params(args)}, indent=2))
        print(f"Saved {MODEL_PATH.relative_to(ROOT)} (macro-F1 {metrics['macro_f1']:.4f}).")
    else:
        print(f"Not saved: macro-F1 {metrics['macro_f1']:.4f} <= best so far {previous_best:.4f}. Use --force to override.")

    if not args.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=args.run_name or f"{args.algorithm}-{int(t0)}"):
            mlflow.log_params(used_params(args))
            mlflow.log_metrics(metrics)
            mlflow.set_tag("saved_as_model_joblib", str(bool(is_best or args.force)))
            mlflow.log_text(report, "classification_report.txt")
            mlflow.log_text(json.dumps({"labels": encoder.classes_.tolist(), "matrix": cm.tolist()}), "confusion_matrix.json")
            mlflow.log_dict(used_params(args), "params.json")
    print(f"Done in {time.time() - t0:.0f}s.")
    return metrics


if __name__ == "__main__":
    main()
