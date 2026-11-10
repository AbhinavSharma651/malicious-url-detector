"""Wrap the four artifacts from the Kaggle run into model/model.joblib.

Use this when you want the deployed model to be exactly the XGBoost model you
trained on Kaggle (instead of re-training it with train.py).

    python scripts/build_bundle_from_kaggle.py --src path/to/kaggle_outputs --macro_f1 0.9xx

Expected files in --src (names as downloaded from Kaggle, the "__1_" suffix is optional):
    malicious_url_model.joblib, label_encoder.joblib, feature_scaler.joblib, feature_columns.joblib

Run it in the project's venv: the xgboost and scikit-learn versions there must be
able to load the pickles (see the version note in README.md).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import FEATURE_COLUMNS  # noqa: E402


def find(src: Path, stem: str) -> Path:
    for pattern in (f"{stem}.joblib", f"{stem}__1_.joblib", f"{stem}*.joblib"):
        hits = sorted(src.glob(pattern))
        if hits:
            return hits[0]
    sys.exit(f"Could not find {stem}*.joblib in {src}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder holding the four Kaggle .joblib files")
    ap.add_argument("--macro_f1", type=float, default=None,
                    help="macro-F1 of this model on your Kaggle test split; stored so train.py will not overwrite it with a worse run")
    args = ap.parse_args()
    src = Path(args.src)

    model = joblib.load(find(src, "malicious_url_model"))
    encoder = joblib.load(find(src, "label_encoder"))
    scaler = joblib.load(find(src, "feature_scaler"))
    columns = list(joblib.load(find(src, "feature_columns")))
    if columns != FEATURE_COLUMNS:
        sys.exit("feature_columns.joblib does not match FEATURE_COLUMNS in features.py")

    import numpy as np

    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": encoder,
        "feature_columns": columns,
        "algorithm": "xgboost",
        "params": {k: v for k, v in model.get_params().items() if v is not None},
        "metrics": {"macro_f1": args.macro_f1} if args.macro_f1 is not None else {},
        "train_stats": {c: {"mean": float(m), "std": float(s)} for c, m, s in
                        zip(columns, scaler.mean_, np.sqrt(scaler.var_))},
        "source": "kaggle",
    }
    out = ROOT / "model" / "model.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    print(f"Wrote {out}")
    if args.macro_f1 is not None:
        (ROOT / "model" / "best_metrics.json").write_text(json.dumps(
            {"macro_f1": args.macro_f1, "algorithm": "xgboost", "params": bundle["params"]}, indent=2))
        print("Wrote model/best_metrics.json")
    else:
        print("Tip: pass --macro_f1 so a weaker train.py run cannot overwrite this model.")


if __name__ == "__main__":
    main()
