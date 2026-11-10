"""Bonus: simple drift check on the prediction log.

Compares the mean/std of every feature seen in logs/predictions.jsonl with the
mean/std saved with the model at training time.

    python check_drift.py                       # default log + model paths
    python check_drift.py --log logs/predictions.jsonl --min_rows 50

Exit code 1 when drift is flagged, so it can run from cron or CI.
Flag rules (rules of thumb, tune them to your traffic):
  * mean shift  |mean_live - mean_train| / std_train > 0.25
  * std ratio   std_live / std_train outside [0.5, 2.0]
Very rare features (training std < 0.1, e.g. count_hash) are skipped because a
handful of requests cannot say anything reliable about them.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(ROOT / "logs" / "predictions.jsonl"))
    ap.add_argument("--model", default=str(ROOT / "model" / "model.joblib"))
    ap.add_argument("--min_rows", type=int, default=30)
    ap.add_argument("--mean_shift", type=float, default=0.25)
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"No log file at {log_path}. Make some predictions first.")
        return 2
    rows = [json.loads(line)["features"] for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < args.min_rows:
        print(f"Only {len(rows)} logged predictions; need at least {args.min_rows} for a meaningful check.")
        return 2

    stats = joblib.load(args.model)["train_stats"]
    cols = list(stats)
    live = np.array([[r[c] for c in cols] for r in rows], dtype=float)

    print(f"{len(rows)} logged predictions vs training data\n")
    print(f"{'feature':28s}{'train mean':>12s}{'live mean':>12s}{'shift (std)':>13s}{'std ratio':>11s}  flag")
    flagged = []
    for i, c in enumerate(cols):
        tm, ts = stats[c]["mean"], stats[c]["std"]
        lm, ls = live[:, i].mean(), live[:, i].std()
        if ts < 0.1:
            print(f"{c:28s}{tm:12.3f}{lm:12.3f}{'-':>13s}{'-':>11s}  (rare, skipped)")
            continue
        shift = abs(lm - tm) / ts if ts > 0 else 0.0
        ratio = ls / ts if ts > 0 else 1.0
        bad = shift > args.mean_shift or not (0.5 <= ratio <= 2.0)
        if bad:
            flagged.append(c)
        print(f"{c:28s}{tm:12.3f}{lm:12.3f}{shift:13.2f}{ratio:11.2f}  {'DRIFT' if bad else ''}")

    if flagged:
        print(f"\nDrift flagged on {len(flagged)} feature(s): {', '.join(flagged)}")
        print("Consider re-collecting data and retraining (python train.py ...).")
        return 1
    print("\nNo drift flagged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
