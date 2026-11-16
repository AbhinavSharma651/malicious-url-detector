# Malicious URL Detector

**CSL236 End-Term Project** — production-style machine-learning system with tracked experiments, validated API, tests, Docker, CI, deployment and monitoring.

## 1. Project overview

| Item | Description |
|---|---|
| Project | Malicious URL Detector |
| Problem | Classify a URL as benign, defacement, malware or phishing |
| Input | URL text only |
| Features | 33 lexical URL features |
| Classes | `benign`, `defacement`, `malware`, `phishing` |
| Output | Predicted class, class probabilities, `P(malicious) = 1 - P(benign)`, risk level and rule-based safety tips |
| Network behaviour | No page download and no network calls during feature extraction/prediction |

## 2. Dataset

| Item | Value |
|---|---|
| Source | [Kaggle — Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) by sid321axn |
| Original rows | 651,191 |
| Rows after URL de-duplication | 641,119 |
| Target column | `type` |
| Input column | `url` |
| Train/test split | 70% / 30%, stratified |
| Random state | `42` |
| Scaling | Scaler fitted on the training split only |
| Feature consistency | `features.py` is shared by training and serving |

### Class distribution after de-duplication

| Class | Rows |
|---|---:|
| benign | 428,080 |
| defacement | 95,308 |
| phishing | 94,086 |
| malware | 23,645 |
| **Total** | **641,119** |

## 3. Data contract and URL features

The user supplies a URL through `POST /predict` (4–2048 characters, with no spaces or control characters). The server derives the 33 features listed below.

`POST /predict/features` accepts the same 33 features directly. Every feature specification is enforced by the Pydantic schema (`features.FEATURE_SPECS` → `app.FeaturesRequest`). Wrong types, out-of-range values, missing fields, null, NaN or unknown fields return **HTTP 422**.

| Validation rule | Requirement |
|---|---|
| Strict integer fields | A string such as `"60"` or a float such as `60.5` is not accepted for an `int` field |
| Digit and letter counts | `count_digits + count_letters <= url_length` |
| Hostname length | `hostname_length <= url_length` |
| Path length | `path_length <= url_length` |
| Ratios | `digit_ratio + letter_ratio <= 1` |

### Feature specification

| # | Feature | Type | Allowed range / values | Why it matters |
|---:|---|---|---|---|
| 1 | `url_length` | int | 1–2048 | Phishing URLs tend to be long, to hide the real domain |
| 2 | `hostname_length` | int | 0–2048 | Very long hostnames often imitate brands |
| 3 | `path_length` | int | 0–2048 | Long paths carry payload names and redirect chains |
| 4 | `count_dot` | int | 0–2048 | Many dots mean many (sub)domains |
| 5 | `count_hyphen` | int | 0–2048 | Hyphens are common in look-alike domains |
| 6 | `count_underscore` | int | 0–2048 | Underscores appear in generated/defaced paths |
| 7 | `count_slash` | int | 0–2048 | Directory depth of the URL |
| 8 | `count_question` | int | 0–2048 | Query string present (tracking / injected params) |
| 9 | `count_equals` | int | 0–2048 | Number of query parameters |
| 10 | `count_at` | int | 0–2048 | `@` can hide the real destination |
| 11 | `count_and` | int | 0–2048 | Number of chained query parameters |
| 12 | `count_percent` | int | 0–2048 | Percent-encoding can obfuscate content |
| 13 | `count_hash` | int | 0–2048 | Fragment markers |
| 14 | `count_excl` | int | 0–2048 | Rare `!` character |
| 15 | `count_tilde` | int | 0–2048 | Home-directory paths, often compromised hosts |
| 16 | `count_comma` | int | 0–2048 | Rare `,` character |
| 17 | `count_plus` | int | 0–2048 | Encoded spaces / obfuscation |
| 18 | `count_star` | int | 0–2048 | Rare `*` character |
| 19 | `count_colon` | int | 0–2048 | Scheme and port separators |
| 20 | `count_digits` | int | 0–2048 | Digit-heavy URLs are often auto-generated |
| 21 | `count_letters` | int | 0–2048 | Readable content in the URL |
| 22 | `digit_ratio` | float | 0.0–1.0 | Share of digits in the URL |
| 23 | `letter_ratio` | float | 0.0–1.0 | Share of letters in the URL |
| 24 | `has_ip` | int | {0, 1} | Raw IP address instead of a domain name |
| 25 | `num_subdomains` | int | 0–2048 | Deep subdomain chains imitate trusted brands |
| 26 | `uses_https_token` | int | {0, 1} | `https` text inside an `http://` URL (fake security cue) |
| 27 | `has_at_symbol` | int | {0, 1} | Presence of `@` in the URL |
| 28 | `has_double_slash_redirect` | int | {0, 1} | `//` after the scheme signals a redirect |
| 29 | `is_shortened` | int | {0, 1} | URL shortener hides the destination |
| 30 | `num_suspicious_words` | int | 0–15 | Words like login/verify/secure used in lures |
| 31 | `entropy` | float | 0.0–8.0 | Random-looking strings have high Shannon entropy |
| 32 | `tld_length` | int | 0–63 | Unusual top-level domains |
| 33 | `abnormal_url` | int | {0, 1} | Hostname is not present in the URL as written |
| — | `type` (target) | category | benign, defacement, malware, phishing | Label; this is the prediction target, not an input feature |

## 4. Repository layout

| Path | Purpose |
|---|---|
| `data/malicious_phish.csv` | Dataset |
| `data/README.md` | Dataset notes |
| `model/model.joblib` | Best model bundle: model, scaler, label encoder, feature list and training statistics |
| `static/index.html` | Web UI |
| `tests/test_app.py` | Pytest suite |
| `.github/workflows/ci.yml` | GitHub Actions workflow |
| `app.py` | FastAPI app, Pydantic schemas and prediction logging |
| `features.py` | URL-to-33-feature extraction, data contract and rule-based tips |
| `train.py` | Training script with MLflow tracking and CLI hyperparameters |
| `check_drift.py` | Drift check on logged inputs |
| `scripts/build_bundle_from_kaggle.py` | Builds a model bundle from the Kaggle-trained model |
| `scripts/export_mlflow_results.py` | Exports MLflow results |
| `Dockerfile` | Docker image definition |
| `requirements.txt` | Project/training dependencies |
| `requirements-serve.txt` | Serving dependencies |
| `.gitignore` | Git ignore rules |

## 5. Setup and run locally

### Environment setup

| Step | Windows PowerShell | macOS / Linux |
|---|---|---|
| Create virtual environment | `python -m venv venv` | `python3 -m venv venv` |
| Activate environment | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` | `pip install -r requirements.txt` |

### Training commands

Each training run is logged to MLflow.

| Experiment | Command |
|---|---|
| Logistic regression baseline | `python train.py --algorithm logistic_regression --C 1.0` |
| Random forest configuration | `python train.py --algorithm random_forest --n_estimators 200 --max_depth 20` |
| XGBoost configuration | `python train.py --algorithm xgboost --n_estimators 200 --max_depth 6 --learning_rate 0.1` |
| Larger XGBoost configuration | `python train.py --algorithm xgboost --n_estimators 400 --max_depth 8 --learning_rate 0.1` |
| Random forest, 20k sample | `python train.py --algorithm random_forest --sample 20000 --run_name RF_Test` |
| Random forest, 80k sample | `python train.py --algorithm random_forest --sample 80000 --run_name Random_Forest_Experiment` |
| Logistic regression, 80k sample | `python train.py --algorithm logistic_regression --sample 80000 --run_name Logistic_Regression_Experiment` |
| XGBoost, 80k sample | `python train.py --algorithm xgboost --sample 80000 --run_name XGBoost_Experiment` |
| Final full-dataset XGBoost | `python train.py --algorithm xgboost --force --run_name Final_Model_Full_Dataset` |

### Start the tools and application

| Purpose | Command | Default address / notes |
|---|---|---|
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | `http://127.0.0.1:5000` |
| FastAPI server | `uvicorn app:app --reload --port 8000` | `http://127.0.0.1:8000` |
| API documentation | Open `/docs` while the server is running | `http://127.0.0.1:8000/docs` |
| Run tests | `pytest -v --tb=short` | Displays test results in the terminal |

### Training behaviour and model bundle

| Item | Behaviour |
|---|---|
| Model replacement | `train.py` only overwrites `model/model.joblib` when a run beats the best macro-F1 so far |
| Best-score record | `model/best_metrics.json` |
| Force replacement | Add `--force` to override the best-score guard |
| Feature cache | `data/.features_cache.joblib` |
| Cache tracking | The feature cache is git-ignored |
| Full feature build | The first full run builds features for all 641k URLs; this takes about a minute |

### Training experiments and results

The sample experiments below are the results shown in the recorded terminal runs. The 20k and 80k sample experiments use an 80/20 train/test split of the selected sample. The final run uses the full 641,119-row de-duplicated dataset and the documented 70/30 stratified split. Metrics are reported as shown by the training script.

| Run | Algorithm | Rows used | Accuracy | Macro-F1 | Weighted-F1 | Malicious recall | Benign false-alarm rate | Fit time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Random_Forest_Experiment` | Random Forest | 80,000 | 0.8069 | 0.6945 | 0.7825 | — | — | 1.3 s |
| `Logistic_Regression_Experiment` | Logistic Regression | 80,000 | 0.7659 | 0.6114 | 0.7286 | 0.4378 | 0.0525 | 1.2 s |
| `XGBoost_Experiment` | XGBoost | 80,000 | 0.8630 | 0.8201 | 0.8586 | 0.7521 | 0.0617 | 4.9 s |
| `Final_Model_Full_Dataset` | XGBoost | 641,119 | 0.8801 | 0.8452 | 0.8759 | 0.7738 | 0.0510 | 21.5 s |

**Selected model:** XGBoost trained on the full de-duplicated dataset. It achieved the highest macro-F1 among the displayed experiments (0.8452). The final run was explicitly forced with `--force`.

### Final model: per-class test performance

| Class / metric | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| benign | 0.8940 | 0.9490 | 0.9207 | 128,424 |
| defacement | 0.8970 | 0.8599 | 0.8781 | 28,592 |
| malware | 0.9832 | 0.8719 | 0.9242 | 7,094 |
| phishing | 0.7454 | 0.5890 | 0.6581 | 28,226 |
| **Accuracy** | — | — | **0.8801** | **192,336** |
| **Macro average** | **0.8799** | **0.8175** | **0.8452** | **192,336** |
| **Weighted average** | **0.8759** | **0.8801** | **0.8759** | **192,336** |

Phishing remains the weakest class by F1-score (0.6581). Overall accuracy should therefore not be interpreted as equally strong performance across all classes.

### Use a model trained on Kaggle instead of retraining

| Purpose | Command / detail |
|---|---|
| Build the bundle | `python scripts/build_bundle_from_kaggle.py --src <folder> --macro_f1 <value>` |
| scikit-learn version used to write the pickles | 1.6.1 |
| Local dependency | `requirements.txt` pins scikit-learn 1.6.1 |
| XGBoost compatibility | Check `xgboost.__version__` on Kaggle and pin the same major/minor version of `xgboost-cpu` locally if loading fails |

## 6. Docker

| Step | Command / detail |
|---|---|
| Build image | `docker build -t malicious-url-detector .` |
| Run container | `docker run -d -p 8000:8000 --restart unless-stopped -v mud-logs:/app/logs malicious-url-detector` |
| Base image | `python:3.11-slim` |
| Dependencies in image | `requirements-serve.txt` only |
| Excluded from serving image | MLflow, pytest and pandas |
| Image size | Run `docker images` after building and record the displayed size |

## 7. API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | UI: prediction, probability bars and 2–3 rule-based tips |
| GET | `/health` | Liveness and model information |
| POST | `/predict` | Accepts a URL and returns verdict, probabilities and tips |
| POST | `/predict/features` | Accepts the 33 features and returns a verdict; strict data contract applies |
| GET | `/docs` | OpenAPI interactive documentation |

### API examples

**Predict from a URL**

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":"github.com/torvalds/linux"}'
```

**Invalid URL example — expected HTTP 422**

```bash
curl -i -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":""}'
```

## 8. Tests and continuous integration

### Test coverage

`pytest -v --tb=short` runs 30+ checks.

| Test area | What is checked |
|---|---|
| Health endpoint | Health response and model information |
| UI | Main web UI route |
| Response schema | Prediction response structure |
| Domain-logic test | A phishing-style URL with an IP host, login words and `http` must score a higher `P(malicious)` than `wikipedia.org/wiki/Machine_learning` |
| URL boundaries | Empty, too short, too long and whitespace-containing URLs return HTTP 422 |
| Type validation | Wrong types are rejected |
| Feature validation | Unknown fields, out-of-range features, NaN and null return HTTP 422 |
| Prediction logging | Successful predictions are logged |
| Feature extraction | Unit tests for URL feature extraction |

### GitHub Actions CI pipeline

Workflow file: `.github/workflows/ci.yml`

| Trigger | Pipeline steps |
|---|---|
| Every push to `main` | Install dependencies → train baseline (random forest on an 80k sample) → run pytest → build Docker image → container smoke test |

## 9. Deployment and monitoring

### Deployment

| Item | Detail |
|---|---|
| Suggested platforms | Render, Railway or an EC2 t3.micro |
| Container restart policy | `--restart unless-stopped` |
| Prediction log path | `logs/predictions.jsonl` |
| Docker log persistence | Mount `/app/logs` as a volume |

### Prediction log contents

Each successful prediction is appended to `logs/predictions.jsonl`.

| Logged field | Description |
|---|---|
| `timestamp` | Time of prediction |
| `URL` | Submitted URL |
| Features | 33 extracted URL features |
| Predicted class | Model's predicted class |
| `P(malicious)` | Predicted probability of a URL being malicious |

### Drift monitoring

| Item | Detail |
|---|---|
| Script | `python check_drift.py` |
| Comparison | Mean and standard deviation of logged features against training statistics saved in the model bundle |
| Drift signal | Exits with code 1 if drift is flagged |
| Suggested response | Retrain when drift is flagged, or when labelled feedback shows macro-F1 has dropped |

## 10. Limitations

| Limitation | Evidence / impact | Implication |
|---|---|---|
| Dataset shortcut: URL schemes | 91.7% of benign URLs have no `http(s)://` prefix; 100% of defacement URLs start with `http://`; only 0.5% of benign URLs use `https://` | The model can learn the presence of a scheme as a shortcut. For example, `https://www.google.com/` can score as malicious while `github.com/torvalds/linux` scores benign. The API adds a notice to requests that include a scheme. |
| Scheme-stripping experiment | In a quick random-forest experiment on a 150k sample, scheme-stripped URLs fixed google.com/wikipedia.org/github.com but reduced accuracy from 93.8% to 87.8% and exposed other shortcuts such as a leading `www.` | Scheme stripping was not a clean fix. Better data, including benign URLs collected with schemes (e.g. Tranco/Common Crawl), is needed. |
| Lexical features only | No WHOIS age, page content or reputation lists are used | Look-alike domains and short random-looking benign URLs are difficult to classify. |
| Phishing performance | Final model phishing F1-score is 0.6581 | Phishing is the weakest class; do not use the service as the only safeguard. |
| Sensitive prediction logs | Logged URLs may contain sensitive information | Protect or rotate the log file in a real deployment. |

## 11. Project members

| # | Name |
|---:|---|
| 1 | Abhinav Sharma |
| 2 | Daksh Yadav |
| 3 | Sagar Sharma |
