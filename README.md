# Malicious URL Detector

CSL236 End-Term Project – production-style ML system (tracked experiments, validated API, tests, Docker, CI, deployment).

**Live URL:** `TODO – add after deployment`  ·  **API docs:** `<live-url>/docs`

## 1. Problem statement

Given a URL, predict whether it is **benign**, **defacement**, **malware** or **phishing**, using only the text of the URL
(33 lexical features, no page download, no network calls). This is a 4-class classification problem; the service also reports
`P(malicious) = 1 - P(benign)` and a risk level, and the UI adds rule-based safety tips.

## 2. Dataset

* **Source:** Kaggle, *Malicious URLs dataset* (sid321axn) – https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
* 651,191 rows (`url`, `type`); 641,119 after removing duplicate URLs. Classes after de-duplication: benign 428,080 · defacement 95,308 · phishing 94,086 · malware 23,645.
* Split: 70 % train / 30 % test, stratified, `random_state=42`. The scaler is fitted on the training split only.
* The 33 features are built by `features.py` and are shared by training and serving, so the model sees identical inputs in both places.

## 3. Data contract

The user supplies a URL (`POST /predict`, 4–2048 characters, no spaces/control characters). The server derives the 33 features below.
`POST /predict/features` accepts the same 33 features directly, and **every row of this table is enforced by the Pydantic schema**
(`features.FEATURE_SPECS` → `app.FeaturesRequest`); wrong type, out-of-range, missing, null, NaN or unknown fields return **HTTP 422**.
Types are strict: `"60"` or `60.5` is not accepted for an `int` field. Cross-field rules: `count_digits + count_letters ≤ url_length`,
`hostname_length, path_length ≤ url_length`, `digit_ratio + letter_ratio ≤ 1`.

| Feature | Type | Allowed range / values | Why it matters |
|---|---|---|---|
| `url_length` | int | 1 – 2048 | Phishing URLs tend to be long, to hide the real domain |
| `hostname_length` | int | 0 – 2048 | Very long hostnames often imitate brands |
| `path_length` | int | 0 – 2048 | Long paths carry payload names and redirect chains |
| `count_dot` | int | 0 – 2048 | Many dots mean many (sub)domains |
| `count_hyphen` | int | 0 – 2048 | Hyphens are common in look-alike domains |
| `count_underscore` | int | 0 – 2048 | Underscores appear in generated/defaced paths |
| `count_slash` | int | 0 – 2048 | Directory depth of the URL |
| `count_question` | int | 0 – 2048 | Query string present (tracking / injected params) |
| `count_equals` | int | 0 – 2048 | Number of query parameters |
| `count_at` | int | 0 – 2048 | '@' can hide the real destination |
| `count_and` | int | 0 – 2048 | Number of chained query parameters |
| `count_percent` | int | 0 – 2048 | Percent-encoding can obfuscate content |
| `count_hash` | int | 0 – 2048 | Fragment markers |
| `count_excl` | int | 0 – 2048 | Rare '!' character |
| `count_tilde` | int | 0 – 2048 | Home-directory paths, often compromised hosts |
| `count_comma` | int | 0 – 2048 | Rare ',' character |
| `count_plus` | int | 0 – 2048 | Encoded spaces / obfuscation |
| `count_star` | int | 0 – 2048 | Rare '*' character |
| `count_colon` | int | 0 – 2048 | Scheme and port separators |
| `count_digits` | int | 0 – 2048 | Digit-heavy URLs are often auto-generated |
| `count_letters` | int | 0 – 2048 | Readable content in the URL |
| `digit_ratio` | float | 0.0 – 1.0 | Share of digits in the URL |
| `letter_ratio` | float | 0.0 – 1.0 | Share of letters in the URL |
| `has_ip` | int | {0, 1} | Raw IP address instead of a domain name |
| `num_subdomains` | int | 0 – 2048 | Deep subdomain chains imitate trusted brands |
| `uses_https_token` | int | {0, 1} | 'https' text inside an http:// URL (fake security cue) |
| `has_at_symbol` | int | {0, 1} | Presence of '@' in the URL |
| `has_double_slash_redirect` | int | {0, 1} | '//' after the scheme signals a redirect |
| `is_shortened` | int | {0, 1} | URL shortener hides the destination |
| `num_suspicious_words` | int | 0 – 15 | Words like login/verify/secure used for lures |
| `entropy` | float | 0.0 – 8.0 | Random-looking strings have high Shannon entropy |
| `tld_length` | int | 0 – 63 | Unusual top-level domains |
| `abnormal_url` | int | {0, 1} | Hostname is not present in the URL as written |
| `type` (target) | category | benign, defacement, malware, phishing | Label |

## 4. Repository layout

```
data/malicious_phish.csv      dataset (+ data/README.md)
model/model.joblib            best model bundle (model + scaler + label encoder + feature list + training stats)
static/index.html             web UI
tests/test_app.py             pytest suite
.github/workflows/ci.yml      GitHub Actions
app.py                        FastAPI app + Pydantic schemas + prediction logging
features.py                   URL -> 33 features, data contract, rule-based tips
train.py                      training script with MLflow tracking (CLI hyper-parameters)
check_drift.py                bonus: drift check on logged inputs
scripts/                      build_bundle_from_kaggle.py, export_mlflow_results.py
Dockerfile · requirements.txt · requirements-serve.txt · .gitignore
```

## 5. Setup and run (local)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# train (3+ configurations, 2+ algorithms) - every run is logged to MLflow (sqlite:///mlflow.db)
python train.py --algorithm logistic_regression --C 1.0
python train.py --algorithm random_forest --n_estimators 200 --max_depth 20
python train.py --algorithm xgboost --n_estimators 200 --max_depth 6 --learning_rate 0.1
python train.py --algorithm xgboost --n_estimators 400 --max_depth 8 --learning_rate 0.1   # the Kaggle configuration

mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
uvicorn app:app --reload --port 8000                  # http://127.0.0.1:8000  and  /docs
pytest -v --tb=short
```

`train.py` only overwrites `model/model.joblib` when a run beats the best macro-F1 so far (stored in `model/best_metrics.json`);
add `--force` to override. The first full run builds the features for all 641 k URLs (about a minute) and caches them in
`data/.features_cache.joblib` (git-ignored). Use `--sample 80000` for a quick run.

**Using the model you trained on Kaggle instead of re-training:**
`python scripts/build_bundle_from_kaggle.py --src <folder> --macro_f1 <value>`.
Its pickles were written with scikit-learn 1.6.1; `requirements.txt` pins that version. Check `xgboost.__version__` on Kaggle and
pin the same major/minor version of `xgboost-cpu` here if loading fails.

## 6. Docker

```bash
docker build -t malicious-url-detector .
docker run -d -p 8000:8000 --restart unless-stopped -v mud-logs:/app/logs malicious-url-detector
```

Built from `python:3.11-slim` with `requirements-serve.txt` only (no MLflow, pytest, pandas). **Image size:** `TODO – run docker images`.

## 7. Experiments (MLflow)

Generate this table from your runs with `python scripts/export_mlflow_results.py` and paste it here:

| # | Run | Algorithm | Hyperparameters | Accuracy | Macro-F1 | Weighted-F1 | Malicious recall |
|---|-----|-----------|-----------------|----------|----------|-------------|------------------|
| TODO | | | | | | | |

**Chosen model:** `TODO – name it and justify with the table (macro-F1 matters more than accuracy because phishing and malware are the minority classes).`

Metrics: *macro-F1* treats the four classes equally; *malicious recall* is the share of malicious URLs (any non-benign class) that are not called benign.

## 8. API

| Route | Purpose |
|---|---|
| `GET /` | UI: prediction, probability bars, 2–3 rule-based tips |
| `GET /health` | liveness + model info |
| `POST /predict` | `{"url": "github.com/torvalds/linux"}` → verdict, probabilities, tips |
| `POST /predict/features` | the 33 features → verdict (strict contract) |
| `GET /docs` | OpenAPI |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"url":"github.com/torvalds/linux"}'
curl -i -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"url":""}'   # -> 422
```

## 9. Tests and CI

`pytest -v --tb=short` runs 30+ checks: health, UI, response schema, the domain-logic test (a phishing-style URL with an IP host, login words and
`http` must score a higher `P(malicious)` than `wikipedia.org/wiki/Machine_learning`), boundary tests (empty / too short / too long /
whitespace / wrong type / unknown field / out-of-range features / NaN / null → 422), prediction logging, and feature-extraction unit tests.
GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main`: install → train a baseline (random forest on an 80 k sample) → pytest →
docker build → container smoke test.

## 10. Deployment and monitoring

Deploy the Docker image on Render, Railway or an EC2 t3.micro (`--restart unless-stopped`). Every successful prediction is appended to
`logs/predictions.jsonl` (timestamp, URL, 33 features, predicted class, P(malicious)); in Docker mount `/app/logs` as a volume.
**Drift check (bonus):** `python check_drift.py` compares the mean/std of the logged features with the training statistics saved in the
model bundle and exits with code 1 if drift is flagged. Retrain when drift is flagged, or when labelled feedback shows macro-F1 has dropped.

## 11. Limitations (important – be ready to explain them in the viva)

* **Dataset shortcut.** 91.7 % of benign URLs in the dataset have no `http(s)://` prefix, 100 % of defacement URLs start with `http://`, and only 0.5 % of benign
  URLs use `https://`. The model learns this, so a link such as `https://www.google.com/` can score as malicious while `github.com/torvalds/linux` scores benign.
  The API adds a notice to any request that includes a scheme. In a quick random-forest experiment on a 150 k sample, retraining on scheme-stripped URLs fixed google.com/wikipedia.org/github.com but cost about 6 points of accuracy
  (93.8 % → 87.8 %) and exposed other shortcuts (e.g. a leading `www.`), so it is not a clean fix. Better data (benign URLs collected with schemes, e.g. from Tranco/Common Crawl) is.
* Lexical features only: no WHOIS age, no page content, no reputation lists. Look-alike domains and short random-looking benign URLs are hard.
* Phishing is usually the weakest class (check the per-class report in MLflow); do not use the service as the only safeguard.
* Logged URLs may be sensitive; protect or rotate the log file in a real deployment.

## 12. Team

| Name |
|---|
| Abhinav Sharma |
