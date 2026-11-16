# Malicious URL Detector

**CSL236 End-Term Project** — an end-to-end machine-learning application that classifies URLs into benign, defacement, malware, and phishing categories. The project includes dataset preparation, lexical feature engineering, model comparison, MLflow experiment tracking, a validated FastAPI service, a web interface, automated tests, Docker packaging, GitHub Actions CI, cloud deployment, prediction logging, and drift monitoring.

<p align="center">
  <a href="https://malicious-url-detector-lmac.onrender.com/"><strong>Live Application</strong></a> ·
  <a href="https://malicious-url-detector-lmac.onrender.com/docs"><strong>Interactive API Docs</strong></a> ·
  <a href="https://malicious-url-detector-lmac.onrender.com/health"><strong>Health Check</strong></a>
</p>

> **Important:** This is an educational URL-string classification project. It does not visit or inspect the destination website, and its predictions are not a guarantee that a URL is safe or malicious. Do not use it as the only security control.

---

## 1. Project overview

| Item | Description |
|---|---|
| Project | Malicious URL Detector |
| Course | CSL236 End-Term Project |
| Problem type | Multi-class URL classification |
| Input | URL text |
| Engineered features | 33 lexical URL features |
| Output classes | `benign`, `defacement`, `malware`, `phishing` |
| Model serving | FastAPI |
| User interface | HTML/CSS/JavaScript web interface |
| Experiment tracking | MLflow |
| Testing | pytest |
| Packaging | Docker (`python:3.11-slim`) |
| CI | GitHub Actions |
| Deployment | Render |
| Monitoring | JSONL prediction logging and feature-drift checks |

### Live deployment

| Resource | URL |
|---|---|
| Web application | https://malicious-url-detector-lmac.onrender.com/ |
| API health endpoint | https://malicious-url-detector-lmac.onrender.com/health |
| Swagger / OpenAPI documentation | https://malicious-url-detector-lmac.onrender.com/docs |
| OpenAPI JSON schema | https://malicious-url-detector-lmac.onrender.com/openapi.json |

The deployed `/health` endpoint has returned `status: ok` and reported the loaded model, four classes, and 33 input features. The Swagger documentation page is also accessible on the live deployment.

## 2. Problem statement and objectives

Malicious URLs are commonly used in phishing, malware distribution, and website defacement campaigns. This project uses machine learning to identify URL patterns that may indicate these categories.

The main objectives are:

1. Prepare and inspect a labelled URL dataset.
2. Convert raw URL strings into numeric lexical features.
3. Compare multiple machine-learning algorithms using consistent evaluation metrics.
4. Track model experiments and parameters with MLflow.
5. Serialize and load the selected model for inference.
6. Expose predictions through a REST API with strict request validation.
7. Provide a simple web interface and practical safety tips.
8. Package the service in Docker and automate checks with GitHub Actions.
9. Deploy the application and record prediction information for monitoring.
10. Monitor changes in incoming feature distributions.

## 3. Dataset

| Item | Value |
|---|---|
| Dataset source | [Kaggle — Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) |
| Dataset publisher | `sid321axn` |
| Original rows | 651,191 |
| Rows after URL de-duplication | 641,119 |
| Input column | `url` |
| Target column | `type` |
| Classes | benign, defacement, phishing, malware |
| Train/test split | 70% / 30%, stratified |
| Random state | `42` |
| Scaling | Scaler fitted on the training split only |

### Class distribution after de-duplication

| Class | Rows |
|---|---:|
| benign | 428,080 |
| defacement | 95,308 |
| phishing | 94,086 |
| malware | 23,645 |
| **Total** | **641,119** |

The dataset is imbalanced. For this reason, the project reports macro-F1 and per-class precision, recall, and F1-score in addition to accuracy.

## 4. Feature engineering and data contract

The application extracts 33 lexical features from the URL string. It does not download the target page or make network requests to the submitted URL during feature extraction or prediction.

| # | Feature | Type | Purpose |
|---:|---|---|---|
| 1 | `url_length` | Integer | Total URL length |
| 2 | `hostname_length` | Integer | Hostname length |
| 3 | `path_length` | Integer | Path length |
| 4 | `count_dot` | Integer | Number of dots |
| 5 | `count_hyphen` | Integer | Number of hyphens |
| 6 | `count_underscore` | Integer | Number of underscores |
| 7 | `count_slash` | Integer | Number of slashes |
| 8 | `count_question` | Integer | Number of question marks |
| 9 | `count_equals` | Integer | Number of equals signs |
| 10 | `count_at` | Integer | Number of `@` symbols |
| 11 | `count_and` | Integer | Number of ampersands |
| 12 | `count_percent` | Integer | Number of percent signs |
| 13 | `count_hash` | Integer | Number of hash symbols |
| 14 | `count_excl` | Integer | Number of exclamation marks |
| 15 | `count_tilde` | Integer | Number of tildes |
| 16 | `count_comma` | Integer | Number of commas |
| 17 | `count_plus` | Integer | Number of plus signs |
| 18 | `count_star` | Integer | Number of asterisks |
| 19 | `count_colon` | Integer | Number of colons |
| 20 | `count_digits` | Integer | Number of numeric characters |
| 21 | `count_letters` | Integer | Number of alphabetic characters |
| 22 | `digit_ratio` | Float | Proportion of digits |
| 23 | `letter_ratio` | Float | Proportion of letters |
| 24 | `has_ip` | Integer (0/1) | Whether the host is an IP address |
| 25 | `num_subdomains` | Integer | Number of subdomains |
| 26 | `uses_https_token` | Integer (0/1) | HTTPS token indicator |
| 27 | `has_at_symbol` | Integer (0/1) | Whether the URL contains `@` |
| 28 | `has_double_slash_redirect` | Integer (0/1) | Double-slash redirect indicator |
| 29 | `is_shortened` | Integer (0/1) | Known URL-shortener indicator |
| 30 | `num_suspicious_words` | Integer | Count of configured suspicious terms |
| 31 | `entropy` | Float | Shannon entropy of URL characters |
| 32 | `tld_length` | Integer | Top-level-domain length |
| 33 | `abnormal_url` | Integer (0/1) | Abnormal URL structure indicator |

The `type` column is the prediction target; it is not included as an input feature.

### API validation contract

`POST /predict` accepts a URL string between 4 and 2,048 characters. Spaces and control characters are rejected. `POST /predict/features` accepts all 33 engineered features. The Pydantic schema is generated from the shared feature specifications in `features.py`.

The API rejects invalid requests with HTTP `422 Unprocessable Entity`, including missing required fields, unexpected extra fields, incorrect data types, out-of-range values, `NaN` or infinite floating-point values, and inconsistent feature combinations. This ensures model inference receives data that follows the training feature contract.

## 5. Machine-learning workflow

1. Load and inspect the labelled dataset.
2. Remove duplicate URLs.
3. Extract the 33 lexical features.
4. Split the dataset into stratified training and test sets.
5. Fit the scaler using training data only.
6. Train and compare multiple algorithms.
7. Evaluate using accuracy, precision, recall, macro-F1, and weighted-F1.
8. Track experiment parameters and metrics in MLflow.
9. Serialize the selected model, scaler, label encoder, and feature-column metadata.
10. Load the saved model bundle in the FastAPI application.

### Algorithms evaluated

- Logistic Regression
- Random Forest
- XGBoost

### MLflow experiment tracking

Training runs are recorded in MLflow. Tracked information includes the algorithm, training parameters, and evaluation metrics. The project includes multiple runs for model comparison.

Start the local MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000`.

### Reproducible training examples

Run from the project root after installing development requirements:

```bash
python train.py --algorithm logistic_regression --sample 80000 --run_name Logistic_Regression_Experiment
python train.py --algorithm random_forest --sample 80000 --run_name Random_Forest_Experiment
python train.py --algorithm xgboost --sample 80000 --run_name XGBoost_Experiment
python train.py --algorithm xgboost --force --run_name Final_Model_Full_Dataset
```

The final command uses the full de-duplicated dataset according to the training script's configured split and saves the selected model bundle.

## 6. Model evaluation and results

The final model reported by the project is XGBoost trained on the de-duplicated dataset.

| Metric | Result |
|---|---:|
| Accuracy | 0.8801 (88.01%) |
| Macro-F1 | 0.8452 |
| Weighted-F1 | 0.8759 |

### Per-class performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| benign | 0.8940 | 0.9490 | 0.9207 | 128,424 |
| defacement | 0.8970 | 0.8599 | 0.8781 | 28,592 |
| malware | 0.9832 | 0.8719 | 0.9242 | 7,094 |
| phishing | 0.7454 | 0.5890 | 0.6581 | 28,226 |
| **Accuracy** | — | — | **0.8801** | **192,336** |
| **Macro average** | **0.8799** | **0.8175** | **0.8452** | **192,336** |
| **Weighted average** | **0.8759** | **0.8801** | **0.8759** | **192,336** |

The model achieves 88.01% accuracy and macro-F1 of 0.8452 on the reported test set. Phishing has the lowest F1-score (0.6581), indicating that this class is more difficult for the current feature set and model to classify reliably. Accuracy alone does not describe performance across imbalanced classes; macro-F1 and per-class results are also important.

## 7. Application architecture

```text
User
 |
 v
Web interface (GET /)
 |
 v
FastAPI application
 |
 +--> Pydantic request validation
 |
 +--> URL feature extraction (33 features)
 |
 +--> Saved model bundle (scaler, classifier, label encoder, feature metadata)
 |
 +--> Prediction response (class, probabilities, risk level, safety tips)
 |
 +--> Prediction log (JSONL)
          |
          v
      Drift-check script
```

The model bundle is loaded once when the application starts rather than being loaded for every prediction request.

## 8. API documentation and endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web interface |
| GET | `/health` | Application and model health information |
| POST | `/predict` | Predict a class from a URL |
| POST | `/predict/features` | Predict from the 33 engineered features |
| GET | `/docs` | Interactive Swagger / OpenAPI documentation |
| GET | `/openapi.json` | OpenAPI schema |

### Example prediction request

```json
{
  "url": "github.com/torvalds/linux"
}
```

### Call the live API

```bash
curl -X POST "https://malicious-url-detector-lmac.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"url":"github.com/torvalds/linux"}'
```

### Example response structure

The following is an illustrative response structure, not a guaranteed prediction for the example URL:

```json
{
  "predicted_class": "benign",
  "is_malicious": false,
  "malicious_probability": 0.15,
  "risk_level": "low",
  "probabilities": {
    "benign": 0.85,
    "defacement": 0.02,
    "malware": 0.05,
    "phishing": 0.08
  },
  "tips": []
}
```

Requests that violate the API contract are rejected with HTTP `422 Unprocessable Entity`. For example, an empty URL is invalid:

```json
{
  "url": ""
}
```

## 9. Web interface

The home page allows a user to enter or paste a URL, submit it for classification, view the predicted category, review the malicious probability and class probabilities, and see a risk level with safety tips. It is an accessible demonstration of the API, not a browser reputation service or a guarantee of website safety.

## 10. Installation and local execution

### Requirements

- Python 3.11
- Git
- pip
- Docker (for container execution)
- Project files and trained model bundle

### Clone the repository

```bash
git clone https://github.com/AbhinavSharma651/malicious-url-detector.git
cd malicious-url-detector
```

### Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you only need to run the serving application and the repository provides a separate serving dependency file, use:

```bash
pip install -r requirements-serve.txt
```

### Start FastAPI

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open the local web UI at `http://127.0.0.1:8000/`, health endpoint at `http://127.0.0.1:8000/health`, and API documentation at `http://127.0.0.1:8000/docs`.

## 11. Testing

Run the test suite from the repository root:

```bash
pytest -v --tb=short
```

The test suite covers the health endpoint, web UI, prediction response schema, URL boundary validation, feature validation, feature extraction, prediction logging, and comparative classification behaviour for selected examples. The project guideline requires at least four pytest tests; run the command above and record the actual test count and pass/fail result in the final submission evidence.

## 12. Docker containerization

The application is packaged with a Python 3.11 slim base image. The serving image installs the dependencies needed for inference rather than the full model-training and testing toolchain.

### Build

```bash
docker build -t malicious-url-detector .
```

### Run

```bash
docker run -d \
  --name malicious-url-detector \
  -p 8000:8000 \
  -e PORT=8000 \
  -v mud-logs:/app/logs \
  malicious-url-detector
```

### Verify

```bash
docker ps
curl http://localhost:8000/health
```

Test a prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url":"github.com/torvalds/linux"}'
```

Check the built image size:

```bash
docker images malicious-url-detector
```

Record the actual image size displayed by Docker in the final report; it can change with dependency and build updates.

## 13. Continuous integration with GitHub Actions

Workflow file: `.github/workflows/ci.yml`.

The workflow runs on pushes to `main` and includes Python environment setup, dependency installation, baseline model training, pytest execution, Docker image build, and a container smoke test.

The latest workflow run shown during project verification completed successfully, including Docker image build and container smoke-test steps. The smoke-test log showed a successful health response and successful prediction response; an invalid prediction request was also rejected with HTTP 422 as expected.

**Evidence to retain:** capture the green GitHub Actions workflow summary and expanded test, Docker build, and container smoke-test steps.

## 14. Cloud deployment

The FastAPI application is deployed on Render.

| Deployment item | Value |
|---|---|
| Hosting platform | Render |
| Public application URL | https://malicious-url-detector-lmac.onrender.com/ |
| Health endpoint | https://malicious-url-detector-lmac.onrender.com/health |
| API documentation | https://malicious-url-detector-lmac.onrender.com/docs |
| Configured application port | `8000` |
| Framework | FastAPI |
| Container build | Docker |

### Deployment verification

The following checks have been observed during deployment verification:

- `/health` returned `status: ok`.
- `/docs` loaded and displayed the API routes and request schemas.
- The Docker smoke test started the container and received a successful health response.
- The smoke test received HTTP 200 for a valid `/predict` request.
- The smoke test received HTTP 422 for an invalid request.

Free hosting services may temporarily sleep when idle, so the first request after inactivity can take longer to respond.

## 15. Prediction logging and drift monitoring

### Prediction logging

The API appends prediction records to `logs/predictions.jsonl`. Each record includes a UTC timestamp, endpoint, submitted URL where applicable, extracted feature values, predicted class, and malicious probability. Logging is designed not to interrupt a prediction response if writing to the log fails.

**Privacy note:** URLs can contain personal or sensitive query parameters. Protect access to prediction logs, avoid sharing real user URLs in public screenshots, and apply suitable retention limits before using the application with real users.

### Drift monitoring

The project includes `check_drift.py`, which compares the distribution of logged input features with training statistics saved in the model bundle.

```bash
python check_drift.py
```

A drift warning indicates that incoming URL feature distributions may differ from the training data. Drift is a signal to investigate; it does not by itself prove that model accuracy has decreased. Where labelled feedback is available, evaluate the model on recent labelled data before deciding whether retraining is necessary.

## 16. Limitations and future improvements

### Current limitations

1. **Lexical-only analysis:** The model uses URL-string features and does not inspect page content, DNS records, WHOIS information, or live threat-intelligence feeds.
2. **Dataset imbalance:** The four classes do not have equal representation.
3. **Phishing performance:** The reported phishing F1-score is 0.6581, lower than the other class F1-scores.
4. **Dataset bias:** Patterns in the training dataset may not represent all current URLs or future attacker behaviour.
5. **No safety guarantee:** A benign prediction does not prove a site is safe, and a malicious prediction may be a false positive.
6. **Concept drift:** URL patterns can change over time, so performance may degrade after deployment.
7. **No page fetching:** A compromised legitimate website may have a benign-looking URL and cannot reliably be detected using URL text alone.

### Future improvements

- Add more recent and diverse labelled URL samples.
- Improve phishing recall while controlling false positives.
- Evaluate class-weighting and resampling approaches.
- Calibrate probability estimates and review risk thresholds.
- Add explainability for influential features.
- Add scheduled monitoring and model-performance evaluation on labelled feedback.
- Improve automated deployment and model-version tracking.
- Add stronger privacy controls and retention policies for prediction logs.

## 17. Repository structure

| Path | Purpose |
|---|---|
| `app.py` | FastAPI application, request schemas, predictions, and logging |
| `features.py` | Feature extraction, feature specifications, and safety tips |
| `train.py` | Model training and MLflow experiment tracking |
| `check_drift.py` | Feature-drift check |
| `model/model.joblib` | Serialized model bundle |
| `model/best_metrics.json` | Best model metrics |
| `data/malicious_phish.csv` | Dataset (if included in the checkout) |
| `data/README.md` | Dataset notes |
| `static/index.html` | Web application interface |
| `tests/test_app.py` | API and application tests |
| `scripts/` | Model/data helper scripts |
| `Dockerfile` | Docker image definition |
| `requirements.txt` | Project and training dependencies |
| `requirements-serve.txt` | Serving dependencies |
| `.github/workflows/ci.yml` | GitHub Actions CI workflow |

## 18. Team details and contribution



| # | Team member | Roll number | Contribution |
|---:|---|---|---|
| 1 | Abhinav Sharma | 24CSU004 | Project integration; machine-learning model training and evaluation; FastAPI implementation and Pydantic validation; Docker packaging, GitHub Actions CI, and Render deployment. |
| 2 | Daksh Yadav | 24CSU050  | Dataset sourcing, cleaning and exploratory analysis; class-distribution review; support for URL lexical feature engineering and preparation of the model input data. |
| 3 | Sagar Sharma | 24CSU177 | Web-interface development; API and input-validation testing; documentation and presentation support; prediction-logging and drift-monitoring review. |

## 19. References

1. [Malicious URLs Dataset — Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
2. [FastAPI documentation](https://fastapi.tiangolo.com/)
3. [Pydantic documentation](https://docs.pydantic.dev/)
4. [MLflow documentation](https://mlflow.org/docs/latest/index.html)
5. [scikit-learn documentation](https://scikit-learn.org/stable/)
6. [XGBoost documentation](https://xgboost.readthedocs.io/)
7. [Docker documentation](https://docs.docker.com/)
8. [GitHub Actions documentation](https://docs.github.com/actions)
9. [Render documentation](https://render.com/docs)


