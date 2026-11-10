"""pytest suite. Run `python train.py ...` first so model/model.joblib exists (CI does this)."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module  # noqa: E402
from features import FEATURE_COLUMNS, extract_features, generate_tips  # noqa: E402

client = TestClient(app_module.app)

CLEAN_URL = "wikipedia.org/wiki/Machine_learning"
PHISHY_URL = "http://192.168.0.1/paypal-secure-login/verify/account.php?id=3&update=1"


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """Never write test traffic into the real prediction log."""
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))
    return tmp_path / "predictions.jsonl"


def valid_features():
    return extract_features("github.com/torvalds/linux")


# ---- 1. health --------------------------------------------------------------
def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["classes"]) == {"benign", "defacement", "malware", "phishing"}
    assert body["n_features"] == 33


# ---- 2. UI loads ------------------------------------------------------------
def test_ui_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Malicious URL Detector" in r.text


# ---- 3. response shape ------------------------------------------------------
def test_predict_returns_probabilities_and_tips():
    r = client.post("/predict", json={"url": CLEAN_URL})
    assert r.status_code == 200
    body = r.json()
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-2
    assert 0.0 <= body["malicious_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}
    assert 2 <= len(body["tips"]) <= 3
    assert len(body["features"]) == 33


# ---- 4. domain logic: a suspicious URL must score riskier than a clean one ---
def test_suspicious_url_scores_higher_than_clean_url():
    clean = client.post("/predict", json={"url": CLEAN_URL}).json()
    phishy = client.post("/predict", json={"url": PHISHY_URL}).json()
    assert phishy["malicious_probability"] > clean["malicious_probability"]
    assert phishy["is_malicious"] is True


# ---- 5. boundary tests: every bad input must be a 422, never a 500 ----------
@pytest.mark.parametrize(
    "payload",
    [
        {},                                  # missing field
        {"url": ""},                         # empty
        {"url": "ab"},                       # shorter than 4 chars
        {"url": "a" * 2049},                 # longer than 2048 chars
        {"url": "http://exa mple.com/x"},    # whitespace inside
        {"url": "http://example.com/\x00"},  # control character
        {"url": 12345},                      # wrong type
        {"url": None},                       # null
        {"url": "example.com", "extra": 1},  # unknown field
    ],
)
def test_invalid_url_payload_returns_422(payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_url_at_exact_max_length_is_accepted():
    url = "example.com/" + "a" * (2048 - len("example.com/"))
    assert len(url) == 2048
    assert client.post("/predict", json={"url": url}).status_code == 200


# ---- 6. feature-level contract (the data-contract table) --------------------
def test_features_endpoint_accepts_valid_vector():
    r = client.post("/predict/features", json=valid_features())
    assert r.status_code == 200
    assert r.json()["predicted_class"] in {"benign", "defacement", "malware", "phishing"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("url_length", 0),            # below minimum
        ("url_length", 5000),         # above maximum
        ("digit_ratio", 1.5),         # ratio above 1
        ("digit_ratio", -0.1),        # ratio below 0
        ("has_ip", 2),                # flag must be 0/1
        ("num_suspicious_words", 99),
        ("entropy", 12.0),
        ("url_length", "60"),         # wrong type (string)
        ("url_length", 60.5),         # wrong type (float for int)
        ("entropy", None),            # null / missing value
    ],
)
def test_features_out_of_contract_returns_422(field, value):
    payload = valid_features()
    payload[field] = value
    assert client.post("/predict/features", json=payload).status_code == 422


def test_features_nan_returns_422():
    payload = valid_features()
    raw = json.dumps(payload).replace(f'"entropy": {payload["entropy"]}', '"entropy": NaN')
    assert "NaN" in raw
    r = client.post("/predict/features", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_features_missing_field_returns_422():
    payload = valid_features()
    del payload["has_ip"]
    assert client.post("/predict/features", json=payload).status_code == 422


def test_inconsistent_features_return_422():
    payload = valid_features()
    payload["url_length"] = 10
    payload["count_digits"] = 9
    payload["count_letters"] = 9  # 18 characters cannot fit in a 10-character URL
    assert client.post("/predict/features", json=payload).status_code == 422


# ---- 7. logging (used by check_drift.py) ------------------------------------
def test_every_prediction_is_logged(isolated_log):
    client.post("/predict", json={"url": CLEAN_URL})
    client.post("/predict/features", json=valid_features())
    lines = isolated_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["url"] == CLEAN_URL and set(first["features"]) == set(FEATURE_COLUMNS)


def test_rejected_requests_are_not_logged(isolated_log):
    client.post("/predict", json={"url": ""})
    assert not isolated_log.exists()


# ---- 8. feature extraction unit tests ---------------------------------------
def test_feature_extraction_flags_ip_hosts_and_shorteners():
    assert extract_features("http://10.0.0.5/index.html")["has_ip"] == 1
    assert extract_features("example.com/index.html")["has_ip"] == 0
    assert extract_features("http://bit.ly/abc123")["is_shortened"] == 1
    assert extract_features("alex.com/page")["is_shortened"] == 0  # 'x.co' inside a word is not a shortener


def test_feature_extraction_handles_malformed_urls():
    feats = extract_features("http://[bad-ipv6/path")  # urlparse raises ValueError on this
    assert len(feats) == 33


def test_tips_are_two_or_three_and_relevant():
    feats = extract_features(PHISHY_URL)
    tips = generate_tips(PHISHY_URL, feats, 0.9)
    assert 2 <= len(tips) <= 3
    assert any("IP" in t for t in tips)
