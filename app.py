"""FastAPI service for the malicious-URL detector.

Routes
    GET  /                  web UI (static/index.html)
    GET  /health            liveness + model info
    POST /predict           {"url": "..."}  -> verdict, probabilities, tips
    POST /predict/features  the 33 engineered features -> verdict (strict contract validation)
    GET  /docs              interactive OpenAPI page

Every constraint of the README data contract is enforced by the Pydantic
schemas below; anything out of range or of the wrong type returns HTTP 422.
"""
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from features import FEATURE_COLUMNS, FEATURE_SPECS, MAX_URL_LEN, extract_features, generate_tips, to_vector

logger = logging.getLogger("malicious-url-detector")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.joblib"))

# ---- Model is loaded ONCE at import/startup, never inside a request handler ----
BUNDLE = joblib.load(MODEL_PATH)
MODEL = BUNDLE["model"]
SCALER = BUNDLE["scaler"]
CLASSES = [str(c) for c in BUNDLE["label_encoder"].classes_]
BENIGN_IDX = CLASSES.index("benign")
ALGORITHM = BUNDLE.get("algorithm", type(MODEL).__name__)
assert list(BUNDLE["feature_columns"]) == FEATURE_COLUMNS, "model features differ from features.py"

HIGH_RISK, MEDIUM_RISK = 0.60, 0.30
SCHEME_NOTICE = (
    "The training data lists most safe sites without 'http://' or 'https://', so links that include a "
    "scheme lean towards 'malicious'. Treat the score as a warning signal, not a final verdict."
)


# ----------------------------------------------------------------------------
# Request schemas (the data contract)
# ----------------------------------------------------------------------------
class URLRequest(BaseModel):
    """A single URL as typed or pasted by the user."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    url: str = Field(min_length=4, max_length=MAX_URL_LEN, description="URL to check, 4-2048 characters, no spaces",
                     examples=["github.com/torvalds/linux"])

    @field_validator("url")
    @classmethod
    def no_whitespace_or_control_chars(cls, v: str) -> str:
        if re.search(r"\s", v) or any(ord(ch) < 32 or ord(ch) == 127 for ch in v):
            raise ValueError("URL must not contain spaces or control characters")
        return v


def _feature_field(kind: str, low, high):
    if kind == "int":
        return (int, Field(strict=True, ge=low, le=high))
    return (float, Field(strict=True, ge=low, le=high, allow_inf_nan=False))


class _FeaturesBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def features_must_be_consistent(self):
        d = self.model_dump()
        if d["count_digits"] + d["count_letters"] > d["url_length"]:
            raise ValueError("count_digits + count_letters cannot exceed url_length")
        if d["hostname_length"] > d["url_length"] or d["path_length"] > d["url_length"]:
            raise ValueError("hostname_length and path_length cannot exceed url_length")
        if d["digit_ratio"] + d["letter_ratio"] > 1.0 + 1e-9:
            raise ValueError("digit_ratio + letter_ratio cannot exceed 1")
        return self


# One required field per row of the contract table; built from features.FEATURE_SPECS.
FeaturesRequest = create_model(
    "FeaturesRequest",
    __base__=_FeaturesBase,
    **{name: _feature_field(kind, lo, hi) for name, kind, lo, hi, _ in FEATURE_SPECS},
)


class Prediction(BaseModel):
    predicted_class: str = Field(description="benign, or the most likely threat type")
    is_malicious: bool
    malicious_probability: float = Field(description="1 - P(benign)")
    risk_level: str = Field(description="low / medium / high")
    probabilities: dict[str, float]
    tips: list[str] = []
    notice: Optional[str] = None
    features: Optional[dict[str, float]] = None


# ----------------------------------------------------------------------------
# Prediction + logging
# ----------------------------------------------------------------------------
def _score(feats: dict) -> dict:
    x = SCALER.transform(np.asarray([to_vector(feats)], dtype=float))
    proba = MODEL.predict_proba(x)[0]
    p_mal = float(1.0 - proba[BENIGN_IDX])
    is_mal = p_mal >= 0.5
    if is_mal:  # name the most likely threat type, not "benign"
        threats = [(proba[i], CLASSES[i]) for i in range(len(CLASSES)) if i != BENIGN_IDX]
        label = max(threats)[1]
    else:
        label = "benign"
    risk = "high" if p_mal >= HIGH_RISK else "medium" if p_mal >= MEDIUM_RISK else "low"
    return {
        "predicted_class": label,
        "is_malicious": is_mal,
        "malicious_probability": round(p_mal, 4),
        "risk_level": risk,
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
    }


_log_lock = threading.Lock()


def _log_prediction(endpoint: str, url: Optional[str], feats: dict, result: dict) -> None:
    """Append one JSON line per prediction; used by check_drift.py. Never breaks a request."""
    try:
        path = Path(os.getenv("PREDICTION_LOG", BASE_DIR / "logs" / "predictions.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "endpoint": endpoint,
            "url": url,
            "features": feats,
            "predicted_class": result["predicted_class"],
            "malicious_probability": result["malicious_probability"],
        }
        with _log_lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("could not write prediction log")


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Malicious URL Detector",
    version="1.0.0",
    description="Classifies a URL as benign, defacement, malware or phishing from 33 lexical features.",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "algorithm": ALGORITHM, "classes": CLASSES, "n_features": len(FEATURE_COLUMNS)}


@app.post("/predict", response_model=Prediction)
def predict(req: URLRequest):
    feats = extract_features(req.url)
    result = _score(feats)
    result["tips"] = generate_tips(req.url, feats, result["malicious_probability"])
    if re.match(r"(?i)^[a-z][a-z0-9+.-]*://", req.url):
        result["notice"] = SCHEME_NOTICE
    result["features"] = {k: float(v) for k, v in feats.items()}
    _log_prediction("/predict", req.url, feats, result)
    return result


@app.post("/predict/features", response_model=Prediction)
def predict_from_features(req: FeaturesRequest):  # type: ignore[valid-type]
    feats = req.model_dump()
    result = _score(feats)
    _log_prediction("/predict/features", None, feats, result)
    return result
