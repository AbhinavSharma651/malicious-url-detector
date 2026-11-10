"""URL -> 33 lexical features, plus the data contract and rule-based tips.

This module is shared by train.py (offline) and app.py (online) so the model
sees exactly the same feature definitions in both places. The definitions were
reverse-checked against the StandardScaler saved from the Kaggle run: the mean
and std of every feature computed here match the scaler's mean_/scale_.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# Data contract: one entry per model feature (name, type, low, high, why).
# app.py builds the Pydantic schema for POST /predict/features from this list,
# and the README table is generated from it, so they can never drift apart.
# --------------------------------------------------------------------------
MAX_URL_LEN = 2048

FEATURE_SPECS = [
    ("url_length", "int", 1, MAX_URL_LEN, "Phishing URLs tend to be long, to hide the real domain"),
    ("hostname_length", "int", 0, MAX_URL_LEN, "Very long hostnames often imitate brands"),
    ("path_length", "int", 0, MAX_URL_LEN, "Long paths carry payload names and redirect chains"),
    ("count_dot", "int", 0, MAX_URL_LEN, "Many dots mean many (sub)domains"),
    ("count_hyphen", "int", 0, MAX_URL_LEN, "Hyphens are common in look-alike domains"),
    ("count_underscore", "int", 0, MAX_URL_LEN, "Underscores appear in generated/defaced paths"),
    ("count_slash", "int", 0, MAX_URL_LEN, "Directory depth of the URL"),
    ("count_question", "int", 0, MAX_URL_LEN, "Query string present (tracking / injected params)"),
    ("count_equals", "int", 0, MAX_URL_LEN, "Number of query parameters"),
    ("count_at", "int", 0, MAX_URL_LEN, "'@' can hide the real destination"),
    ("count_and", "int", 0, MAX_URL_LEN, "Number of chained query parameters"),
    ("count_percent", "int", 0, MAX_URL_LEN, "Percent-encoding can obfuscate content"),
    ("count_hash", "int", 0, MAX_URL_LEN, "Fragment markers"),
    ("count_excl", "int", 0, MAX_URL_LEN, "Rare '!' character"),
    ("count_tilde", "int", 0, MAX_URL_LEN, "Home-directory paths, often compromised hosts"),
    ("count_comma", "int", 0, MAX_URL_LEN, "Rare ',' character"),
    ("count_plus", "int", 0, MAX_URL_LEN, "Encoded spaces / obfuscation"),
    ("count_star", "int", 0, MAX_URL_LEN, "Rare '*' character"),
    ("count_colon", "int", 0, MAX_URL_LEN, "Scheme and port separators"),
    ("count_digits", "int", 0, MAX_URL_LEN, "Digit-heavy URLs are often auto-generated"),
    ("count_letters", "int", 0, MAX_URL_LEN, "Readable content in the URL"),
    ("digit_ratio", "float", 0.0, 1.0, "Share of digits in the URL"),
    ("letter_ratio", "float", 0.0, 1.0, "Share of letters in the URL"),
    ("has_ip", "int", 0, 1, "Raw IP address instead of a domain name"),
    ("num_subdomains", "int", 0, MAX_URL_LEN, "Deep subdomain chains imitate trusted brands"),
    ("uses_https_token", "int", 0, 1, "'https' text inside an http:// URL (fake security cue)"),
    ("has_at_symbol", "int", 0, 1, "Presence of '@' in the URL"),
    ("has_double_slash_redirect", "int", 0, 1, "'//' after the scheme signals a redirect"),
    ("is_shortened", "int", 0, 1, "URL shortener hides the destination"),
    ("num_suspicious_words", "int", 0, 15, "Words like login/verify/secure used for lures"),
    ("entropy", "float", 0.0, 8.0, "Random-looking strings have high Shannon entropy"),
    ("tld_length", "int", 0, 63, "Unusual top-level domains"),
    ("abnormal_url", "int", 0, 1, "Hostname is not present in the URL as written"),
]
FEATURE_COLUMNS = [spec[0] for spec in FEATURE_SPECS]
assert len(FEATURE_COLUMNS) == 33

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "bank", "confirm",
    "paypal", "password", "webscr", "ebay", "free", "lucky", "bonus",
]

_SHORTENERS = (
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|"
    r"cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|"
    r"qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net"
)
_SHORT_RE = re.compile(r"(?<![\w-])(?:" + _SHORTENERS + ")")
_IP_RE = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")
_COUNTED_CHARS = [
    ("dot", "."), ("hyphen", "-"), ("underscore", "_"), ("slash", "/"), ("question", "?"),
    ("equals", "="), ("at", "@"), ("and", "&"), ("percent", "%"), ("hash", "#"),
    ("excl", "!"), ("tilde", "~"), ("comma", ","), ("plus", "+"), ("star", "*"), ("colon", ":"),
]


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def extract_features(url: str) -> dict:
    """Return the 33 features for one URL as {name: number}."""
    full = url if "//" in url else "http://" + url
    try:
        parsed = urlparse(full)
        host, path, hostname = parsed.netloc or "", parsed.path or "", parsed.hostname
    except ValueError:  # e.g. malformed IPv6 brackets
        host, path, hostname = "", "", None

    n = max(len(url), 1)
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    lowered = url.lower()

    feats = {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
    }
    for name, char in _COUNTED_CHARS:
        feats["count_" + name] = url.count(char)
    feats.update(
        count_digits=digits,
        count_letters=letters,
        digit_ratio=digits / n,
        letter_ratio=letters / n,
        has_ip=int(bool(_IP_RE.search(host))),
        num_subdomains=max(host.count(".") - 1, 0),
        uses_https_token=int(lowered.startswith("http://") and "https" in lowered[7:]),
        has_at_symbol=int("@" in url),
        has_double_slash_redirect=int(url.rfind("//") > 7),
        is_shortened=int(bool(_SHORT_RE.search(url))),
        num_suspicious_words=sum(w in lowered for w in SUSPICIOUS_WORDS),
        entropy=_entropy(url),
        tld_length=len(host.split(".")[-1]) if host else 0,
        abnormal_url=int(hostname is not None and hostname not in full),
    )
    return {k: feats[k] for k in FEATURE_COLUMNS}


def to_vector(feats: dict) -> list:
    """Order a feature dict exactly like the training columns."""
    return [feats[c] for c in FEATURE_COLUMNS]


def contract_markdown() -> str:
    """Markdown data-contract table (used for the README)."""
    rows = ["| Feature | Type | Allowed range / values | Why it matters |", "|---|---|---|---|"]
    for name, typ, lo, hi, why in FEATURE_SPECS:
        rng = "{0, 1}" if (lo, hi) == (0, 1) and typ == "int" else f"{lo} – {hi}"
        rows.append(f"| `{name}` | {typ} | {rng} | {why} |")
    rows.append("| `type` (target) | category | benign, defacement, malware, phishing | Label |")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Rule-based tips shown in the UI (2-3 per prediction).
# --------------------------------------------------------------------------
def generate_tips(url: str, feats: dict, p_malicious: float) -> list[str]:
    tips: list[str] = []
    lowered = url.lower()
    if feats["has_ip"]:
        tips.append("The address uses a raw IP instead of a domain name. Legitimate sites rarely do this.")
    if feats["is_shortened"]:
        tips.append("This is a shortened link, so the real destination is hidden. Expand it with a preview tool before opening.")
    if feats["has_at_symbol"]:
        tips.append("An '@' in a URL can make the browser ignore everything before it. Check where the link really goes.")
    if feats["num_suspicious_words"] >= 1:
        tips.append("It contains lure words such as login, verify or secure. Type the official site address yourself instead of clicking.")
    if lowered.startswith("http://"):
        tips.append("The link is not encrypted (http, not https). Never enter passwords or card details on it.")
    if feats["num_subdomains"] >= 3 or feats["count_hyphen"] >= 4:
        tips.append("Many subdomains or hyphens are typical of look-alike domains. Read the main domain right before the first '/'.")
    if feats["url_length"] > 100:
        tips.append("This URL is unusually long, which is often used to bury the real domain.")
    if feats["entropy"] > 4.8 and feats["count_digits"] > 10:
        tips.append("The URL looks randomly generated (many digits, high entropy), which is common for throw-away malware hosts.")

    if p_malicious >= 0.5:
        fallback = [
            "Do not open this link or download anything from it.",
            "If it arrived by email or chat, report it and delete the message.",
            "If you already entered a password, change it and enable two-factor authentication.",
        ]
    else:
        fallback = [
            "No strong warning signs were found, but this tool only reads the URL text, not the page itself.",
            "Check the domain spelling carefully before logging in anywhere.",
            "Keep your browser and antivirus up to date.",
        ]
    for extra in fallback:
        if len(tips) >= 3:
            break
        tips.append(extra)
    return tips[:3]
