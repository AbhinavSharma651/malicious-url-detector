# Dataset

`malicious_phish.csv` – 651,191 rows, two columns: `url` and `type`
(`benign`, `defacement`, `phishing`, `malware`).

Source: Kaggle, "Malicious URLs dataset" (sid321axn), https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
The file is about 46 MB, under the course's 50 MB limit, so it is committed directly.
If you prefer not to commit it, delete it, add `data/malicious_phish.csv` to `.gitignore`, and
download it with `kaggle datasets download -d sid321axn/malicious-urls-dataset -p data --unzip`.

Preprocessing done in `train.py` (same as the Kaggle run): drop rows with missing values, drop duplicate URLs
(651,191 -> 641,119 rows), build 33 lexical features (`features.py`), 70/30 stratified split.
