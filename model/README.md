`model.joblib` is generated, not hand-written. Create it with either:

    python train.py --algorithm xgboost --n_estimators 400 --max_depth 8 --learning_rate 0.1
    # or wrap the model you trained on Kaggle:
    python scripts/build_bundle_from_kaggle.py --src <folder with the 4 Kaggle .joblib files> --macro_f1 <your value>

then commit it (about 10-15 MB) so the Docker image and the deployment can use it.
