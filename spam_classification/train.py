from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import load_dataset, split_dataset
from .preprocess import build_vectorizer


def build_pipeline(
    calibrated: bool = False,
    random_state: int = 42,
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Pipeline:
    vectorizer: TfidfVectorizer = build_vectorizer(max_features=max_features, ngram_range=ngram_range)
    if calibrated:
        base = LinearSVC(random_state=random_state)
        # scikit-learn >=1.4 uses `estimator` instead of deprecated `base_estimator`
        clf = CalibratedClassifierCV(estimator=base, cv=5, method="sigmoid")
    else:
        clf = LinearSVC(random_state=random_state)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train(
    csv_path: str = "sms_spam_no_header.csv",
    out_dir: str = "artifacts",
    test_size: float = 0.2,
    random_state: int = 42,
    calibrated: bool = False,
    max_features: int = 20000,
) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=test_size, random_state=random_state)
    pipeline = build_pipeline(calibrated=calibrated, random_state=random_state, max_features=max_features)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "calibrated": calibrated,
        "random_state": random_state,
        "test_size": test_size,
        "max_features": max_features,
    }

    # persist artifacts
    dump(pipeline, Path(out_dir) / "model.joblib")
    # also persist vectorizer separately for tooling/demo purposes
    dump(pipeline.named_steps["tfidf"], Path(out_dir) / "vectorizer.joblib")

    with open(Path(out_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Train spam classification baseline")
    parser.add_argument("--data", default="sms_spam_no_header.csv", help="CSV path")
    parser.add_argument("--out", default="artifacts", help="artifacts output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--calibrated", action="store_true", help="use probability calibration")
    parser.add_argument("--max-features", type=int, default=20000, help="TF-IDF max features")
    args = parser.parse_args()

    metrics = train(
        csv_path=args.data,
        out_dir=args.out,
        test_size=args.test_size,
        random_state=args.seed,
        calibrated=args.calibrated,
        max_features=args.max_features,
    )
    pprint.pp(metrics)
