from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset without header and normalize labels/messages.

    Expected CSV format (no header):
      0: label ("spam"|"ham")
      1: message (text)
    """
    df = pd.read_csv(csv_path, header=None, names=["label", "message"])
    # basic cleanup
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    # keep only expected labels
    df = df[df["label"].isin(["spam", "ham"])]
    df = df.dropna(subset=["label", "message"])  # ensure no missing
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split into train/test, prefer stratified; fallback if insufficient per class."""
    X = df["message"]
    y = df["label"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # not enough samples to stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load and summarize dataset")
    parser.add_argument("--data", default="sms_spam_no_header.csv", help="CSV path")
    parser.add_argument("--test-size", type=float, default=0.2, help="test size fraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = split_dataset(df, args.test_size, args.seed)
    summary = {
        "total": int(len(df)),
        "train": int(len(X_train)),
        "test": int(len(X_test)),
        "labels": {lbl: int((df["label"] == lbl).sum()) for lbl in ["spam", "ham"]},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

