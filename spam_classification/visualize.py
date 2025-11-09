from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from .data import load_dataset, split_dataset
from .infer import load_pipeline


def _positive_scores(pipeline, X_test: List[str]) -> np.ndarray | None:
    """Return scores for positive class (spam). Prefer calibrated probabilities; fallback to decision_function.
    When falling back, map margins to [0,1] via logistic for visualization.
    """
    try:
        proba = pipeline.predict_proba(X_test)
        classes = list(pipeline.named_steps["clf"].classes_)
        pos_idx = classes.index("spam")
        return np.asarray(proba[:, pos_idx])
    except Exception:
        try:
            margins = pipeline.decision_function(X_test)
            margins = np.asarray(margins)
            # logistic to [0,1]
            return 1.0 / (1.0 + np.exp(-margins))
        except Exception:
            return None


def top_tokens_by_class(pipeline, top_n: int = 20) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return Top-N tokens for ham (most negative weights) and spam (most positive weights)."""
    try:
        tfidf = pipeline.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        clf = pipeline.named_steps["clf"]
        base = getattr(clf, "estimator", clf)
        coefs = getattr(base, "coef_", None)
        if coefs is None:
            return None, None
        weights = coefs[0] if coefs.ndim == 2 else coefs
        spam_idx = np.argsort(weights)[::-1][:top_n]
        ham_idx = np.argsort(weights)[:top_n]
        df_spam = pd.DataFrame({"token": feature_names[spam_idx], "weight": weights[spam_idx]})
        df_ham = pd.DataFrame({"token": feature_names[ham_idx], "weight": weights[ham_idx]})
        return df_ham, df_spam
    except Exception:
        return None, None


def top_tokens_by_class_from_data(
    pipeline, X_train: List[str], y_train: List[str], top_n: int = 20
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fallback: derive class-specific tokens by average TF-IDF over training texts."""
    try:
        tfidf = pipeline.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        X = tfidf.transform(X_train)
        y = np.array(y_train)
        # ham = 0, spam = 1
        mask_spam = y == "spam"
        mask_ham = y == "ham"
        # average tfidf per class
        avg_spam = X[mask_spam].mean(axis=0).A1
        avg_ham = X[mask_ham].mean(axis=0).A1
        spam_idx = np.argsort(avg_spam)[::-1][:top_n]
        ham_idx = np.argsort(avg_ham)[::-1][:top_n]
        df_spam = pd.DataFrame({"token": feature_names[spam_idx], "avg_tfidf": avg_spam[spam_idx]})
        df_ham = pd.DataFrame({"token": feature_names[ham_idx], "avg_tfidf": avg_ham[ham_idx]})
        return df_ham, df_spam
    except Exception:
        return None, None

def save_curves_and_confusion(
    pipeline, X_test: List[str], y_test: List[str], out_dir: str = "artifacts"
) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    y_true_bin = np.array([1 if y == "spam" else 0 for y in y_test])
    scores = _positive_scores(pipeline, X_test)
    results: dict = {}
    if scores is not None:
        fpr, tpr, _ = roc_curve(y_true_bin, scores)
        prec, rec, _ = precision_recall_curve(y_true_bin, scores)

        # ROC
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve (positive=spam)")
        plt.legend()
        roc_path = Path(out_dir) / "roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        results["roc_curve"] = str(roc_path)

        # PR
        plt.figure(figsize=(5, 4))
        plt.plot(rec, prec, label="PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (positive=spam)")
        plt.legend()
        pr_path = Path(out_dir) / "pr_curve.png"
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()
        results["pr_curve"] = str(pr_path)

    # Confusion Matrix
    preds = pipeline.predict(X_test)
    labels = ["ham", "spam"]
    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(ticks=[0, 1], labels=labels)
    plt.yticks(ticks=[0, 1], labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = Path(out_dir) / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    results["confusion_matrix"] = str(cm_path)

    return results


def save_top_tokens_csv(pipeline, out_dir: str = "artifacts", top_n: int = 20) -> Tuple[str | None, str | None]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_ham, df_spam = top_tokens_by_class(pipeline, top_n=top_n)
    ham_path = Path(out_dir) / "top_tokens_ham.csv"
    spam_path = Path(out_dir) / "top_tokens_spam.csv"
    if df_ham is not None:
        df_ham.to_csv(ham_path, index=False)
    if df_spam is not None:
        df_spam.to_csv(spam_path, index=False)
    return (str(ham_path) if df_ham is not None else None, str(spam_path) if df_spam is not None else None)


def run_visualize_cli(
    csv_path: str = "sms_spam_no_header.csv",
    artifacts_dir: str = "artifacts",
    out_dir: str = "artifacts",
    test_size: float = 0.2,
    seed: int = 42,
    top_n: int = 20,
) -> dict:
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=test_size, random_state=seed)
    pipeline = load_pipeline(artifacts_dir)

    results = save_curves_and_confusion(pipeline, X_test, y_test, out_dir=out_dir)
    ham_csv, spam_csv = save_top_tokens_csv(pipeline, out_dir=out_dir, top_n=top_n)
    # fallback via training data averages if coef_ not available
    if ham_csv is None or spam_csv is None:
        df_ham, df_spam = top_tokens_by_class_from_data(pipeline, X_train, y_train, top_n=top_n)
        if df_ham is not None:
            ham_path = Path(out_dir) / "top_tokens_ham.csv"
            df_ham.to_csv(ham_path, index=False)
            ham_csv = str(ham_path)
        if df_spam is not None:
            spam_path = Path(out_dir) / "top_tokens_spam.csv"
            df_spam.to_csv(spam_path, index=False)
            spam_csv = str(spam_path)
    results.update({"top_tokens_ham": ham_csv, "top_tokens_spam": spam_csv})
    meta = {
        "csv_path": csv_path,
        "artifacts_dir": artifacts_dir,
        "out_dir": out_dir,
        "test_size": test_size,
        "seed": seed,
        "top_n": top_n,
        "generated": list(results.values()),
    }
    with open(Path(out_dir) / "visualize_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _cli():
    parser = argparse.ArgumentParser(description="Generate visualization artifacts (ROC/PR, confusion, top tokens)")
    parser.add_argument("--data", default="sms_spam_no_header.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    meta = run_visualize_cli(
        csv_path=args.data,
        artifacts_dir=args.artifacts,
        out_dir=args.out,
        test_size=args.test_size,
        seed=args.seed,
        top_n=args.top_n,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    _cli()
