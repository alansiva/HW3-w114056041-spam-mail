from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from joblib import load


def load_pipeline(artifacts_dir: str) -> object:
    pipeline_path = Path(artifacts_dir) / "model.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {pipeline_path}")
    return load(pipeline_path)


def infer_single(message: str, artifacts_dir: str = "artifacts") -> Tuple[str, float]:
    pipeline = load_pipeline(artifacts_dir)
    label = pipeline.predict([message])[0]

    # Try probability, otherwise approximate confidence via decision function
    confidence: float
    try:
        proba = pipeline.predict_proba([message])[0]
        classes = list(pipeline.named_steps["clf"].classes_)
        idx = classes.index(label)
        confidence = float(proba[idx])
    except Exception:
        # decision_function returns margin; map to [0,1] via logistic on absolute margin
        margins = pipeline.decision_function([message])
        margin = float(np.atleast_1d(margins)[0])
        confidence = float(1.0 / (1.0 + math.exp(-abs(margin))))
    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer single message label and confidence")
    parser.add_argument("message", help="text message to classify")
    parser.add_argument("--artifacts", default="artifacts", help="artifacts directory")
    args = parser.parse_args()

    lbl, conf = infer_single(args.message, args.artifacts)
    print({"label": lbl, "confidence": round(conf, 4)})

