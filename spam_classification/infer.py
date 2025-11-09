from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, List, Dict

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


def _cli():
    parser = argparse.ArgumentParser(description="Infer message(s) label and confidence")
    parser.add_argument("message", nargs="?", help="text message to classify")
    parser.add_argument("--artifacts", default="artifacts", help="artifacts directory")
    parser.add_argument("--input", default="", help="JSON file of messages: [{text, expected_label?}]")
    parser.add_argument("--out", default="", help="output JSON path for batch predictions")
    args = parser.parse_args()

    if args.input:
        inp_path = Path(args.input)
        if not inp_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {inp_path}")
        items = json.loads(inp_path.read_text())
        results: List[Dict[str, object]] = []
        for s in items:
            txt = s.get("text", "")
            lbl, conf = infer_single(txt, args.artifacts)
            res = {
                "text": txt,
                "predicted": lbl,
                "confidence": round(conf, 4),
            }
            if "expected_label" in s:
                res["expected_label"] = s["expected_label"]
                res["match"] = (s["expected_label"] == lbl)
            results.append(res)
        if args.out:
            Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
            print(f"Saved {len(results)} predictions to {args.out}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if not args.message:
            parser.error("please provide either --input JSON or a single 'message'")
        lbl, conf = infer_single(args.message, args.artifacts)
        print({"label": lbl, "confidence": round(conf, 4)})


if __name__ == "__main__":
    _cli()
