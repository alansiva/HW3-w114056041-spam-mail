from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from spam_classification.train import train as train_fn
from spam_classification.infer import infer_single, load_pipeline
from spam_classification.data import load_dataset, split_dataset


st.set_page_config(page_title="Spam Classification Demo", layout="wide")
st.title("📧 Spam Classification — Baseline Demo")
st.caption("TF-IDF + LinearSVC baseline with optional calibration; metrics and inference UI.")


ARTIFACTS_DIR = "artifacts"
DATA_DEFAULT = "sms_spam_no_header.csv"


@st.cache_resource
def _load_pipeline_cached(artifacts_dir: str = ARTIFACTS_DIR):
    try:
        return load_pipeline(artifacts_dir)
    except Exception as e:
        st.warning(f"Pipeline 未載入：{e}")
        return None


def _load_metrics(artifacts_dir: str = ARTIFACTS_DIR) -> dict | None:
    p = Path(artifacts_dir) / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _plot_confusion(cm: np.ndarray, labels: Tuple[str, str]):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm = df_cm.reset_index().melt(id_vars="index")
    df_cm.columns = ["True", "Pred", "Count"]
    chart = (
        alt.Chart(df_cm)
        .mark_rect()
        .encode(
            x=alt.X("Pred:N", title="Predicted"),
            y=alt.Y("True:N", title="True"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["True", "Pred", "Count"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _top_features(pipeline, top_n: int = 20) -> pd.DataFrame | None:
    try:
        tfidf = pipeline.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        clf = pipeline.named_steps["clf"]
        # unwrap calibrated classifier if used
        if hasattr(clf, "estimator"):
            base = clf.estimator
        else:
            base = clf
        # For binary classification, coef_ shape is (1, n_features) or (2, n_features)
        coefs = getattr(base, "coef_", None)
        if coefs is None:
            return None
        # assume positive class is spam; take largest positive weights
        weights = coefs[0] if coefs.ndim == 2 else coefs
        idx = np.argsort(weights)[::-1][:top_n]
        return pd.DataFrame({"feature": feature_names[idx], "weight": weights[idx]})
    except Exception:
        return None


tab = st.sidebar.radio("選擇功能", ["訓練", "推論", "指標/視覺化", "Artifacts"])


if tab == "訓練":
    st.subheader("模型訓練")
    data_path = st.text_input("資料檔案路徑", value=DATA_DEFAULT)
    test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input("隨機種子", value=42, step=1)
    calibrated = st.checkbox("啟用概率校準 (CalibratedClassifierCV)", value=True)
    max_features = st.number_input("TF-IDF 最大特徵數", value=20000, step=1000)
    run = st.button("開始訓練")

    if run:
        if not Path(data_path).exists():
            st.error(f"資料檔案不存在：{data_path}")
        else:
            with st.spinner("訓練中，請稍候..."):
                metrics = train_fn(
                    csv_path=data_path,
                    out_dir=ARTIFACTS_DIR,
                    test_size=float(test_size),
                    random_state=int(seed),
                    calibrated=bool(calibrated),
                    max_features=int(max_features),
                )
            st.success("訓練完成！")
            st.json(metrics)


elif tab == "推論":
    st.subheader("單訊息推論")
    message = st.text_area("輸入欲分類的訊息文本")
    predict = st.button("推論")

    if predict:
        pipeline = _load_pipeline_cached(ARTIFACTS_DIR)
        if pipeline is None:
            st.error("尚未找到已訓練的模型，請先到『訓練』頁進行訓練。")
        else:
            label, conf = infer_single(message, ARTIFACTS_DIR)
            st.write({"label": label, "confidence": round(conf, 4)})


elif tab == "指標/視覺化":
    st.subheader("評估指標與視覺化")
    metrics = _load_metrics(ARTIFACTS_DIR)
    if not metrics:
        st.warning("尚未找到 metrics.json，請先至『訓練』頁執行一次訓練以產生評估指標。")
    else:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("F1 (weighted)", f"{metrics['f1_weighted']:.4f}")

        # 依據記錄的 test_size/seed 重建測試集以產生視覺化（使用已訓練 pipeline 預測）
        csv_path = DATA_DEFAULT
        df = load_dataset(csv_path)
        X_train, X_test, y_train, y_test = split_dataset(
            df, test_size=float(metrics.get("test_size", 0.2)), random_state=int(metrics.get("random_state", 42))
        )
        pipeline = _load_pipeline_cached(ARTIFACTS_DIR)
        if pipeline is None:
            st.error("尚未找到已訓練的模型，請先到『訓練』頁進行訓練。")
        else:
            preds = pipeline.predict(X_test)
            labels = ("ham", "spam")
            cm = confusion_matrix(y_test, preds, labels=list(labels))
            st.write("Confusion Matrix")
            _plot_confusion(cm, labels)

            st.write("Classification Report (per class)")
            report = classification_report(y_test, preds, labels=list(labels), output_dict=True, zero_division=0)
            df_rep = pd.DataFrame(
                {
                    "label": labels,
                    "precision": [report[l]["precision"] for l in labels],
                    "recall": [report[l]["recall"] for l in labels],
                    "f1": [report[l]["f1-score"] for l in labels],
                    "support": [report[l]["support"] for l in labels],
                }
            )
            st.dataframe(df_rep, use_container_width=True)

            st.write("Top TF-IDF features (by LinearSVC weights)")
            df_top = _top_features(pipeline, top_n=20)
            if df_top is not None:
                st.dataframe(df_top, use_container_width=True)
            else:
                st.info("無法擷取特徵權重（可能不支援或尚未訓練）。")


elif tab == "Artifacts":
    st.subheader("Artifacts 檢視/下載")
    p = Path(ARTIFACTS_DIR)
    if not p.exists():
        st.warning("尚未生成 artifacts。")
    else:
        files = list(p.glob("*"))
        st.write("現有檔案：", [f.name for f in files])
        # 提供下載按鈕（metrics.json / model.joblib）
        m = p / "metrics.json"
        if m.exists():
            st.download_button("下載 metrics.json", data=m.read_text(), file_name="metrics.json")
        model = p / "model.joblib"
        if model.exists():
            st.download_button("下載 model.joblib", data=model.read_bytes(), file_name="model.joblib")

