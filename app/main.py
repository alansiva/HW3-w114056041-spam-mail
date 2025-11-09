from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve

from spam_classification.train import train as train_fn
from spam_classification.infer import infer_single, load_pipeline
from spam_classification.data import load_dataset, split_dataset
from spam_classification.samples import generate_batch


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


def _top_tokens_by_class(pipeline, top_n: int = 20) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return Top-N tokens for ham (negative weights) and spam (positive weights)."""
    try:
        tfidf = pipeline.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "estimator"):
            base = clf.estimator
        else:
            base = clf
        coefs = getattr(base, "coef_", None)
        if coefs is None:
            return None, None
        weights = coefs[0] if coefs.ndim == 2 else coefs
        # spam: largest positive weights
        spam_idx = np.argsort(weights)[::-1][:top_n]
        # ham: most negative weights
        ham_idx = np.argsort(weights)[:top_n]
        df_spam = pd.DataFrame({"token": feature_names[spam_idx], "weight": weights[spam_idx]})
        df_ham = pd.DataFrame({"token": feature_names[ham_idx], "weight": weights[ham_idx]})
        return df_ham, df_spam
    except Exception:
        return None, None


# 單頁介面：移除側欄選單，改為在同一頁連續呈現各區塊
data_path = st.text_input("資料檔案路徑", value=DATA_DEFAULT)
st.divider()


st.subheader("模型訓練")
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


st.divider()
st.subheader("單訊息推論")
message = st.text_area("輸入欲分類的訊息文本")
predict = st.button("推論")

if predict:
    pipeline = _load_pipeline_cached(ARTIFACTS_DIR)
    if pipeline is None:
        st.error("尚未找到已訓練的模型，請先在上方『模型訓練』區塊進行訓練。")
    else:
        label, conf = infer_single(message, ARTIFACTS_DIR)
        st.write({"label": label, "confidence": round(conf, 4)})


st.divider()
st.subheader("訊息推論測試器（自動產生常見文本）")
cols_gen = st.columns(4)
with cols_gen[0]:
    lang_opt = st.radio("語言", options=["中文", "English"], index=0, horizontal=True)
    lang = "zh" if lang_opt == "中文" else "en"
with cols_gen[1]:
    category_opt = st.selectbox("類別", options=["隨機", "spam", "ham", "混合"], index=0)
    category_map = {"隨機": "random", "spam": "spam", "ham": "ham", "混合": "mixed"}
    category = category_map[category_opt]
with cols_gen[2]:
    n_samples = st.slider("生成數量", min_value=1, max_value=10, value=3)
with cols_gen[3]:
    spam_ratio = st.slider("混合中的 spam 比例", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

do_generate = st.button("產生並推論")
if do_generate:
    pipeline = _load_pipeline_cached(ARTIFACTS_DIR)
    if pipeline is None:
        st.error("尚未找到已訓練的模型，請先在上方『模型訓練』區塊進行訓練。")
    else:
        batch = generate_batch(n=n_samples, lang=lang, category=category, spam_ratio=spam_ratio)
        for i, s in enumerate(batch, start=1):
            lbl, conf = infer_single(s["text"], ARTIFACTS_DIR)
            ok = (lbl == s["expected_label"]) if s.get("expected_label") else None
            with st.container(border=True):
                st.markdown(f"**訊息 {i}**（{s['lang']} / 期望：{s['expected_label']}）")
                st.write(s["text"])
                st.write({"predicted": lbl, "confidence": round(conf, 4)})
                if ok is True:
                    st.success("預測與期望一致。")
                elif ok is False:
                    st.warning("預測與期望不一致，請檢視樣本或調整模型。")


st.divider()
st.subheader("評估指標與視覺化")
metrics = _load_metrics(ARTIFACTS_DIR)
if not metrics:
    st.warning("尚未找到 metrics.json，請先在上方『模型訓練』區塊執行一次訓練以產生評估指標。")
else:
    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    st.metric("F1 (weighted)", f"{metrics['f1_weighted']:.4f}")

    # 依據記錄的 test_size/seed 重建測試集以產生視覺化（使用已訓練 pipeline 預測）
    csv_path = data_path if data_path else DATA_DEFAULT
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = split_dataset(
        df, test_size=float(metrics.get("test_size", 0.2)), random_state=int(metrics.get("random_state", 42))
    )
    pipeline = _load_pipeline_cached(ARTIFACTS_DIR)
    if pipeline is None:
        st.error("尚未找到已訓練的模型，請先在上方『模型訓練』區塊進行訓練。")
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

        # ROC / PR curves
        st.write("ROC 與 Precision-Recall 曲線")
        # positive class is spam -> y_true_binary: 1 for spam, 0 for ham
        y_true_bin = np.array([1 if y == "spam" else 0 for y in y_test])
        try:
            # Use probability if available
            proba = pipeline.predict_proba(X_test)
            # Find index for positive class
            classes = list(pipeline.named_steps["clf"].classes_)
            pos_idx = classes.index("spam")
            y_scores = proba[:, pos_idx]
        except Exception:
            # Fall back to decision_function
            try:
                y_scores = pipeline.decision_function(X_test)
                # normalize to [0,1]
                y_scores = 1 / (1 + np.exp(-y_scores))
                st.info("模型未校準，ROC/PR 以 decision_function 近似生成。")
            except Exception:
                y_scores = None
        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
            prec, rec, _ = precision_recall_curve(y_true_bin, y_scores)
            df_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr})
            df_pr = pd.DataFrame({"Recall": rec, "Precision": prec})
            roc_chart = alt.Chart(df_roc).mark_line().encode(x="FPR", y="TPR")
            pr_chart = alt.Chart(df_pr).mark_line().encode(x="Recall", y="Precision")
            cols = st.columns(2)
            with cols[0]:
                st.altair_chart(roc_chart, use_container_width=True)
            with cols[1]:
                st.altair_chart(pr_chart, use_container_width=True)

        st.write("Top Tokens by Class（ham/spam）")
        df_ham, df_spam = _top_tokens_by_class(pipeline, top_n=20)
        if df_ham is not None and df_spam is not None:
            cols2 = st.columns(2)
            with cols2[0]:
                st.dataframe(df_ham, use_container_width=True)
            with cols2[1]:
                st.dataframe(df_spam, use_container_width=True)
        else:
            st.info("無法擷取 ham/spam 關鍵字排名（可能不支援或尚未訓練）。")


st.divider()
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
