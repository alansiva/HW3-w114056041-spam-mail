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
from spam_classification.visualize import top_tokens_by_class_from_data, save_top_tokens_csv


st.set_page_config(page_title="Spam Classification Demo", layout="wide")
st.title("📧 Spam Classification — Baseline Demo")
st.caption("TF-IDF + LinearSVC baseline with optional calibration; metrics and inference UI.")

# Anchors 與側邊導覽
st.markdown("<a id='top'></a>", unsafe_allow_html=True)
st.sidebar.header("選擇功能")
nav = st.sidebar.radio(
    "選擇功能",
    options=["訓練", "推論", "指標/視覺化", "關鍵字排行", "Artifacts"],
    index=0,
)
st.markdown(
    "目錄： [訓練](#train) | [推論](#infer) | [指標/視覺化](#metrics) | [關鍵字排行](#keywords) | [Artifacts](#artifacts)",
    unsafe_allow_html=True,
)


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


st.markdown("<a id='train'></a>", unsafe_allow_html=True)
with st.expander("訓練", expanded=(nav == "訓練")):
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
    st.markdown("[回到頂端](#top)", unsafe_allow_html=True)


st.markdown("<a id='infer'></a>", unsafe_allow_html=True)
with st.expander("推論（單則與自動測試器）", expanded=(nav == "推論")):
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
    st.markdown("[回到頂端](#top)", unsafe_allow_html=True)


st.markdown("<a id='metrics'></a>", unsafe_allow_html=True)
with st.expander("指標/視覺化", expanded=(nav == "指標/視覺化")):
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
            # 可調決策閾值（spam 為正類），優先使用 predict_proba，否則以 decision_function 經 sigmoid 近似
            cols_thr = st.columns(2)
            with cols_thr[0]:
                threshold = st.slider("決策閾值（spam）", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            with cols_thr[1]:
                st.caption("說明：若模型未校準，將以 decision_function 經 sigmoid 近似分數再套用閾值。")

            try:
                proba = pipeline.predict_proba(X_test)
                classes = list(pipeline.named_steps["clf"].classes_)
                pos_idx = classes.index("spam")
                scores = proba[:, pos_idx]
            except Exception:
                try:
                    margins = pipeline.decision_function(X_test)
                    scores = 1 / (1 + np.exp(-margins))
                    st.info("模型未校準，閾值套用於 decision_function 近似分數。")
                except Exception:
                    scores = None

            if scores is not None:
                preds = np.where(scores >= threshold, "spam", "ham")
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
                    "precision": [report[lab]["precision"] for lab in labels],
                    "recall": [report[lab]["recall"] for lab in labels],
                    "f1": [report[lab]["f1-score"] for lab in labels],
                    "support": [report[lab]["support"] for lab in labels],
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
            top_n = st.slider("Top-N tokens", min_value=5, max_value=50, value=20, step=5)
            df_ham, df_spam = _top_tokens_by_class(pipeline, top_n=int(top_n))
            if df_ham is not None and df_spam is not None:
                cols2 = st.columns(2)
                with cols2[0]:
                    st.dataframe(df_ham, use_container_width=True)
                with cols2[1]:
                    st.dataframe(df_spam, use_container_width=True)
            else:
                st.info("無法擷取 ham/spam 關鍵字排名（可能不支援或尚未訓練）。")
    st.markdown("[回到頂端](#top)", unsafe_allow_html=True)

st.markdown("<a id='keywords'></a>", unsafe_allow_html=True)
with st.expander("關鍵字排行（ham/spam）", expanded=(nav == "關鍵字排行")):
    st.subheader("關鍵字排行（ham/spam）")
    top_n_kw = st.slider("Top-N tokens", min_value=5, max_value=50, value=20, step=5, key="kw_top_n")
    source = st.radio("來源", options=["模型係數", "訓練資料平均TF-IDF"], index=0, horizontal=True, key="kw_source")

    pipeline = _load_pipeline_cached(ARTIFACTS_DIR)
    if pipeline is None:
        st.error("尚未找到已訓練的模型，請先在上方『模型訓練』區塊進行訓練。")
    else:
        df_ham_kw = None
        df_spam_kw = None
        if source == "模型係數":
            df_ham_kw, df_spam_kw = _top_tokens_by_class(pipeline, top_n=int(top_n_kw))
        else:
            metrics_cached = _load_metrics(ARTIFACTS_DIR)
            csv_path_kw = data_path if data_path else DATA_DEFAULT
            df_kw = load_dataset(csv_path_kw)
            ts = float(metrics_cached.get("test_size", 0.2)) if metrics_cached else 0.2
            rs = int(metrics_cached.get("random_state", 42)) if metrics_cached else 42
            X_train_kw, X_test_kw, y_train_kw, y_test_kw = split_dataset(df_kw, test_size=ts, random_state=rs)
            df_ham_kw, df_spam_kw = top_tokens_by_class_from_data(pipeline, X_train_kw, y_train_kw, top_n=int(top_n_kw))

        if df_ham_kw is not None and df_spam_kw is not None:
            cols_kw = st.columns(2)
            with cols_kw[0]:
                st.write("Ham Top Tokens")
                st.dataframe(df_ham_kw, use_container_width=True)
                st.download_button("下載 ham CSV", data=df_ham_kw.to_csv(index=False), file_name="top_tokens_ham.csv")
            with cols_kw[1]:
                st.write("Spam Top Tokens")
                st.dataframe(df_spam_kw, use_container_width=True)
                st.download_button("下載 spam CSV", data=df_spam_kw.to_csv(index=False), file_name="top_tokens_spam.csv")

            # 匯出到 artifacts（若來源為模型係數，直接使用 save_top_tokens_csv；否則以目前結果寫出）
            do_export = st.button("匯出至 artifacts")
            if do_export:
                try:
                    if source == "模型係數":
                        save_top_tokens_csv(pipeline, out_dir=ARTIFACTS_DIR, top_n=int(top_n_kw))
                    else:
                        Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
                        (Path(ARTIFACTS_DIR) / "top_tokens_ham.csv").write_text(df_ham_kw.to_csv(index=False))
                        (Path(ARTIFACTS_DIR) / "top_tokens_spam.csv").write_text(df_spam_kw.to_csv(index=False))
                    st.success("已匯出關鍵字排行至 artifacts。")
                except Exception as e:
                    st.error(f"匯出失敗：{e}")
        else:
            if source == "模型係數":
                st.info("無法透過模型係數擷取關鍵字排行，請改用『訓練資料平均TF-IDF』來源。")
            else:
                st.info("無法透過資料備援方式產生關鍵字排行，請確認資料與模型是否完整。")
    st.markdown("[回到頂端](#top)", unsafe_allow_html=True)


st.markdown("<a id='artifacts'></a>", unsafe_allow_html=True)
with st.expander("Artifacts", expanded=(nav == "Artifacts")):
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
    st.markdown("[回到頂端](#top)", unsafe_allow_html=True)
