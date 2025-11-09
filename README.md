# Email Spam Classification — UI、CLI 與測試指南

本專案提供單頁 Streamlit UI 與 CLI 工具，支援：
- 訓練/推論
- 指標/視覺化（ROC/PR、混淆矩陣、Top tokens）
- 自動產生訊息推論測試器（樣本生成＋批次推論）

## 環境安裝

```bash
pip3 install -r requirements.txt
```

## 啟動單頁 UI

```bash
python3 -m streamlit run app/main.py
```

打開瀏覽器連結（預設）：
- http://localhost:8502/

UI 區塊說明：
- 訓練：載入資料訓練模型；完成後會於 artifacts 產生 model/vectorizer 等檔案。
- 推論：單則訊息推論顯示標籤與信心分數。
- 指標/視覺化：顯示 ROC/PR、混淆矩陣與 Top tokens，提供 Top-N 與決策閾值 slider。
- 訊息推論測試器（自動產生常見文本）：選擇語言/類別/數量/比例，生成樣本並立即推論。
- Artifacts：檢視並下載生成的 PNG/CSV/JSON。

## CLI — 自動產生訊息與批次推論

生成樣本（samples.json）：
```bash
python3 -m spam_classification.samples \
  --lang zh --category mixed --n 20 --spam-ratio 0.5 \
  --out artifacts/samples.json
```

批次推論（auto_tester_predictions.json）：
```bash
python3 -m spam_classification.infer \
  --input artifacts/samples.json \
  --out artifacts/auto_tester_predictions.json
```

## CLI — 指標/視覺化與 Top tokens

生成離線圖檔與 CSV：
```bash
python3 -m spam_classification.visualize \
  --data sms_spam_no_header.csv \
  --artifacts artifacts \
  --out artifacts \
  --test-size 0.2 \
  --seed 42 \
  --top-n 20
```

輸出：
- artifacts/roc_curve.png
- artifacts/pr_curve.png
- artifacts/confusion_matrix.png
- artifacts/top_tokens_ham.csv
- artifacts/top_tokens_spam.csv
- artifacts/visualize_meta.json

排名方法說明：
- 若分類器可提供係數（例如 LinearSVC），Top tokens 以權重（coef）排序：spam 取最大權重、ham 取最小權重。
- 若無係數，回退為以訓練集 TF‑IDF 平均值計算：針對 ham/spam 各自計算平均 TF‑IDF，取 Top‑N 最高者。

ROC/PR 分數來源：
- 優先使用 predict_proba 的 spam 機率作為正類分數。
- 若無機率，使用 decision_function 的 margin，並以邏輯函數映射到 [0,1] 作為近似分數，用於曲線繪製。

## 測試與品質

執行測試：
```bash
python3 -m pytest -q
```

靜態檢查（可選）：
```bash
ruff check .
```

