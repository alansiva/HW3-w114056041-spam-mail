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
 - http://localhost:8501/

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


## 專案分析（Problem & Approach）

- 任務：以 `sms_spam_no_header.csv` 資料集作為基準，將短訊/郵件文字分類為 `spam` 或 `ham`。
- 方法：
  - 前處理採用 `TfidfVectorizer`（n-gram 範圍 1–2，`max_features=20000`）。
  - 基準分類器為 `LinearSVC`；若需要機率分數以繪製 ROC/PR，則以 `CalibratedClassifierCV(method="sigmoid", cv=5)` 進行校準。
  - 可重現：固定隨機種子（預設 `42`）、測試集比例（預設 `0.2`）。
- 產出：模型與向量器（`artifacts/model.joblib`、`artifacts/vectorizer.joblib`）、指標（`artifacts/metrics.json`）、視覺化與關鍵字排行檔案（PNG/CSV）。


## Spec 重點功能（OpenSpec 摘要）

本專案遵循 OpenSpec 工作流，重點要求如下：
- SHALL 以固定隨機種子進行可重現的 train/test 切分。
- SHALL 以斷詞與 TF‑IDF 向量化進行文字前處理。
- SHALL 以 SVM（LinearSVC）做為基準模型；需要信心分數時 SHOULD 透過 CalibratedClassifierCV 校準。
- SHALL 在測試集上產出 accuracy、precision、recall、F1 指標並持久化為 `artifacts/metrics.json`。
- SHALL 持久化模型與向量器供後續推論使用（`model.joblib`、`vectorizer.joblib`）。
- SHOULD 提供 CLI/Notebook 或 UI 介面完成訓練與推論。

更多細節可見：
- `openspec/project.md`
- `openspec/specs/spam-classification/spec.md`
- `openspec/changes/` 目錄下的 proposal/docs/specs/tasks/validation


## 結果報告（Results & Artifacts）

- 指標來源：`artifacts/metrics.json`
  - 以範例設定（`test_size=0.2`, `random_state=42`, `max_features=20000`, `calibrated=true`；UI 中校準選項預設為開啟）訓練所得：
    - accuracy ≈ 0.98296
    - precision_weighted ≈ 0.98328
    - recall_weighted ≈ 0.98296
    - f1_weighted ≈ 0.98308
- 視覺化：
  - ROC 曲線：`artifacts/roc_curve.png`
  - PR 曲線：`artifacts/pr_curve.png`
  - 混淆矩陣：`artifacts/confusion_matrix.png`
- 關鍵字排行（Top tokens）：
  - `artifacts/top_tokens_spam.csv`
  - `artifacts/top_tokens_ham.csv`
  - 若分類器提供係數（如 LinearSVC），以權重進行排序；若無係數則回退為訓練集 TF‑IDF 平均值排序。


## 平均效能（Test-set Performance）

使用 `random_state=42`、`test_size=0.2` 之測試集平均效能如下：
- Accuracy：0.983
- Precision（weighted）：0.983
- Recall（weighted）：0.983
- F1（weighted）：0.983

註：不同隨機種子或資料切分比例可能造成微幅變動；若固定版本並重新訓練，數值可持續穩定在上述水準。


## 如何使用（操作導覽）

1) 安裝相依：
```bash
pip3 install -r requirements.txt
```

2) 啟動 UI（單頁 Streamlit）：
```bash
# 預設埠：8501
python3 -m streamlit run app/main.py

# 或指定埠（例如 8504）：
streamlit run app/main.py --server.port 8504 --server.headless true
```

3) 訓練模型（CLI）：
```bash
python3 -m spam_classification.train \
  --data sms_spam_no_header.csv \
  --out artifacts \
  --test-size 0.2 \
  --seed 42 \
  --calibrated \
  --max-features 20000
```

4) 生成樣本與批次推論（CLI）：
```bash
# 生成樣本
python3 -m spam_classification.samples \
  --lang zh --category mixed --n 20 --spam-ratio 0.5 \
  --out artifacts/samples.json

# 批次推論
python3 -m spam_classification.infer \
  --input artifacts/samples.json \
  --out artifacts/auto_tester_predictions.json
```

5) 離線視覺化與排行（CLI）：
```bash
python3 -m spam_classification.visualize \
  --data sms_spam_no_header.csv \
  --artifacts artifacts \
  --out artifacts \
  --test-size 0.2 \
  --seed 42 \
  --top-n 20
```


## 測試（Testing & Quality）

- 執行測試：
```bash
python3 -m pytest -q
```

- 靜態檢查（可選）：
```bash
ruff check .
```

- 測試覆蓋：
  - 單則推論（`tests/test_infer.py`）
  - 視覺化與關鍵字排行（`tests/test_visualize.py`）
  - 樣本生成（`tests/test_samples.py`）

若需更高覆蓋率，建議新增：
- 訓練流程 smoke test（`spam_classification/train.py`）
- 指標計算與 artifacts 輸出整合測試
