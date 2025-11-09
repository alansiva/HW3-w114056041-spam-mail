# 規格驗證與示範

本文件記錄視覺化與關鍵字排名增補的驗證步驟：

## 本地驗證
1. 先執行訓練，確保 `artifacts/model.joblib` 與 `artifacts/vectorizer.joblib` 存在。
2. 執行 CLI 生成離線 artifacts：
   ```bash
   python3 -m spam_classification.visualize \
     --data sms_spam_no_header.csv \
     --artifacts artifacts \
     --out artifacts \
     --test-size 0.2 \
     --seed 42 \
     --top-n 20
   ```
   期望檔案：`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `top_tokens_ham.csv`, `top_tokens_spam.csv`, `visualize_meta.json`。
3. 啟動單頁 UI 並於「指標/視覺化」區塊檢視 ROC/PR、混淆矩陣與 Top tokens；調整 Top‑N 與決策閾值 slider 驗證互動。

## 規格對照
- ADDED Requirements：
  - UI 中提供 ROC/PR、混淆矩陣與 Top tokens：已完成。
  - 可調整 Top‑N 與決策閾值：已完成。
  - CLI 匯出 PNG/CSV artifacts：已完成。
  - 檔案與方法說明（未校準近似、排名計算）：已在 docs.md 與 README 補充。

## 部署示範（可選）
- 本地：`python3 -m streamlit run app/main.py`，預設 http://localhost:8502/。
- 雲端（Community Cloud 指引簡要）：
  - 將專案推送到公共倉庫（需包含 requirements.txt）。
  - 在 Streamlit Cloud 建立新應用，入口指向 `app/main.py`。
  - 設定環境變數與檔案路徑後部署，並在「Artifacts」區確認 PNG/CSV 是否生成。

