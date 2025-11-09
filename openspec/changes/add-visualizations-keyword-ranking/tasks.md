# Tasks — UI 視覺化與關鍵字排名增補

本變更分階段實作，聚焦於在 Streamlit 與 CLI 產出線圖（ROC、Precision-Recall）、混淆矩陣與 ham/spam Top tokens 排名。

## Phase 2 — 實作與整合
- App/UI：
  - 在 `app/main.py` 增設分頁或區塊：
    - Model Performance：
      - 顯示混淆矩陣（表格與熱力圖）。
      - 顯示 ROC 曲線與 Precision-Recall 曲線（線圖）。
    - Top Tokens by Class：
      - 顯示 ham/spam 各自 Top-N tokens 排名（可視化表格/條形圖）。
  - 介面參數：Top-N、test_size、seed、是否 calibrated、（可選）決策閾值 slider。
- 核心分析模組：
  - 新增 `spam_classification/visualize.py`：
    - 函式：`plot_roc_pr(pipeline, X_test, y_test)` 生成 ROC/PR 曲線資料與圖表。
    - 函式：`confusion(pipeline, X_test, y_test)` 回傳混淆矩陣數值。
    - 函式：`top_tokens_by_class(vectorizer, texts, labels, top_n)` 回傳 ham/spam 排名（詞頻或 TF-IDF 權重）。
- CLI：
  - 新增批次輸出指令（名稱可定稿）：
    - `python -m spam_classification.visualize --data <csv> --artifacts artifacts --out artifacts`。
    - 產出：`roc_curve.png`、`pr_curve.png`、`confusion_matrix.png`、`top_tokens_ham.csv`、`top_tokens_spam.csv`。
- 文件：
  - 說明排名計算方法（TF-IDF 權重 vs 類內詞頻）與 ROC/PR 在未校準時的近似法。

## Phase 3 — 測試與品質
- 單元測試：
  - `top_tokens_by_class` 對小型合成資料驗證排名合理性（含停用詞過濾）。
  - `plot_roc_pr` 對概率輸出與決策分數輸出皆可計算曲線資料。
- 整合測試：
  - 執行 CLI 生成 artifacts，檢查檔案存在與基本結構（CSV 欄位、PNG 可開啟）。
- 程式品質：
  - 型別檢查、lint、格式化；視覺化的輸出介面穩健性（缺檔與未訓練情況錯誤訊息）。

## Phase 4 — 驗證與示範
- 規格驗證：
  - 逐條對照 ADDED Requirements 與 Scenario，確認 UI 與 CLI 均符合。
- 部署示範（可選）：
  - 提供部署指引到 Streamlit Community Cloud（`streamlit.app`）。
- 文件與截圖：
  - 更新 README/openspec 內嵌截圖，佐證完成狀態。

## Notes
- 視覺化建議使用 Altair（互動）與 Matplotlib（離線匯出），以利不同情境使用。
- 若分類器未校準，ROC/PR 的數值解讀需保留說明（以 decision_function 近似）。
