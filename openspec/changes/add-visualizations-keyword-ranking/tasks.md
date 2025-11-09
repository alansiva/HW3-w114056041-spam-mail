# Tasks — UI 視覺化與關鍵字排名增補（含勾選）

本變更分階段實作，聚焦於在 Streamlit 與 CLI 產出線圖（ROC、Precision-Recall）、混淆矩陣與 ham/spam Top tokens 排名。

## Phase 2 — 實作與整合（UI 與模組）
- [x] 在 `app/main.py` 顯示混淆矩陣（表格/熱力圖）。
- [x] 在 `app/main.py` 顯示 ROC 曲線與 Precision-Recall 曲線（線圖）。
- [x] 在 `app/main.py` 顯示 Top Tokens by Class（ham/spam 各自 Top-N）。
- [x] 在 `app/main.py` 加入 Top-N slider、決策閾值 slider（與校準機率連動）。
- [x] 介面支援 test_size、seed、是否 calibrated（訓練區塊可設定）。
- [x] 新增 `spam_classification/visualize.py`（曲線資料、混淆矩陣、排名計算封裝）。

## Phase 2 — CLI 批次輸出（Artifacts）
- [x] `python -m spam_classification.visualize --data <csv> --artifacts artifacts --out artifacts` 指令。
- [x] 產出 `roc_curve.png`、`pr_curve.png`、`confusion_matrix.png`、`top_tokens_ham.csv`、`top_tokens_spam.csv`。
- [x] 文件：說明排名計算方法（TF-IDF 權重 vs 類內詞頻），以及未校準時的 ROC/PR 近似法。

## Phase 3 — 測試與品質
- [x] 單元測試：`top_tokens_by_class` 對資料驗證產出 Top‑N 結構（含係數不可用時的 TF‑IDF 平均回退）。
- [x] 單元/功能測試：曲線資料計算涵蓋概率與決策分數，能成功生成 ROC/PR PNG。
- [x] 整合測試：執行 CLI 生成 artifacts，檢查檔案存在與基本結構（CSV 欄位、PNG 可開啟）。
- [x] 程式品質：lint（ruff）檢查通過，錯誤訊息涵蓋缺檔與未訓練情況。

## Phase 4 — 驗證與示範
- [x] 規格驗證：逐條對照 ADDED Requirements 與 Scenario，確認 UI 與 CLI 均符合（詳見 validation.md）。
- [x] 部署示範（可選）：提供本地／雲端部署指引（validation.md 內附簡要步驟）。
- [x] 文件與截圖：更新 README/openspec（CLI 範例與方法說明）。

## Notes
- 視覺化採 Altair（互動）與 Matplotlib（離線匯出）的混合策略。
- 未校準分類器的 ROC/PR 解讀，以 decision_function 近似並在文件中說明。
