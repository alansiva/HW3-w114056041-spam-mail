# Proposal: 在 UI 顯示線圖與 ham/spam 關鍵字排名（Spam Classification 視覺化增補）

## Why
- 提升模型可解釋性與可示範性：除了輸出整體指標外，透過線圖（ROC、Precision-Recall）與混淆矩陣、關鍵字排名，讓使用者直觀理解分類器的表現與錯誤型態。
- 與教學/作業需求相符：常見的分類問題需要展示 ROC/PR 曲線與每類別的 Top tokens，以便在報告或互動式頁面中呈現模型行為。
- 對應使用者需求：使用者要求「畫面中顯示相關的線圖和相關的 ham/spam 關鍵字排名」。

## What Changes
- ADDED Requirements 到現有 spam-classification 規格：
  - UI 必須提供：
    - 線圖：ROC curve 與 Precision-Recall curve（採用已校準概率或決策分數生成）。
    - 混淆矩陣（數值表格與熱力圖皆可）。
    - ham 與 spam 的關鍵字排名（Top-N tokens），可由 TF-IDF 權重或每類詞頻計算；方法需在說明/文件中透明化。
  - CLI/工具需支援將上述視覺化與排名輸出至 artifacts（圖檔與 CSV/JSON）。
  - 若分類器未提供概率（未校準）時，ROC/PR 需以 decision_function 近似並在 UI/文件中提示。
- 新增/調整程式：
  - 新增可重用的可視化/分析模組（如 `spam_classification/visualize.py`）產生 ROC/PR、混淆矩陣與 Top tokens（ham/spam）。
  - 擴充 Streamlit app（`app/main.py`）頁面：
    - Model Performance：顯示混淆矩陣、ROC/PR 線圖。
    - Top Tokens by Class：分別展示 ham/spam Top-N tokens 排名。
  - 新增 CLI 指令輸出：`python -m spam_classification.visualize --artifacts artifacts --data <csv> --out artifacts`（名稱可在實作時定稿）。

## Impact
- 訓練流程不變；但為支援 ROC/PR 建議預設採用 CalibratedClassifierCV 以取得 probabilities。若停用校準，則以 decision_function 近似，並在 UI 明示近似來源。
- 增加新的 artifacts（如 `roc_curve.png`、`pr_curve.png`、`confusion_matrix.png`、`top_tokens_ham.csv`、`top_tokens_spam.csv`），有助於報告與驗證。
- UI 與 CLI 的一致性提升：互動與批次皆可產生同款視覺化與排名，利於可複製與教學。

## Notes
- 保留 OpenSpec 必要標記（ADDED Requirements、Scenario、WHEN/THEN/AND）。
- 視覺化可採 Altair 或 Matplotlib；排名可採用：
  - TF-IDF 權重（線性 SVM 係數的解釋以正類權重為主）或
  - 類內詞頻（對每類文本的詞頻統計）。
- 若需雲端示範，可延伸部署至 Streamlit Community Cloud（`streamlit.app`），作為 Phase 後續項目。
