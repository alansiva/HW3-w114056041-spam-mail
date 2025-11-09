# ADDED Requirements — Spam Classification 視覺化與關鍵字排名

本增補針對既有 spam-classification 規格，新增 UI 與 CLI 的視覺化與分析要求。

- UI MUST 顯示：
  - ROC curve 與 Precision-Recall curve（線圖）。
  - Confusion matrix（表格或熱力圖）。
  - Top tokens by class：分別顯示 ham 與 spam 的 Top-N tokens 排名，並在文件中說明計算方法（TF-IDF 權重或類內詞頻）。
- CLI MUST 匯出上述視覺化與排名至 artifacts：
  - 圖檔：`roc_curve.png`、`pr_curve.png`、`confusion_matrix.png`。
  - 資料檔：`top_tokens_ham.csv`、`top_tokens_spam.csv`（至少包含 token 與 score/count 欄位）。
- 若分類器未提供概率（未校準）：
  - UI/CLI SHOULD 使用 decision_function 近似生成 ROC/PR，並在 UI/文件中提示近似方式。
- 視覺化/輸出 MUST 可在無網路環境下生成（不依賴外部服務）。

---

# Scenarios

## Scenario: 產出並展示 ROC 與 Precision-Recall 線圖
- GIVEN 已完成訓練並存在 artifacts/model.joblib
- AND 使用者於 UI 開啟「Model Performance」分頁
- WHEN 系統載入測試集並計算曲線資料
- THEN UI 顯示 ROC 與 Precision-Recall 線圖
- AND 若模型未校準，UI 顯示「以 decision_function 近似」之提示

## Scenario: 顯示混淆矩陣
- GIVEN 已完成訓練並存在 artifacts/model.joblib
- WHEN 使用已訓練模型對測試集推論
- THEN 生成混淆矩陣並在 UI 顯示（表格或熱力圖）
- AND CLI 匯出 `confusion_matrix.png` 至 artifacts

## Scenario: 類別關鍵字排名（ham/spam Top-N tokens）
- GIVEN 已完成 TF-IDF 向量化或可拆解為詞頻計算
- WHEN 使用者在 UI 開啟「Top Tokens by Class」
- THEN 顯示 ham、spam 的 Top-N tokens 排名
- AND 排名方法（TF-IDF 權重或類內詞頻）在文件中說明
- AND CLI 匯出 `top_tokens_ham.csv` 與 `top_tokens_spam.csv`

## Scenario: CLI 批次輸出視覺化與排名
- GIVEN 使用者執行 `python -m spam_classification.visualize --data <csv> --artifacts artifacts --out artifacts`
- WHEN 指令完成
- THEN 在 artifacts 生成 `roc_curve.png`、`pr_curve.png`、`confusion_matrix.png`
- AND 生成 `top_tokens_ham.csv`、`top_tokens_spam.csv`

## Notes
- Scenario 關鍵詞（GIVEN/WHEN/THEN/AND）與 ADDED Requirements 保留英文，以符合 OpenSpec 檢驗要求。
