## ADDED Requirements

### Requirement: Spam Email Classification Capability
系統應（SHALL）提供基準能力，使用 `sms_spam_no_header.csv` 將簡訊/電子郵件訊息分類為 `spam` 或 `ham`。

- 系統應（SHALL）載入資料集，並以固定隨機種子進行可重現的訓練/測試切分。
- 系統應（SHALL）以斷詞與 TF-IDF 向量化進行文字前處理。
- 系統應（SHALL）在第一階段以 SVM（LinearSVC）作為基準分類器；若需要信心分數，分類器應（SHOULD）透過 `CalibratedClassifierCV` 進行機率校準。
- 系統應（SHALL）在測試集上產出評估指標（accuracy、precision、recall、F1）。
- 系統應（SHALL）持久化模型與向量器，供後續推論使用。
- 系統宜（SHOULD）提供簡單的 CLI 或 Notebook 介面進行訓練與推論。

#### Scenario: Train baseline model and evaluate metrics
- WHEN 執行訓練命令，資料來源為 `sms_spam_no_header.csv`
- THEN 系統完成模型訓練、於測試集評估並將指標寫入 `artifacts/metrics.json`
- AND 模型產物保存於 `artifacts/`（例如 `model.pkl`、`vectorizer.pkl`）

#### Scenario: Infer single message returns label and confidence
- WHEN 使用者針對單一訊息字串執行推論
- THEN 系統回傳標籤（`spam` 或 `ham`）與信心分數
- AND 使用已持久化的模型與向量器產物
