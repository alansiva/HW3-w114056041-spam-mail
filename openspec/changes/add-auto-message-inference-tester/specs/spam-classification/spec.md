# 變更規格：訊息推論測試器（自動產生常見文本）

## 新增需求（ADDED Requirements）
1. UI（單頁）必須提供「訊息推論測試器」區塊：
   - 支援語言選擇（中文/English）。
   - 支援類別選擇（spam/ham/混合/隨機）。
   - 可設定生成數量（至少 1 至多 10）。
   - 混合模式可設定 spam 比例（0.0–1.0、步進 0.05）。
   - 生成後，系統自動對每則訊息進行推論並顯示：期望標籤、預測標籤與信心分數。
   - 若無已訓練模型，提示使用者先執行訓練。

2. 模組層：
   - `spam_classification/samples.py` 提供生成 API：
     - `generate_message(lang, category)`、`generate_batch(n, lang, category, spam_ratio)`。
     - 以固定模板＋安全占位符生成，不含真實個資與網址。

3. CLI（後續）：
   - 允許批次生成＋推論並輸出 JSON artifacts（含訊息、期望標籤、預測標籤、信心分數）。

## 情境（Scenario）
1. 使用者在單頁 UI 選擇：語言＝中文、類別＝混合、生成數量＝5、spam 比例＝0.6，按下「產生並推論」。
2. 系統生成 5 則常見訊息（約 3 則 spam、2 則 ham），並使用當前 artifacts 的模型進行推論。
3. UI 逐則顯示（期望 vs 預測 vs 信心），若有不一致則顯示警示訊息；否則顯示成功訊息。
4. 若使用者尚未訓練模型，系統顯示錯誤提示並引導回「模型訓練」。

## Artifacts（可選）
- `artifacts/samples.json`：批次生成的常見訊息集合。
- `artifacts/auto_tester_predictions.json`：批次推論結果（訊息、期望、預測、信心）。

