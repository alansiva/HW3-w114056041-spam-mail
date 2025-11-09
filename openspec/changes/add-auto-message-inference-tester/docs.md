# add-auto-message-inference-tester — 文件與 CLI 範例

本變更提供「訊息推論測試器」功能，涵蓋：
- 模組 `spam_classification/samples.py`：生成中/英 spam/ham 常見文本，支援批次生成 API 與 CLI。
- 單頁 UI：新增測試器區塊，控制語言、類別、數量與 spam 比例，並即時顯示期望/預測與信心分數。
- CLI：批次生成樣本與批次推論，輸出 JSON 以便離線檢查。

## CLI 範例

生成樣本：
```bash
python3 -m spam_classification.samples --lang zh --category mixed --n 20 --spam-ratio 0.5 --out artifacts/samples.json
```

批次推論：
```bash
python3 -m spam_classification.infer --input artifacts/samples.json --out artifacts/auto_tester_predictions.json
```

輸出格式說明：
- `samples.json`：`[{text, lang, category, expected_label}]`
- `auto_tester_predictions.json`：`[{text, predicted, confidence, expected_label?, match?}]`

## 使用建議
1. 先於 UI 的「訓練」區塊完成模型訓練，確保 `artifacts/model.joblib` 與 `artifacts/vectorizer.joblib` 存在。
2. 於 UI 使用測試器生成樣本並立即推論；或以 CLI 生成並批次推論，檢查 `match` 欄位。

