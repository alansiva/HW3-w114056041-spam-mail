# 任務拆解：自動產生訊息推論測試器

## Phase A — 文件與規格
- [ ] 撰寫提案（proposal.md）與需求規格（specs/spam-classification/spec.md）
- [ ] 對齊既有 OpenSpec（單頁 UI、可視化與關鍵字排名）

## Phase B — 核心模組
- [ ] 新增 `spam_classification/samples.py`：
  - `generate_message(lang, category)`：產生單則訊息與期望標籤
  - `generate_batch(n, lang, category, spam_ratio)`：批次產生，支援混合比例
  - 安全占位符：`{url}`, `{phone}`, `{code}`, `{time}`

## Phase C — Streamlit UI 擴充（單頁）
- [ ] 在 app/main.py 加入區塊「訊息推論測試器（自動產生常見郵件隨機文本）」
- [ ] 控制項：語言、類別、生成數量、混合比例、生成與推論按鈕
- [ ] 顯示每則：期望標籤、預測標籤、信心分數；不一致者顯示提示

## Phase D — CLI（後續）
- [ ] `python -m spam_classification.samples --lang zh --category mixed --n 20 --spam-ratio 0.5 --out artifacts/samples.json`
- [ ] `python -m spam_classification.infer --input artifacts/samples.json --out artifacts/auto_tester_predictions.json`
- [ ] 更新 README 與 OpenSpec 對應的 CLI 使用範例

## Phase E — 測試與驗收
- [ ] 單元測試：生成器輸出符合預期（語言、類別、比例）
- [ ] 整合測試：UI 呼叫生成與推論、正確展示結果
- [ ] 文件驗收：OpenSpec 與 README 完成並一致

