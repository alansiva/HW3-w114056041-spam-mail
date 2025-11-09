# 任務拆解：自動產生訊息推論測試器

## Phase A — 文件與規格
- [x] 撰寫提案（proposal.md）與需求規格（specs/spam-classification/spec.md）
- [x] 對齊既有 OpenSpec（單頁 UI、可視覺化與關鍵字排名）

## Phase B — 核心模組
- [x] 新增 `spam_classification/samples.py`：
  - `generate_message(lang, category)`：產生單則訊息與期望標籤
  - `generate_batch(n, lang, category, spam_ratio)`：批次產生，支援混合比例
  - 安全占位符：`{url}`, `{phone}`, `{code}`, `{time}`

## Phase C — Streamlit UI 擴充（單頁）
- [x] 在 app/main.py 加入區塊「訊息推論測試器（自動產生常見文本）」
- [x] 控制項：語言、類別、生成數量、混合比例、生成與推論按鈕
- [x] 顯示每則：期望標籤、預測標籤、信心分數；不一致者顯示提示

## Phase D — CLI（後續）
- [x] `python -m spam_classification.samples --lang zh --category mixed --n 20 --spam-ratio 0.5 --out artifacts/samples.json`
- [x] `python -m spam_classification.infer --input artifacts/samples.json --out artifacts/auto_tester_predictions.json`
- [x] 更新 README 與 OpenSpec 對應的 CLI 使用範例

## Phase E — 測試與驗收
- [x] 單元測試：生成器輸出符合預期（語言、類別、比例）
- [x] 整合測試：功能層級與 CLI 驗證，並已人工檢視 UI 預覽
- [x] 文件驗收：OpenSpec 與 README 完成並一致
