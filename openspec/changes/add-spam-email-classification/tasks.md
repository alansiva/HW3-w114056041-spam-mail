# 分階段實作計畫

## 第一階段：基準能力
- [ ] 1.1 建立 `spam_classification/` 模組結構：`data.py`、`preprocess.py`、`train.py`、`infer.py`
- [ ] 1.2 實作 `sms_spam_no_header.csv` 的資料載入，並以固定隨機種子進行分層訓練/測試切分
- [ ] 1.3 實作前處理管線（斷詞＋TF-IDF）
- [ ] 1.4 以 SVM（LinearSVC）作為基準模型；若需要信心分數，使用 `CalibratedClassifierCV` 取得機率估計
- [ ] 1.5 訓練並將產物（模型＋向量器）保存至 `artifacts/`
- [ ] 1.6 新增評估指標（accuracy、precision、recall、F1），並保存為 `artifacts/metrics.json`

## 第二階段：推論與介面
- [ ] 2.1 提供 CLI 入口：
  - 訓練：`python -m spam_classification.train --data sms_spam_no_header.csv --out artifacts/`
  - 單訊息推論：`python -m spam_classification.infer "訊息內容" --artifacts artifacts/`
  - 支援選項：`--calibrated`（是否使用 CalibratedClassifierCV）、`--seed`、`--test-size`、`--max-features`
- [ ] 2.2 單訊息推論返回標籤與信心分數：
  - 使用已持久化的模型與向量器
  - 若啟用校準，提供機率型信心分數；否則以決策函數或距離分數近似
- [ ] 2.3 可視化輸出（保存於 artifacts/ 並於 CLI 顯示）：
  - 生成混淆矩陣圖（`artifacts/confusion_matrix.png`）
  - 生成分類報告（precision/recall/F1/accuracy，`artifacts/classification_report.txt`）
  - 顯示前處理步驟摘要與示例（斷詞、TF-IDF 前 20 特徵）
- [ ] 2.4 建立 Streamlit 應用（`app/`）：
  - `app/main.py`：分頁包含「訓練」、「推論」、「指標/視覺化」
  - 上傳或選擇資料集（預設使用 `sms_spam_no_header.csv`）
  - 執行訓練並即時顯示指標與圖表（混淆矩陣、Top 特徵）
  - 單訊息推論互動視圖，顯示標籤與信心分數
- [ ] 2.5 文件與快速操作：
  - README 增補 CLI 使用範例與 Streamlit 啟動方式：`streamlit run app/main.py`
  - 說明 artifacts/ 的內容與如何重複使用
- [ ] 2.6 依賴與環境：
  - 更新/新增 `requirements.txt`（包含 streamlit、scikit-learn、pandas、numpy、joblib 等）
  - 說明環境設定與版本建議（Python 3.10+）
- [ ] 2.7 `app/` 目錄結構骨架：`main.py`、`pages/`（可選）、`components/`（可選）、`assets/`（圖檔）、`__init__.py`

### 合理性說明
- 以 Packt 程式碼庫第三章的垃圾郵件模式與資料集為基礎，擴展前處理與視覺化有助於學術性與可解釋性展示。
- CLI 提供最小可用介面，便於自動化與重現；Streamlit 提供互動展示，適合作業/報告呈現。
- 混淆矩陣與分類報告為標準監測指標，能有效比較不同模型或參數設定。

## 第三階段：品質與測試
- [ ] 3.1 單元測試：
  - 前處理函式（斷詞、TF-IDF）輸入/輸出一致性
  - 缺失值/空字串處理（應能安全跳過或給預設值）
- [ ] 3.2 統合測試（CLI）：
  - 訓練流程可在固定種子下產生 artifacts 與 metrics
  - 推論流程在 artifacts 存在/不存在時皆有明確錯誤/提示
- [ ] 3.3 UI 測試（Streamlit）：
  - 手動 QA 清單：能載入資料、執行訓練、顯示指標、進行單訊息推論
  - 截圖或錄影保存於 `artifacts/ui/`（可選）
- [ ] 3.4 程式碼品質：整合 black/isort/flake8（或 ruff），加上 pre-commit（可選）
- [ ] 3.5 效能檢查：
  - 小型資料集訓練時間與推論延遲評估
  - 記錄硬體/環境資訊以便重現
- [ ] 3.6 安全性與穩健性：
  - 輸入清理與長度限制（避免超長訊息造成效能問題）
  - 例外處理與錯誤訊息一致性
- [ ] 3.7 文件更新：README 增補安裝、使用、測試與視覺化說明

### 合理性說明
- 測試確保前處理與模型管線的穩定性，降低迭代風險。
- 程式碼品質與 pre-commit 可提升一致性與可維護性，符合課程/作業長期維護需求。

## 第四階段：驗證與核准 / 部署展示
- [ ] 4.1 規格驗證：`openspec validate add-spam-email-classification --strict`，修正所有問題
- [ ] 4.2 核准流程：請求審查並在核准後進行最終整理
- [ ] 4.3 部署到 Streamlit（本地或雲端）：
  - 本地啟動：`streamlit run app/main.py`
  - 若使用雲端（可選）：準備 `requirements.txt` 與 `app/`，上傳至 Streamlit Community Cloud
- [ ] 4.4 發佈與標記：更新 README、標記發佈版本（可選），記錄評估結果
- [ ] 4.5 事後驗證：手動檢查 UI、指標、推論輸出與 artifacts 完整性

### 合理性說明
- 嚴格驗證與審查能保證提案與實作一致，降低偏差與遺漏。
- Streamlit 部署提供可視化互動展示，符合作業呈現與評分需求。

## 備註
- 實作盡量簡潔（單檔 <100 行）
- 所有備註使用中文註記
- 以固定種子與可重現的前處理確保結果一致
- 優先本地訓練，無需外部付費服務
- 
