# 專案背景

## 目的
建置一個端到端的簡訊/電子郵件垃圾訊息分類能力（HW3 作業），涵蓋資料讀取、前處理、模型訓練、評估，以及最小化的 CLI/Notebook 推論使用。近期目標是以提供的資料集完成第一階段可重現的基準（使用 SVM 的 LinearSVC），並預留第二階段及之後的迭代空間（目前僅作為占位）。

## 技術棧
- Python 3.10+（建議）
- 數據與機器學習：pandas、scikit-learn、numpy
- 模型：LinearSVC（SVM）作為基準；後續階段可評估邏輯迴歸
- 產物持久化：使用 joblib 儲存/載入模型與向量器（artifacts）
- 實驗：Jupyter Notebook（可選）、CLI 腳本
- 工具：OpenSpec（規格驅動變更）、ripgrep（rg）全文檢索

## 專案慣例

### 程式碼風格
- Python：遵循 PEP 8
- 格式化：使用 black；匯入排序可選擇 isort
- 命名規則：函式/變數使用 snake_case；類別使用 CapWords

### 架構模式
- 以簡單單一模組為主：資料載入、前處理管線、模型訓練/評估各一
- 使用 scikit-learn Pipeline 保證前處理與訓練的可重現性
- 基準管線：TF-IDF 向量化 → LinearSVC；若需要信心分數，可使用 CalibratedClassifierCV 進行機率校準
- 保持以能力為中心的模組（例如 `spam_classification/`），避免過度工程化

### 測試策略
- 前處理函式（例：斷詞、向量化）的基本單元測試
- 固定隨機種子，使用可重現的訓練/測試切分
- 評估指標：accuracy、precision、recall、F1；可視需要檢視混淆矩陣
- 將指標以 JSON 儲存於 `artifacts/metrics.json`

### Git 工作流程
- 分支：每個 OpenSpec 變更以 change-id 建立對應的 feature 分支（例如 `feature/add-spam-email-classification`）
- 提交訊息：採用 Conventional Commits（例如 `feat: add spam classification baseline`）
- PR：引用 change-id 與規格檔；要求通過驗證與測試

## 領域背景
- 資料集：`sms_spam_no_header.csv`（兩欄：label、message；無表頭）
- 標籤：`spam` 與 `ham`（二元分類）
- 第一階段基準：SVM（LinearSVC）搭配 TF-IDF；後續可比較邏輯迴歸

## 重要限制
- 實作保持簡潔（單檔盡量 <100 行）
- 可重現性：固定種子、明確記錄前處理步驟
- 不需外部付費服務
- 優先使用本地訓練（資料集規模小）

## 數據結構
- `sms_spam_no_header.csv` 無表頭，每列兩個欄位：
  1. `label`（字串：`spam` 或 `ham`）
  2. `message`（字串：簡訊/郵件內容）

## 目錄結構
- `spam_classification/`
  - `data.py`（載入 CSV，使用固定 random_state 進行分層訓練/測試切分）
  - `preprocess.py`（文字清理、TF-IDF 向量化）
  - `train.py`（建立 Pipeline，訓練 LinearSVC 或校準分類器，並持久化產物）
  - `infer.py`（載入產物，針對單一訊息進行分類，回傳標籤與信心分數）
- `artifacts/`（保存模型/向量器與 `metrics.json`）
- `notebooks/`（選擇性探索）

## 可重現性
- 固定隨機種子：例如訓練/測試切分與任何隨機元件皆採用 `random_state=42`
- 透過 scikit-learn Pipeline 確保前處理與訓練步驟的確定性
- 在 README/specs 中記錄參數與步驟；可選擇於 `requirements.txt` 固定版本

## 外部相依
- 除了上述標準 Python 套件外不需要其他依賴
- 可選：pre-commit hooks（black/isort）、Jupyter 進行探索
