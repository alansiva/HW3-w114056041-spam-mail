## 為什麼
本專案需要一個基準能力，使用提供的資料集將簡訊/電子郵件訊息分類為垃圾（spam）或非垃圾（ham），以便進行評估、迭代與後續改進。

## 變更內容
- 新增能力：垃圾訊息分類（訓練＋推論）
- 實作可重現的前處理（斷詞、TF-IDF）
- 第一階段以 SVM（LinearSVC）作為基準模型；若需要信心分數，使用 CalibratedClassifierCV 進行機率校準。產物會被保存。
- 提供訓練與推論的 CLI/Notebook 入口
- 新增評估指標（accuracy、precision、recall、F1）

## 影響
- 受影響的規格：`spam-classification`
- 受影響的程式：於 `spam_classification/`（或腳本/Notebook）新增模組，使用 `sms_spam_no_header.csv`
- 無破壞性變更；屬於新增能力

## 備註
- 第二階段之後的改進將在後續規劃，目前刻意保留為占位。
