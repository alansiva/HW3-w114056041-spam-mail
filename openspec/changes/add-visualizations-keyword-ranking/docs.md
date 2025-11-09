# add-visualizations-keyword-ranking — 文件與方法說明

本變更新增視覺化與關鍵字排名，包含：
- 單頁 UI：ROC/PR、混淆矩陣、Top‑N tokens、決策閾值 slider。
- CLI：`spam_classification.visualize` 生成離線圖檔（PNG）與 CSV。

## 排名方法（Top‑N tokens）
1) 係數可用（LinearSVC 等）：
   - 使用分類器的權重 `coef_` 作為 token 重要度。
   - spam：選取權重最高的 Top‑N；ham：選取權重最低的 Top‑N。
2) 係數不可用（無 `coef_`）：
   - 以 TF‑IDF 向量器對訓練文本進行轉換，分別計算 ham/spam 的平均 TF‑IDF。
   - 對各類別取平均值最高的 Top‑N 作為代表性 tokens。

## ROC/PR 分數來源與未校準情境
- 優先使用 `predict_proba` 的 spam 機率作為正類分數。
- 若無法取得機率（例如使用某些 SVM 變體），則使用 `decision_function` 的 margin，並以 logistic 函式映射到 [0,1]。
  - 該近似分數僅用於曲線繪製與視覺化，不代表校準機率。

## CLI 範例
```bash
python3 -m spam_classification.visualize \
  --data sms_spam_no_header.csv \
  --artifacts artifacts \
  --out artifacts \
  --test-size 0.2 \
  --seed 42 \
  --top-n 20
```

將生成：
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`
- `top_tokens_ham.csv`, `top_tokens_spam.csv`
- `visualize_meta.json`

