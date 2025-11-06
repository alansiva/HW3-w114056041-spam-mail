# Phased Implementation Plan

## Phase 1: Baseline Capability
- [ ] 1.1 Create `spam_classification/` module with structure: `data.py`, `preprocess.py`, `train.py`, `infer.py`
- [ ] 1.2 Implement data loader for `sms_spam_no_header.csv` with stratified train/test split (fixed random seed)
- [ ] 1.3 Implement preprocessing pipeline (tokenization + TF-IDF)
- [ ] 1.4 Implement baseline model using SVM (LinearSVC). If confidence scores are required, use `CalibratedClassifierCV` to enable probability estimates.
- [ ] 1.5 Train and persist artifacts (model + vectorizer) to `artifacts/`
- [ ] 1.6 Add evaluation metrics (accuracy, precision, recall, F1) and save to JSON at `artifacts/metrics.json`

## Phase 2: Inference & Interfaces (Placeholder)
- [ ] TBD — Phase 2 tasks will be added later (intentionally left empty now)

## Phase 3: Quality & Testing (Placeholder)
- [ ] TBD — Phase 3 tasks will be added later (intentionally left empty now)

## Phase 4: Validation & Approval (Placeholder)
- [ ] TBD — Validation and approval steps will be added later (intentionally left empty now)

## Notes
- Keep implementation simple (<100 lines per file where reasonable)
- Ensure reproducibility with fixed seeds and documented preprocessing steps
- Prefer local training; no external paid services required
