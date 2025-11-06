## Why
The project needs a baseline capability to classify SMS/email messages as spam or ham using the provided dataset. This enables evaluation, iterations, and future improvements.

## What Changes
- Add a new capability: Spam Classification (training + inference)
- Implement deterministic preprocessing (tokenization, TF-IDF)
- Train a Phase 1 baseline model using SVM (LinearSVC). If confidence scores are required, wrap with CalibratedClassifierCV for probability estimates. Save artifacts.
- Provide CLI/notebook entry points for training and inference
- Add evaluation metrics (accuracy, precision, recall, F1)

## Impact
- Affected specs: `spam-classification`
- Affected code: new module(s) under `spam_classification/` (or scripts/notebooks), uses `sms_spam_no_header.csv`
- No breaking changes; new capability added

## Notes
- Phase 2+ improvements will be planned later and are intentionally left as placeholders for now.
