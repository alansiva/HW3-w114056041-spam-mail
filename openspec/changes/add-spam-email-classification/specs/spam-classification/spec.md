## ADDED Requirements

### Requirement: Spam Email Classification Capability
The system SHALL provide a baseline capability to classify SMS/email messages as `spam` or `ham` using the `sms_spam_no_header.csv` dataset.

- The system SHALL load the dataset and perform a deterministic train/test split with a fixed random seed.
- The system SHALL preprocess text via tokenization and TF-IDF vectorization.
- The system SHALL train a Phase 1 baseline classifier using SVM (LinearSVC). If confidence scores are required, the classifier SHOULD be calibrated (e.g., via `CalibratedClassifierCV`) to provide probability estimates.
- The system SHALL produce evaluation metrics (accuracy, precision, recall, F1) on the test set.
- The system SHALL persist model and vectorizer artifacts for later inference.
- The system SHOULD provide a simple CLI or notebook interface for training and inference.

#### Scenario: Train baseline model and evaluate metrics
- WHEN the training command is executed against `sms_spam_no_header.csv`
- THEN the system trains the model, evaluates on the test set, and writes metrics to `artifacts/metrics.json`
- AND model artifacts are saved under `artifacts/` (e.g., `model.pkl`, `vectorizer.pkl`)

#### Scenario: Infer single message returns label and confidence
- WHEN a user runs inference with a single message string
- THEN the system returns a label (`spam` or `ham`) and a confidence score
- AND uses the persisted model/vectorizer artifacts
