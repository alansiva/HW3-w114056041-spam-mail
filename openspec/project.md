# Project Context

## Purpose
Build an end-to-end SMS/email spam classification capability for the HW3 assignment, including data ingestion, preprocessing, model training, evaluation, and minimal CLI/notebook usage for inference. The immediate goal is to deliver a reproducible Phase 1 baseline using SVM (LinearSVC) on the provided dataset, and leave room for iterative Phase 2+ improvements (placeholders for now).

## Tech Stack
- Python 3.10+ (recommended)
- Data & ML: pandas, scikit-learn, numpy
- Models: LinearSVC (SVM) baseline; Logistic Regression may be evaluated in later phases
- Persistence: joblib for saving/loading artifacts (model, vectorizer)
- Experimentation: Jupyter Notebook (optional), CLI scripts
- Tooling: OpenSpec for spec-driven changes, ripgrep (rg) for searches

## Project Conventions

### Code Style
- Python: PEP 8 style guide
- Formatting: black; imports sorted with isort (optional)
- Naming: snake_case for functions/variables; CapWords for classes

### Architecture Patterns
- Simple, single-module approach to start: one data loader, one preprocessing pipeline, one model trainer/evaluator
- Use scikit-learn Pipeline for deterministic preprocessing + model training
- Baseline pipeline: TF-IDF vectorizer → LinearSVC; optionally wrap with CalibratedClassifierCV for probability estimates used as confidence scores
- Keep capability-focused modules (e.g., `spam_classification/`) and avoid over-engineering until required

### Testing Strategy
- Unit tests for preprocessing functions (e.g., tokenization, vectorization)
- Deterministic train/test split with fixed random seed
- Evaluation using accuracy, precision, recall, F1; confusion matrix for inspection
- Store metrics in a simple JSON file under `artifacts/`

### Git Workflow
- Branching: feature branches per OpenSpec change-id (e.g., `feature/add-spam-email-classification`)
- Commits: Conventional Commits (e.g., `feat: add spam classification baseline`)
- PRs: reference change-id and spec files; require passing validation and tests

## Domain Context
- Dataset: `sms_spam_no_header.csv` (two columns: label, message; no header)
- Labels: `spam` vs `ham` (binary classification)
- Phase 1 Baseline: SVM (LinearSVC) with TF-IDF features; Logistic Regression may be compared in later phases

## Important Constraints
- Keep implementation simple (<100 lines per file where reasonable)
- Reproducibility: fixed seeds, documented preprocessing steps
- No external paid services required
- Prefer local training (small dataset)

## Data Schema
- CSV file `sms_spam_no_header.csv` has no header and two fields per row:
  1. `label` (string: `spam` or `ham`)
  2. `message` (string: SMS/email text)

## Directory Layout
- `spam_classification/`
  - `data.py` (load CSV, stratified train/test split with fixed random_state)
  - `preprocess.py` (text cleaning, TF-IDF vectorizer)
  - `train.py` (build Pipeline, train LinearSVC or calibrated classifier, persist artifacts)
  - `infer.py` (load artifacts, classify single message, return label + confidence)
- `artifacts/` (persisted model/vectorizer and `metrics.json`)
- `notebooks/` (optional exploration)

## Reproducibility
- Fix random seeds: `random_state=42` for train/test split and any stochastic components
- Deterministic preprocessing via scikit-learn Pipeline
- Document exact steps and parameters in README/specs; keep versions pinned via `requirements.txt` (optional)

## External Dependencies
- None required beyond standard Python libraries listed above.
- Optional: pre-commit hooks for black/isort; Jupyter for exploration.
