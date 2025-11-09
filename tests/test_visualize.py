from pathlib import Path

from spam_classification.visualize import (
    run_visualize_cli,
    top_tokens_by_class,
    top_tokens_by_class_from_data,
)
from spam_classification.infer import load_pipeline
from spam_classification.data import load_dataset, split_dataset


ARTIFACTS_DIR = Path("artifacts")
DATA_PATH = Path("sms_spam_no_header.csv")


def test_run_visualize_cli_outputs_exist(tmp_path):
    out_dir = tmp_path / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_visualize_cli(
        csv_path=str(DATA_PATH),
        artifacts_dir=str(ARTIFACTS_DIR),
        out_dir=str(out_dir),
        test_size=0.2,
        seed=42,
        top_n=10,
    )
    # check images
    assert (out_dir / "roc_curve.png").exists()
    assert (out_dir / "pr_curve.png").exists()
    assert (out_dir / "confusion_matrix.png").exists()
    # check csvs
    assert (out_dir / "top_tokens_ham.csv").exists()
    assert (out_dir / "top_tokens_spam.csv").exists()


def test_top_tokens_by_class_basic():
    pipeline = load_pipeline(str(ARTIFACTS_DIR))
    df_ham, df_spam = top_tokens_by_class(pipeline, top_n=5)
    if df_ham is None or df_spam is None:
        df = load_dataset(str(DATA_PATH))
        X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2, random_state=42)
        df_ham, df_spam = top_tokens_by_class_from_data(pipeline, X_train, y_train, top_n=5)
    assert df_ham is not None and df_spam is not None
    assert len(df_ham) == 5
    assert len(df_spam) == 5
    for df in (df_ham, df_spam):
        assert set(["token"]).issubset(df.columns)
