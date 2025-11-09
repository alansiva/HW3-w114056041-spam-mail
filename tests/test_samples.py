import random
from spam_classification.samples import generate_batch


def test_generate_batch_spam_all():
    random.seed(42)
    batch = generate_batch(n=5, lang="zh", category="spam")
    assert len(batch) == 5
    assert all(item["expected_label"] == "spam" for item in batch)


def test_generate_batch_mixed_ratio_bounds():
    random.seed(0)
    batch = generate_batch(n=10, lang="en", category="mixed", spam_ratio=0.7)
    assert len(batch) == 10
    labels = [b["expected_label"] for b in batch]
    assert set(labels).issubset({"spam", "ham"})

