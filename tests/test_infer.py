from spam_classification.infer import infer_single


def test_infer_single_basic_ham():
    text = "晚餐想吃什麼？我七點到。"
    label, confidence = infer_single(text)
    assert label in {"spam", "ham"}
    assert 0.0 <= confidence <= 1.0


def test_infer_single_basic_spam_like():
    text = "Congratulations! You have won a prize. Click http://example.com to claim now."
    label, confidence = infer_single(text)
    assert label in {"spam", "ham"}
    assert 0.0 <= confidence <= 1.0
