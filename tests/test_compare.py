import pytest

from src.compare import window
from src.evaluate import mcnemar_p, subset_macro_f1


def test_window_keeps_first_and_last_tokens():
    ids = list(range(10))
    assert window(ids, 4, 0) == [0, 1, 2, 3]
    assert window(ids, 0, 3) == [7, 8, 9]
    assert window(ids, 2, 3) == [0, 1, 7, 8, 9]
    assert window(ids, 6, 4) == ids  # short enough: kept whole


def test_subset_macro_f1_ignores_classes_missing_from_the_subset():
    # "neuro" never appears in y_true: it must not count as F1 = 0
    assert subset_macro_f1(["cardio", "ortho"], ["cardio", "ortho"]) == 1.0
    # but a wrong prediction of a missing class still costs recall on the true class
    assert subset_macro_f1(["cardio", "ortho"], ["cardio", "neuro"]) == pytest.approx((1.0 + 0.0) / 2)


def test_mcnemar():
    y = ["a"] * 10
    # A right on all 10, B right on none: 10 vs 0 -> p = 2 / 2**10
    r = mcnemar_p(y, ["a"] * 10, ["b"] * 10)
    assert (r["a_only_right"], r["b_only_right"]) == (10, 0)
    assert r["p_value"] == round(2 / 2**10, 4)
    # identical models: no discordant reports, nothing to test
    assert mcnemar_p(y, ["a"] * 10, ["a"] * 10)["p_value"] == 1.0
    # 6 vs 4 (the main BERT vs baseline case): p = 0.7539
    assert mcnemar_p(y, ["a"] * 6 + ["b"] * 4, ["b"] * 6 + ["a"] * 4)["p_value"] == 0.7539
