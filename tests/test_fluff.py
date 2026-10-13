from nlp.fluff import FLUFF_LABELS, SUBSTANTIVE_LABELS, is_fluff_from_scores


def test_fluff_when_fluff_label_dominates():
    assert is_fluff_from_scores({FLUFF_LABELS[0]: 0.9, SUBSTANTIVE_LABELS[0]: 0.1})


def test_not_fluff_when_substantive_beats_fluff():
    assert not is_fluff_from_scores({FLUFF_LABELS[0]: 0.7, SUBSTANTIVE_LABELS[0]: 0.95})


def test_not_fluff_below_threshold():
    assert not is_fluff_from_scores({FLUFF_LABELS[0]: 0.4, SUBSTANTIVE_LABELS[0]: 0.1})
