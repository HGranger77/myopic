from nlp.tagging import _classification_label


def test_classification_label_picks_longest_alias():
    entry = {"canonical_name": "AI", "aliases": ["ai", "artificial intelligence"]}
    assert _classification_label(entry) == "artificial intelligence"


def test_classification_label_falls_back_to_canonical_name_when_no_aliases():
    entry = {"canonical_name": "AI", "aliases": []}
    assert _classification_label(entry) == "AI"


def test_classification_label_single_alias():
    entry = {"canonical_name": "Parramatta Eels", "aliases": ["parramatta eels"]}
    assert _classification_label(entry) == "parramatta eels"
