from nlp.matching import build_watchlist_index, match_entities, normalize


def test_normalize_lowercases_strips_punctuation_and_collapses_whitespace():
    assert normalize("  Joe  Biden's ") == "joe bidens"


def test_alias_matches_shorter_extracted_entity():
    watchlist = [{"watchlist_entity_id": 1, "canonical_name": "Joe Biden", "aliases": ["biden"]}]
    index = build_watchlist_index(watchlist)
    matches = match_entities([normalize("Biden")], index)
    assert matches == [("biden", 1, "biden")]


def test_multiple_aliases_of_same_entity_both_match_but_point_to_one_entity():
    watchlist = [{"watchlist_entity_id": 1, "canonical_name": "Joe Biden", "aliases": ["biden", "joe biden"]}]
    index = build_watchlist_index(watchlist)
    matches = match_entities([normalize("Biden")], index)
    assert {watchlist_entity_id for _, watchlist_entity_id, _ in matches} == {1}


def test_alias_matches_longer_extracted_entity():
    watchlist = [{"watchlist_entity_id": 1, "canonical_name": "Joe Biden", "aliases": ["biden"]}]
    index = build_watchlist_index(watchlist)
    matches = match_entities([normalize("Joe Biden")], index)
    assert matches == [("joe biden", 1, "biden")]


def test_no_match_for_unrelated_entity():
    watchlist = [{"watchlist_entity_id": 1, "canonical_name": "Joe Biden", "aliases": ["biden"]}]
    index = build_watchlist_index(watchlist)
    assert match_entities([normalize("Sydney")], index) == []


def test_substring_match_requires_whole_word_boundary():
    watchlist = [{"watchlist_entity_id": 1, "canonical_name": "Aidan", "aliases": ["aidan"]}]
    index = build_watchlist_index(watchlist)
    assert match_entities([normalize("Aidan Smith")], index) == [("aidan smith", 1, "aidan")]
    assert match_entities([normalize("Zaidani")], index) == []
