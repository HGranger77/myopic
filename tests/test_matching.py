from nlp.matching import build_watchlist_index, filter_significant_entities, match_entities, normalize


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


def _entity(text, start, normalized=None):
    return {"text": text, "normalized_text": normalized or text.lower(), "start_char": start}


def test_significance_drops_late_one_off_mention():
    # simulates a byline name: mentioned once, far from the lede
    entities = [_entity("Jane Smith", 900)]
    assert filter_significant_entities(entities) == []


def test_significance_keeps_early_one_off_mention():
    # a short brief's actual subject, mentioned only once but in the lede
    entities = [_entity("Jane Smith", 10)]
    assert filter_significant_entities(entities) == ["Jane Smith"]


def test_significance_keeps_late_repeated_mention():
    entities = [_entity("Jane Smith", 50), _entity("Jane Smith", 900)]
    assert filter_significant_entities(entities) == ["Jane Smith"]


def test_significance_merges_name_variants_and_promotes_to_longest_form():
    entities = [_entity("Joe Biden", 10), _entity("Biden", 900)]
    # merged into one group (2 mentions total) - survives, and is reported
    # under the fuller "Joe Biden" form even though the second mention used
    # the shorter one
    assert filter_significant_entities(entities) == ["Joe Biden"]


def test_significance_does_not_merge_unrelated_entities():
    entities = [_entity("Joe Biden", 10), _entity("Kamala Harris", 900)]
    # neither is a whole-word substring of the other, so they stay separate -
    # "Joe Biden" survives on its early mention, "Kamala Harris" is dropped
    # as a late one-off
    assert filter_significant_entities(entities) == ["Joe Biden"]


def test_significance_orders_by_first_mention():
    entities = [_entity("Second", 5), _entity("Second", 6), _entity("First", 1), _entity("First", 2)]
    assert filter_significant_entities(entities) == ["First", "Second"]
