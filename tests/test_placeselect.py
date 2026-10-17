"""geolocate-story's place choice (nlp/placeselect.py): every step except the
NLI model call is plain logic, tested directly; the model call is stubbed."""
from database import operations as ops
from nlp import placeselect


def _cand(text, sig=1, geo_name=None, mentions=1, person=0):
    return {"key": placeselect.place_key(text), "text": text, "significant_articles": sig,
            "mentions": mentions, "person_mentions": person,
            "geo": {"display_name": geo_name or text, "lat": 1.0, "lon": 2.0, "place_rank": 16, "radius_km": 5}}


def test_screen_drops_text_that_is_mostly_a_person_name():
    kabila = _cand("Kabila", mentions=1, person=7)
    sydney = _cand("Sydney", mentions=3, person=0)
    assert placeselect.screen_candidates([kabila, sydney]) == [sydney]


def test_choose_takes_highest_nli_score_not_most_mentioned():
    yemen, riyadh = _cand("Yemen", sig=3), _cand("Riyadh", sig=1)
    chosen, score = placeselect.choose_place([yemen, riyadh], {"Yemen": 0.5, "Riyadh": 0.9})
    assert chosen["text"] == "Riyadh" and score == 0.9


def test_choose_steps_down_to_a_place_inside_the_winner():
    australia = _cand("Australia")
    sydney = _cand("Sydney", geo_name="Sydney, New South Wales, Australia")
    chosen, _ = placeselect.choose_place([australia, sydney], {"Australia": 0.8, "Sydney": 0.7})
    assert chosen["text"] == "Sydney"
    # ...but not when the inner place is much weaker
    chosen, _ = placeselect.choose_place([australia, sydney], {"Australia": 0.9, "Sydney": 0.3})
    assert chosen["text"] == "Australia"


def test_choose_gives_no_pin_below_min_confidence():
    assert placeselect.choose_place([_cand("Perth")], {"Perth": 0.2}) is None


def test_story_place_candidates_counts_mentions_and_person_names():
    entity = ops.create_watchlist_entity("politics")
    run_id = ops.create_scrape_run()
    article_id, _ = ops.upsert_article("src", "https://x.test/a", "Kabila visits Goma", "w", run_id)
    ops.mark_article_fetched(article_id, "Joseph Kabila visited Goma. Kabila said Goma is calm.")
    ops.insert_watchlist_match(article_id, entity["id"], None, "politics", run_id)
    ops.set_article_relevance(article_id, True)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ent = lambda t, l, p: {"text": t, "normalized_text": t.lower(), "label": l, "start_char": p, "end_char": p + len(t)}
    ops.insert_article_entities(article_id, [
        ent("Joseph Kabila", "PERSON", 0), ent("Goma", "GPE", 22), ent("Kabila", "GPE", 28), ent("Goma", "GPE", 44),
    ])
    by_key = {c["key"]: c for c in ops.get_story_place_candidates(story_id)}
    assert by_key["goma"]["mentions"] == 2 and by_key["goma"]["headline_hits"] == 1
    assert by_key["kabila"]["person_mentions"] >= by_key["kabila"]["mentions"]  # screened out later


def test_geolocate_stage_runs_the_five_steps_and_stamps_the_story(monkeypatch):
    import nlp.geocode
    import run_batch
    from tests.test_stories_page import _seed_matched_story

    story_id, article_id = _seed_matched_story()
    ops.set_article_relevance(article_id, True)
    ops.insert_article_entities(article_id, [
        {"text": "Washington", "normalized_text": "washington", "label": "GPE", "start_char": 0, "end_char": 10},
    ])
    calls = []
    monkeypatch.setattr(nlp.geocode, "geocode", lambda p: calls.append(p) or
                        {"lat": 1.5, "lon": 2.5, "display_name": "Washington, DC", "place_rank": 16, "radius_km": 5})
    monkeypatch.setattr(placeselect, "nli_scores", lambda premise, places: {p: 0.9 for p in places})
    assert run_batch.geolocate_story(1) == 0
    assert calls == ["washington"]
    assert ops.get_cached_geocode("washington")["lat"] == 1.5
    (story,) = ops.get_map_stories()
    assert story["location_text"] == "Washington"
    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT confidence FROM story_locations WHERE story_id = %s", (story_id,))
            assert abs(cur.fetchone()["confidence"] - 0.9) < 1e-6
    assert ops.get_relevant_stories_missing_location() == []
