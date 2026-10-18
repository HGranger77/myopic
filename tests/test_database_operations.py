import pytest

from database import operations as ops


def test_create_watchlist_entity_auto_adds_canonical_name_as_alias():
    entity = ops.create_watchlist_entity("Joe Biden")
    assert entity["canonical_name"] == "Joe Biden"
    assert [a["normalized_alias"] for a in entity["aliases"]] == ["joe biden"]


def test_add_alias_and_get_active_watchlist_with_aliases():
    entity = ops.create_watchlist_entity("Joe Biden")
    ops.add_alias(entity["id"], "Biden")
    watchlist = ops.get_active_watchlist_with_aliases()
    assert len(watchlist) == 1
    # original casing, not normalized_alias - nlp.tagging's zero-shot
    # classifier is casing-sensitive (see its docstring: "AFL" vs "afl")
    assert sorted(watchlist[0]["aliases"]) == ["Biden", "Joe Biden"]


def test_inactive_entities_excluded_from_active_watchlist():
    entity = ops.create_watchlist_entity("Joe Biden")
    ops.update_watchlist_entity(entity["id"], is_active=False)
    assert ops.get_active_watchlist_with_aliases() == []


def test_delete_watchlist_entity_cascades_aliases():
    entity = ops.create_watchlist_entity("Joe Biden")
    assert ops.delete_watchlist_entity(entity["id"]) is True
    assert ops.get_watchlist_entity(entity["id"]) is None


def test_upsert_article_is_idempotent_on_url():
    run_id = ops.create_scrape_run()
    first_id, is_new_1 = ops.upsert_article("nine_com_au", "https://example.com/a", "Headline", "home", run_id)
    second_id, is_new_2 = ops.upsert_article("nine_com_au", "https://example.com/a", "Headline", "home", run_id)
    assert first_id == second_id
    assert is_new_1 is True
    assert is_new_2 is False


def test_watchlist_match_unique_per_article_entity_pair():
    entity = ops.create_watchlist_entity("Joe Biden")
    run_id = ops.create_scrape_run()
    article_id, _ = ops.upsert_article("nine_com_au", "https://example.com/a", "Headline", "home", run_id)
    ops.insert_watchlist_match(article_id, entity["id"], None, "biden", run_id)
    ops.insert_watchlist_match(article_id, entity["id"], None, "biden", run_id)
    assert ops.count_matched_articles(run_id) == 1


def test_translation_worklist_and_set_translation():
    run_id = ops.create_scrape_run()
    zh_id, _ = ops.upsert_article("chinanews", "https://example.com/zh", "测试标题", "home", run_id)
    ops.mark_article_fetched(zh_id, "测试正文内容。")
    en_id, _ = ops.upsert_article("nine_com_au", "https://example.com/en", "Headline", "home", run_id)
    ops.mark_article_fetched(en_id, "Body text.")

    # only articles from the requested source list come back - the English
    # article never shows up when only the Chinese source is asked for
    # (run_batch.py's translate_article() only ever passes non-English
    # modules here)
    assert [a["id"] for a in ops.get_articles_needing_translation(["chinanews"])] == [zh_id]
    assert ops.get_articles_needing_translation([]) == []

    ops.set_article_translation(zh_id, "Test Headline", "Test body content.")
    article = ops.get_article(zh_id)
    assert article["headline"] == "Test Headline"
    assert article["body_text"] == "Test body content."
    assert article["original_headline"] == "测试标题"
    assert article["original_body_text"] == "测试正文内容。"

    # translated articles drop out of their own worklist
    assert ops.get_articles_needing_translation(["chinanews"]) == []


def test_embedding_worklist_excludes_untranslated_non_english_articles():
    # Regression test: embed-article must never embed a non-English
    # article's RAW untranslated text - that happened for real (49
    # untranslated chinanews articles got embedded, clustered into one
    # garbage story, and summarized into nonsense). See
    # get_articles_needing_embedding()'s docstring.
    run_id = ops.create_scrape_run()
    zh_id, _ = ops.upsert_article("chinanews", "https://example.com/zh-embed", "测试标题", "home", run_id)
    ops.mark_article_fetched(zh_id, "测试正文内容。")
    en_id, _ = ops.upsert_article("nine_com_au", "https://example.com/en-embed", "Headline", "home", run_id)
    ops.mark_article_fetched(en_id, "Body text.")

    # English article is ready immediately; untranslated Chinese one waits
    ids = [a["id"] for a in ops.get_articles_needing_embedding(["chinanews"])]
    assert en_id in ids
    assert zh_id not in ids

    # once translated, it joins the worklist like any other fetched article
    ops.set_article_translation(zh_id, "Test Headline", "Test body content.")
    ids = [a["id"] for a in ops.get_articles_needing_embedding(["chinanews"])]
    assert zh_id in ids

    # no non_english_modules passed (default) - nothing is held back
    assert {a["id"] for a in ops.get_articles_needing_embedding()} >= {zh_id, en_id}


def test_entities_and_sentiment_worklists_are_gap_based_not_run_scoped():
    # Regression test: these worklists must pick up a relevant article no
    # matter how long ago its story membership was created - a relevance
    # backfill (reclassifying old stories after a watchlist change, with no
    # new story_articles row at all) must still surface it. The old
    # run-scoped version of this worklist silently skipped articles like
    # this forever - see get_relevant_articles_missing_entities()'s
    # docstring for the real incident this prevents.
    old_run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/old-run-article", old_run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, old_run_id)
    ops.set_article_relevance(article_id, False)
    assert ops.get_relevant_articles_missing_entities() == []
    assert ops.get_relevant_articles_missing_sentiment() == []

    # simulates a relevance backfill: is_relevant flips with no new run
    ops.set_article_relevance(article_id, True)
    assert ops.get_relevant_articles_missing_entities() == [{"article_id": article_id}]
    assert ops.get_relevant_articles_missing_sentiment() == [{"article_id": article_id, "story_id": story_id}]

    ops.insert_article_entities(article_id, [_entity("Someone", "PERSON", 0)])
    ops.upsert_article_sentiment(article_id, story_id, "POSITIVE", 0.9, "test-model")
    assert ops.get_relevant_articles_missing_entities() == []
    assert ops.get_relevant_articles_missing_sentiment() == []


def test_entities_worklist_does_not_reprocess_zero_entity_articles_forever():
    # A genuinely zero-entity article (very short text, or non-English text
    # an English NER model finds nothing recognizable in - this happened
    # for real with untranslated Chinese articles that slipped through
    # before the embed-article translation gate existed) must still drop
    # out of the worklist once checked. Keying off "no article_entities
    # rows exist" can't tell that apart from "never checked" and would
    # reprocess the same zero-entity article every run forever - see
    # insert_article_entities()'s docstring for the entities_checked_at fix.
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/zero-entities", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.set_article_relevance(article_id, True)
    assert ops.get_relevant_articles_missing_entities() == [{"article_id": article_id}]

    ops.insert_article_entities(article_id, [])  # ran NER, found nothing
    assert ops.get_relevant_articles_missing_entities() == []


def test_entity_sentiment_worklist_does_not_reprocess_zero_entity_articles_forever():
    # same fix as entities_checked_at (see the test above), for
    # entity_sentiment_checked_at - an article with zero significant
    # entities must still drop out of entity-sentiment's worklist.
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/zero-entity-sentiment", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.set_article_relevance(article_id, True)
    assert ops.get_relevant_articles_missing_entity_sentiment() == [{"article_id": article_id}]

    ops.insert_article_entity_sentiment(article_id, [], "test-absa-model")
    assert ops.get_relevant_articles_missing_entity_sentiment() == []


def test_entity_sentiment_insert_and_lookup_round_trip():
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/entity-sentiment-roundtrip", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.set_article_relevance(article_id, True)

    ops.insert_article_entity_sentiment(
        article_id,
        [
            {"entity_text": "Jane Smith", "entity_category": "people", "sentiment_label": "Positive", "sentiment_score": 0.91},
            {"entity_text": "Acme Corp", "entity_category": "organizations", "sentiment_label": "Negative", "sentiment_score": 0.88},
        ],
        "test-absa-model",
    )

    lookup = ops.get_article_entity_sentiment(article_id)
    assert lookup[("people", "jane smith")] == {"label": "Positive", "score": pytest.approx(0.91)}
    assert lookup[("organizations", "acme corp")] == {"label": "Negative", "score": pytest.approx(0.88)}
    assert ops.get_relevant_articles_missing_entity_sentiment() == []


def test_entity_sentiment_retry_does_not_duplicate_or_error_on_unique_constraint():
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/entity-sentiment-retry", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.set_article_relevance(article_id, True)

    entry = [{"entity_text": "Jane Smith", "entity_category": "people", "sentiment_label": "Positive", "sentiment_score": 0.9}]
    ops.insert_article_entity_sentiment(article_id, entry, "test-absa-model")
    # simulate a retried task: delete-then-reinsert, same as entity_sentiment() in run_batch.py
    ops.delete_article_entity_sentiment(article_id)
    ops.insert_article_entity_sentiment(article_id, entry, "test-absa-model")

    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM article_entity_sentiment WHERE article_id = %s", (article_id,)
            )
            assert cur.fetchone()["n"] == 1


def _seed_relevant_story(entity_id: int, url: str, summary: str, headline: str = "Headline") -> int:
    run_id = ops.create_scrape_run()
    article_id, _ = ops.upsert_article("nine_com_au", url, headline, "home", run_id)
    ops.mark_article_fetched(article_id, "body")
    ops.insert_watchlist_match(article_id, entity_id, None, "joe biden", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.upsert_story_summary(story_id, summary, "test-model")
    return story_id


def test_recent_stories_pagination_and_total_count():
    entity = ops.create_watchlist_entity("Joe Biden")
    for i in range(3):
        _seed_relevant_story(entity["id"], f"https://example.com/pg-{i}", f"Summary {i}")

    page1, total = ops.get_recent_stories_with_summaries(limit=2, offset=0)
    assert total == 3
    assert len(page1) == 2

    page2, total2 = ops.get_recent_stories_with_summaries(limit=2, offset=2)
    assert total2 == 3  # same total regardless of which page
    assert len(page2) == 1

    # together, both pages cover every story with no overlap
    page1_ids = {s["story_id"] for s in page1}
    page2_ids = {s["story_id"] for s in page2}
    assert page1_ids.isdisjoint(page2_ids)
    assert len(page1_ids | page2_ids) == 3


def test_recent_stories_date_filter_and_get_story_dates():
    entity = ops.create_watchlist_entity("Joe Biden")
    old_story = _seed_relevant_story(entity["id"], "https://example.com/date-old", "Old summary")
    new_story = _seed_relevant_story(entity["id"], "https://example.com/date-new", "New summary")

    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-01-01 10:00:00+00", old_story))
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-01-02 10:00:00+00", new_story))

    import datetime

    jan1_stories, jan1_total = ops.get_recent_stories_with_summaries(date="2026-01-01")
    assert jan1_total == 1
    assert jan1_stories[0]["story_id"] == old_story

    dates = {row["day"]: row["story_count"] for row in ops.get_story_dates()}
    assert dates[datetime.date(2026, 1, 1)] == 1
    assert dates[datetime.date(2026, 1, 2)] == 1


def test_recent_stories_search_matches_summary_or_headline():
    entity = ops.create_watchlist_entity("Joe Biden")
    summary_match = _seed_relevant_story(
        entity["id"], "https://example.com/search-summary", "A story about tariffs and trade.", headline="Unrelated"
    )
    headline_match = _seed_relevant_story(
        entity["id"], "https://example.com/search-headline", "Nothing special here.", headline="Tariffs hit record high"
    )
    _seed_relevant_story(entity["id"], "https://example.com/search-none", "Completely different topic.")

    results, total = ops.get_recent_stories_with_summaries(q="tariffs")
    assert total == 2
    assert {s["story_id"] for s in results} == {summary_match, headline_match}

    # case-insensitive
    results_upper, total_upper = ops.get_recent_stories_with_summaries(q="TARIFFS")
    assert total_upper == 2

    none_results, none_total = ops.get_recent_stories_with_summaries(q="nonexistentterm")
    assert none_total == 0
    assert none_results == []


def test_get_story_dates_respects_search_term():
    entity = ops.create_watchlist_entity("Joe Biden")
    match_story = _seed_relevant_story(entity["id"], "https://example.com/search-date-a", "Tariffs announced today.")
    other_story = _seed_relevant_story(entity["id"], "https://example.com/search-date-b", "Unrelated news.")

    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-02-01 10:00:00+00", match_story))
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-02-02 10:00:00+00", other_story))

    import datetime

    dates = {row["day"]: row["story_count"] for row in ops.get_story_dates(q="tariffs")}
    assert dates == {datetime.date(2026, 2, 1): 1}


def test_topic_story_counts_respect_date_and_search_filters():
    entity = ops.create_watchlist_entity("AFL")
    old_story = _seed_relevant_story(
        entity["id"], "https://example.com/topic-count-old", "Grand final preview tonight."
    )
    new_story = _seed_relevant_story(entity["id"], "https://example.com/topic-count-new", "Off-season trade news.")
    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-03-01 10:00:00+00", old_story))
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-03-02 10:00:00+00", new_story))

    unfiltered = {row["canonical_name"]: row["story_count"] for row in ops.get_topic_story_counts()}
    assert unfiltered["AFL"] == 2

    by_date = {row["canonical_name"]: row["story_count"] for row in ops.get_topic_story_counts(date="2026-03-01")}
    assert by_date["AFL"] == 1

    by_search = {row["canonical_name"]: row["story_count"] for row in ops.get_topic_story_counts(q="grand final")}
    assert by_search["AFL"] == 1

    # a topic with zero matches under the filter still appears, with count 0
    # - not silently dropped (the LEFT JOIN requirement the docstring calls out)
    ops.create_watchlist_entity("NRL")
    by_search_zero = {row["canonical_name"]: row["story_count"] for row in ops.get_topic_story_counts(q="grand final")}
    assert by_search_zero["NRL"] == 0


def test_translation_worklist_respects_limit():
    run_id = ops.create_scrape_run()
    ids = []
    for i in range(3):
        article_id, _ = ops.upsert_article(
            "chinanews", f"https://example.com/zh-{i}", f"标题{i}", "home", run_id
        )
        ops.mark_article_fetched(article_id, f"正文{i}。")
        ids.append(article_id)

    assert len(ops.get_articles_needing_translation(["chinanews"])) == 3
    # oldest (lowest id) first, capped - matches TRANSLATE_MAX_ARTICLES
    # pacing a backlog across multiple runs rather than all at once
    limited = ops.get_articles_needing_translation(["chinanews"], limit=2)
    assert [a["id"] for a in limited] == ids[:2]


def _make_fetched_article(url: str, run_id: int) -> int:
    article_id, _ = ops.upsert_article("nine_com_au", url, "Headline", "home", run_id)
    ops.mark_article_fetched(article_id, "body text")
    return article_id


def test_story_lifecycle_create_join_and_touched_ids():
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/story-a", run_id)

    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)

    assert ops.get_touched_story_ids(run_id) == [story_id]
    # get_relevant_articles_missing_entities/sentiment (article-entities' and
    # sentiment-article's worklists) and get_touched_relevant_story_ids
    # (finalize's relevant-count query) are all scoped to tag-relevance's
    # verdict, not just "touched" - empty until relevance is decided, same
    # as in the real DAG order.
    assert ops.get_relevant_articles_missing_entities() == []
    assert ops.get_relevant_articles_missing_sentiment() == []
    assert ops.get_touched_relevant_story_ids(run_id) == []

    ops.set_article_relevance(article_id, True)
    assert ops.get_relevant_articles_missing_entities() == [{"article_id": article_id}]
    assert ops.get_relevant_articles_missing_sentiment() == [{"article_id": article_id, "story_id": story_id}]
    assert ops.get_touched_relevant_story_ids(run_id) == [story_id]

    articles = ops.get_story_articles(story_id)
    assert [a["id"] for a in articles] == [article_id]


def test_story_summary_upsert_regenerates_not_appends():
    story_id = ops.create_story()
    ops.upsert_story_summary(story_id, "first draft", "test-model")
    ops.upsert_story_summary(story_id, "revised after a new source joined", "test-model")

    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n, summary_text FROM story_summaries WHERE story_id = %s GROUP BY summary_text", (story_id,))
            rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["summary_text"] == "revised after a new source joined"


def test_article_sentiment_is_one_shot_per_article():
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/story-b", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)

    ops.upsert_article_sentiment(article_id, story_id, "POSITIVE", 0.9, "test-model")
    ops.upsert_article_sentiment(article_id, story_id, "NEGATIVE", 0.8, "test-model")  # simulate a retry

    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM article_sentiment WHERE article_id = %s", (article_id,))
            assert cur.fetchone()["n"] == 1


def test_article_sentiment_stores_emotion_alongside_sentiment():
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/emotion-story", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)

    ops.upsert_article_sentiment(
        article_id, story_id, "NEGATIVE", 0.9, "test-sentiment-model",
        emotion_label="fear", emotion_score=0.76, emotion_model_name="test-emotion-model",
    )

    source_articles = ops.get_story_source_articles(story_id)
    assert len(source_articles) == 1
    assert source_articles[0]["sentiment_label"] == "NEGATIVE"
    assert source_articles[0]["emotion_label"] == "fear"
    assert source_articles[0]["emotion_score"] == pytest.approx(0.76)


def test_article_sentiment_emotion_fields_default_to_none():
    # existing callers (and old rows) that don't pass emotion data must not
    # break - the columns are nullable and optional
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/no-emotion-story", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)

    ops.upsert_article_sentiment(article_id, story_id, "POSITIVE", 0.9, "test-model")

    source_articles = ops.get_story_source_articles(story_id)
    assert source_articles[0]["emotion_label"] is None


def _entity(text, label, start):
    return {"text": text, "normalized_text": text.lower(), "label": label, "start_char": start, "end_char": start + len(text)}


def test_article_entities_categorized_deduped_and_ordered():
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/entities-a", run_id)

    ops.insert_article_entities(article_id, [
        _entity("Joe Biden", "PERSON", 0),
        _entity("Washington", "GPE", 20),
        _entity("NATO", "ORG", 40),
        _entity("Joe Biden", "PERSON", 60),  # same person mentioned again
        _entity("yesterday", "DATE", 80),  # not a shown category
    ])

    entities = ops.get_article_entities(article_id)
    assert entities["people"] == ["Joe Biden"]  # deduped
    assert entities["organizations"] == ["NATO"]
    assert entities["locations"] == ["Washington"]  # GPE mapped to locations


def test_article_entities_drops_late_one_off_mentions_but_keeps_lede_and_repeats():
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/entities-c", run_id)

    ops.insert_article_entities(article_id, [
        _entity("Jane Smith", "PERSON", 10),  # story subject, mentioned once in the lede - kept
        _entity("NATO", "ORG", 50),
        _entity("NATO", "ORG", 900),  # mentioned twice, one of them late - kept
        _entity("Staff Reporter", "PERSON", 950),  # byline-style: late and only mentioned once - dropped
    ])

    entities = ops.get_article_entities(article_id)
    assert entities["people"] == ["Jane Smith"]
    assert entities["organizations"] == ["NATO"]


def test_article_entities_retry_does_not_duplicate():
    run_id = ops.create_scrape_run()
    article_id = _make_fetched_article("https://example.com/entities-b", run_id)

    ops.insert_article_entities(article_id, [_entity("Joe Biden", "PERSON", 0)])
    # simulate a retried task: delete-then-reinsert, same as article_entities() in run_batch.py
    ops.delete_article_entities(article_id)
    ops.insert_article_entities(article_id, [_entity("Joe Biden", "PERSON", 0)])

    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM article_entities WHERE article_id = %s", (article_id,))
            assert cur.fetchone()["n"] == 1
