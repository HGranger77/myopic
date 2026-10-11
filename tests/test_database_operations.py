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
    assert sorted(watchlist[0]["aliases"]) == ["biden", "joe biden"]


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
    # get_new_story_memberships_for_run (sentiment-article/article-entities'
    # worklist) and get_touched_relevant_story_ids (finalize's relevant-count
    # query) are both scoped to tag-relevance's verdict, not just "touched" -
    # empty until relevance is decided, same as in the real DAG order.
    assert ops.get_new_story_memberships_for_run(run_id) == []
    assert ops.get_touched_relevant_story_ids(run_id) == []

    ops.set_article_relevance(article_id, True)
    members = ops.get_new_story_memberships_for_run(run_id)
    assert members == [{"article_id": article_id, "story_id": story_id}]
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
