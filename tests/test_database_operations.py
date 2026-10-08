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
    members = ops.get_new_story_memberships_for_run(run_id)
    assert members == [{"article_id": article_id, "story_id": story_id}]

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
