from fastapi.testclient import TestClient

from database import operations as ops
from main import app

client = TestClient(app)


def _seed_matched_story(
    sentiment_label: str = "POSITIVE", entity_name: str = "Joe Biden", url: str = None, emotion_label: str = None
):
    entity = ops.create_watchlist_entity(entity_name)
    run_id = ops.create_scrape_run()
    article_id, _ = ops.upsert_article(
        "nine_com_au",
        url or "https://www.nine.com.au/world-news/biden-story.html",
        "Biden announces policy",
        "world-news",
        run_id,
    )
    ops.mark_article_fetched(article_id, "Joe Biden announced a new policy today.")
    ops.insert_watchlist_match(article_id, entity["id"], None, entity_name.lower(), run_id)

    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.upsert_story_summary(story_id, "Biden announced a new policy.", "test-model")
    ops.upsert_article_sentiment(
        article_id, story_id, sentiment_label, 0.9, "test-model",
        emotion_label=emotion_label, emotion_score=0.8 if emotion_label else None,
        emotion_model_name="test-emotion-model" if emotion_label else None,
    )
    return story_id, article_id


def test_stories_page_lists_matched_summarized_story():
    _seed_matched_story()
    resp = client.get("/stories")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Biden announces policy" in resp.text
    assert "Biden announced a new policy." in resp.text
    assert "Joe Biden" in resp.text  # matched entity badge
    assert "nine_com_au" in resp.text  # source tag
    assert "POSITIVE" in resp.text  # sentiment badge


def test_stories_page_shows_emotion_badge_alongside_sentiment():
    _seed_matched_story(sentiment_label="NEGATIVE", emotion_label="fear")
    resp = client.get("/stories")
    assert resp.status_code == 200
    assert '<span class="sentiment negative">NEGATIVE</span>' in resp.text
    assert '<span class="emotion fear">fear</span>' in resp.text


def test_stories_page_omits_emotion_badge_when_no_emotion_data():
    _seed_matched_story()  # no emotion_label passed - simulates a pre-emotion-feature row
    resp = client.get("/stories")
    assert resp.status_code == 200
    assert 'class="emotion' not in resp.text


def test_stories_page_excludes_summarized_but_unmatched_story():
    # tag-relevance now runs after summarize-story, so every clustered story
    # gets a summary regardless of relevance - a story with no
    # watchlist_matches row must still be excluded from the page, not just
    # one that was never summarized at all.
    run_id = ops.create_scrape_run()
    article_id, _ = ops.upsert_article(
        "nine_com_au", "https://www.nine.com.au/world-news/unrelated.html", "Unrelated story", "world-news", run_id
    )
    ops.mark_article_fetched(article_id, "Nothing to do with the watchlist.")
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.upsert_story_summary(story_id, "An unrelated summary.", "test-model")

    resp = client.get("/stories")
    assert resp.status_code == 200
    assert "An unrelated summary." not in resp.text
    assert "No reportable stories yet" in resp.text


def test_stories_page_shows_per_article_entities_by_category():
    _, article_id = _seed_matched_story()
    ops.insert_article_entities(article_id, [
        {"text": "Joe Biden", "normalized_text": "joe biden", "label": "PERSON", "start_char": 0, "end_char": 9},
        {"text": "White House", "normalized_text": "white house", "label": "ORG", "start_char": 20, "end_char": 31},
        {"text": "Washington", "normalized_text": "washington", "label": "GPE", "start_char": 40, "end_char": 50},
        {"text": "yesterday", "normalized_text": "yesterday", "label": "DATE", "start_char": 60, "end_char": 69},
    ])

    resp = client.get("/stories")
    assert resp.status_code == 200
    assert '<span class="entity-tag person">Joe Biden</span>' in resp.text
    assert '<span class="entity-tag org">White House</span>' in resp.text
    assert '<span class="entity-tag location">Washington</span>' in resp.text
    assert "yesterday" not in resp.text  # DATE isn't a shown category


def test_stories_page_shows_entity_sentiment_dot_when_scored():
    _, article_id = _seed_matched_story()
    ops.insert_article_entities(article_id, [
        {"text": "Joe Biden", "normalized_text": "joe biden", "label": "PERSON", "start_char": 0, "end_char": 9},
        {"text": "White House", "normalized_text": "white house", "label": "ORG", "start_char": 20, "end_char": 31},
    ])
    ops.insert_article_entity_sentiment(
        article_id,
        [
            {"entity_text": "Joe Biden", "entity_category": "people", "sentiment_label": "Positive", "sentiment_score": 0.9},
        ],
        "test-absa-model",
    )

    resp = client.get("/stories")
    assert resp.status_code == 200
    # scored entity gets a sentiment dot ahead of its text, inside the same tag
    assert '<span class="entity-tag person"><span class="entity-sentiment-dot positive" title="Positive"></span>Joe Biden</span>' in resp.text
    # unscored entity (no article_entity_sentiment row) renders exactly as before - no dot
    assert '<span class="entity-tag org">White House</span>' in resp.text


def test_stories_page_entity_without_sentiment_has_no_dot():
    _, article_id = _seed_matched_story()
    ops.insert_article_entities(article_id, [
        {"text": "Joe Biden", "normalized_text": "joe biden", "label": "PERSON", "start_char": 0, "end_char": 9},
    ])
    # no insert_article_entity_sentiment call - simulates entity-sentiment
    # not having reached this article yet
    resp = client.get("/stories")
    assert resp.status_code == 200
    assert '<span class="entity-tag person">Joe Biden</span>' in resp.text
    # "entity-sentiment-dot" itself appears in the page's <style> block
    # (the CSS class definition) regardless - check no actual dot element
    # was rendered, not that the bare class name string is absent
    assert '<span class="entity-sentiment-dot' not in resp.text


def test_stories_page_matched_entity_badge_is_capitalized():
    # watchlist entity stored lowercase ("politics"), same as a real topic -
    # the badge should display "Politics", not the raw canonical_name
    _seed_matched_story(entity_name="politics", url="https://www.nine.com.au/world-news/politics-badge.html")
    resp = client.get("/stories")
    assert resp.status_code == 200
    assert '<span class="badge">Politics</span>' in resp.text
    assert '<span class="badge">politics</span>' not in resp.text


def test_stories_page_source_breakdown_is_collapsible_dropdown():
    _seed_matched_story()
    resp = client.get("/stories")
    assert resp.status_code == 200
    assert '<details class="sources">' in resp.text
    assert "<summary>1 source</summary>" in resp.text


def test_stories_page_common_entities_two_sources_requires_both():
    # for exactly 2 sources, "more than half" still means both - same
    # behavior as the old strict-intersection rule for small stories
    story_id, article_id = _seed_matched_story(url="https://www.nine.com.au/world-news/biden-story.html")
    ops.insert_article_entities(article_id, [
        {"text": "Joe Biden", "normalized_text": "joe biden", "label": "PERSON", "start_char": 0, "end_char": 9},
        {"text": "White House", "normalized_text": "white house", "label": "ORG", "start_char": 20, "end_char": 31},
    ])

    run_id = ops.create_scrape_run()
    article2_id, _ = ops.upsert_article(
        "abc_net_au",
        "https://www.abc.net.au/news/biden-story.html",
        "Biden policy coverage",
        "world-news",
        run_id,
    )
    ops.mark_article_fetched(article2_id, "Joe Biden announced a new policy today.")
    ops.add_article_to_story(story_id, article2_id, run_id)
    ops.insert_article_entities(article2_id, [
        {"text": "Joe Biden", "normalized_text": "joe biden", "label": "PERSON", "start_char": 0, "end_char": 9},
        {"text": "Canberra", "normalized_text": "canberra", "label": "GPE", "start_char": 20, "end_char": 28},
    ])

    resp = client.get("/stories")
    assert resp.status_code == 200
    common_section = resp.text.split('class="common-entities"')[1].split('<details class="sources">')[0]
    assert "Joe Biden" in common_section
    assert "White House" not in common_section
    assert "Canberra" not in common_section


def test_stories_page_common_entities_use_majority_not_strict_intersection():
    # 5 total sources: "Jane Smith" is mentioned by 3 of them (a majority -
    # should show), "Rare Person" by only 2 (not a majority - should not).
    # A strict "every single source" rule would have hidden both.
    story_id, first_article_id = _seed_matched_story(url="https://www.nine.com.au/world-news/majority-a.html")
    ops.insert_article_entities(first_article_id, [
        {"text": "Jane Smith", "normalized_text": "jane smith", "label": "PERSON", "start_char": 0, "end_char": 10},
    ])

    per_source_names = ["Jane Smith", "Jane Smith", "Rare Person", "Rare Person"]
    run_id = ops.create_scrape_run()
    for i, name in enumerate(per_source_names, start=2):
        article_id, _ = ops.upsert_article(
            "nine_com_au", f"https://www.nine.com.au/world-news/majority-{i}.html", f"Source {i}", "world-news", run_id,
        )
        ops.mark_article_fetched(article_id, "body")
        ops.add_article_to_story(story_id, article_id, run_id)
        ops.insert_article_entities(article_id, [
            {"text": name, "normalized_text": name.lower(), "label": "PERSON", "start_char": 0, "end_char": len(name)},
        ])

    resp = client.get("/stories")
    assert resp.status_code == 200
    common_section = resp.text.split('class="common-entities"')[1].split('<details class="sources">')[0]
    assert "Jane Smith" in common_section
    assert "Rare Person" not in common_section


def test_stories_page_topic_filter_shows_only_matching_story():
    _seed_matched_story(entity_name="Joe Biden", url="https://www.nine.com.au/world-news/biden-story.html")
    ops.create_watchlist_entity("politics")
    run_id = ops.create_scrape_run()
    other_id, _ = ops.upsert_article(
        "nine_com_au", "https://www.nine.com.au/world-news/politics-story.html", "Election called", "world-news", run_id
    )
    ops.mark_article_fetched(other_id, "An election has been called.")
    entity2 = ops.get_active_watchlist_with_aliases()
    politics_entity_id = next(e["watchlist_entity_id"] for e in entity2 if e["canonical_name"] == "politics")
    ops.insert_watchlist_match(other_id, politics_entity_id, None, "politics", run_id)
    other_story_id = ops.create_story()
    ops.add_article_to_story(other_story_id, other_id, run_id)
    ops.upsert_story_summary(other_story_id, "An election has been called.", "test-model")

    resp = client.get("/stories?topic=politics")
    assert resp.status_code == 200
    assert "An election has been called." in resp.text
    assert "Biden announced a new policy." not in resp.text


def test_stories_page_unfiltered_shows_every_matched_story():
    _seed_matched_story(entity_name="Joe Biden", url="https://www.nine.com.au/world-news/biden-story.html")
    resp = client.get("/stories")
    assert resp.status_code == 200
    assert "Biden announced a new policy." in resp.text


def test_stories_page_topic_pills_are_sorted_capitalized_and_counted():
    # "AFL" must stay as-is (not mangled to "Afl") while "sports" (lowercase
    # in the DB) gets its display form capitalized - and alphabetical sort
    # is case-insensitive, so "AFL" sorts before "sports", not after every
    # lowercase letter the way a case-sensitive sort would put it.
    ops.create_watchlist_entity("sports")
    _seed_matched_story(entity_name="AFL", url="https://www.nine.com.au/world-news/afl-story.html")

    resp = client.get("/stories")
    assert resp.status_code == 200
    assert resp.text.index(">AFL (1)<") < resp.text.index(">Sports (0)<")
    assert ">Afl" not in resp.text


def test_stories_page_empty_state():
    resp = client.get("/stories")
    assert resp.status_code == 200
    assert "No reportable stories yet" in resp.text


def test_stories_page_escapes_html_in_scraped_content():
    entity = ops.create_watchlist_entity("<script>alert(1)</script>")
    run_id = ops.create_scrape_run()
    article_id, _ = ops.upsert_article(
        "nine_com_au",
        "https://www.nine.com.au/world-news/xss.html",
        "<script>alert('headline')</script>",
        "world-news",
        run_id,
    )
    ops.mark_article_fetched(article_id, "body")
    ops.insert_watchlist_match(article_id, entity["id"], None, "<script>alert(1)</script>", run_id)

    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.upsert_story_summary(story_id, "<script>alert('summary')</script>", "test-model")

    resp = client.get("/stories")
    assert resp.status_code == 200
    assert "<script>" not in resp.text
    assert "&lt;script&gt;" in resp.text


def _seed_n_stories(n, entity_id, entity_name, url_prefix):
    story_ids = []
    for i in range(n):
        run_id = ops.create_scrape_run()
        article_id, _ = ops.upsert_article(
            "nine_com_au", f"{url_prefix}-{i}.html", f"Story {i}", "world-news", run_id,
        )
        ops.mark_article_fetched(article_id, "Some policy content.")
        ops.insert_watchlist_match(article_id, entity_id, None, entity_name.lower(), run_id)
        story_id = ops.create_story()
        ops.add_article_to_story(story_id, article_id, run_id)
        ops.upsert_story_summary(story_id, f"Summary {i}.", "test-model")
        story_ids.append(story_id)
    return story_ids


def test_stories_page_pagination_splits_across_pages():
    from routers.stories import PAGE_SIZE

    entity = ops.create_watchlist_entity("Joe Biden")
    _seed_n_stories(PAGE_SIZE + 1, entity["id"], "Joe Biden", "https://www.nine.com.au/world-news/page-story")

    resp = client.get("/stories")
    assert resp.status_code == 200
    assert "Page 1 of 2" in resp.text
    assert resp.text.count('<article class="story">') == PAGE_SIZE

    resp2 = client.get("/stories?page=2")
    assert resp2.status_code == 200
    assert "Page 2 of 2" in resp2.text
    assert resp2.text.count('<article class="story">') == 1


def test_stories_page_date_filter_shows_only_that_day():
    entity = ops.create_watchlist_entity("Joe Biden")
    old_story_id, new_story_id = _seed_n_stories(
        2, entity["id"], "Joe Biden", "https://www.nine.com.au/world-news/date-story"
    )
    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-01-01 10:00:00+00", old_story_id))

    resp = client.get("/stories?date=2026-01-01")
    assert resp.status_code == 200
    assert resp.text.count('<article class="story">') == 1
    assert "Summary 0." in resp.text
    assert "Summary 1." not in resp.text
    assert "Showing stories updated on" in resp.text


def test_stories_page_calendar_highlights_days_with_content():
    entity = ops.create_watchlist_entity("Joe Biden")
    (story_id,) = _seed_n_stories(1, entity["id"], "Joe Biden", "https://www.nine.com.au/world-news/cal-story")
    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-01-15 10:00:00+00", story_id))

    resp = client.get("/stories?calendar_month=2026-01")
    assert resp.status_code == 200
    assert "January 2026" in resp.text
    assert 'href="/stories?date=2026-01-15"' in resp.text  # clickable - has content
    assert '<span class="calendar-day">14</span>' in resp.text  # other days still render, just not as links


def test_stories_page_search_filters_by_headline_and_summary():
    entity = ops.create_watchlist_entity("Joe Biden")
    run_id = ops.create_scrape_run()
    matching_id, _ = ops.upsert_article(
        "nine_com_au", "https://www.nine.com.au/world-news/search-match.html",
        "Tariffs hit record high", "world-news", run_id,
    )
    ops.mark_article_fetched(matching_id, "body")
    ops.insert_watchlist_match(matching_id, entity["id"], None, "joe biden", run_id)
    matching_story = ops.create_story()
    ops.add_article_to_story(matching_story, matching_id, run_id)
    ops.upsert_story_summary(matching_story, "A story about something else.", "test-model")

    _seed_n_stories(1, entity["id"], "Joe Biden", "https://www.nine.com.au/world-news/search-nomatch")

    resp = client.get("/stories?q=tariffs")
    assert resp.status_code == 200
    assert resp.text.count('<article class="story">') == 1
    assert "Tariffs hit record high" in resp.text
    assert "Searching for" in resp.text and "tariffs" in resp.text

    resp_none = client.get("/stories?q=nonexistentterm")
    assert resp_none.status_code == 200
    assert "No stories found" in resp_none.text


def test_stories_page_topic_counts_reflect_active_date_filter():
    entity = ops.create_watchlist_entity("AFL")
    old_story, new_story = _seed_n_stories(2, entity["id"], "AFL", "https://www.nine.com.au/world-news/topic-date")
    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-03-01 10:00:00+00", old_story))
            cur.execute("UPDATE stories SET updated_at = %s WHERE id = %s", ("2026-03-02 10:00:00+00", new_story))

    resp_unfiltered = client.get("/stories")
    assert ">AFL (2)<" in resp_unfiltered.text

    resp_filtered = client.get("/stories?date=2026-03-01")
    assert ">AFL (1)<" in resp_filtered.text
    assert ">AFL (2)<" not in resp_filtered.text


def test_stories_page_topic_counts_reflect_active_search():
    entity = ops.create_watchlist_entity("AFL")
    run_id = ops.create_scrape_run()
    article_id, _ = ops.upsert_article(
        "nine_com_au", "https://www.nine.com.au/world-news/topic-search-match.html",
        "Grand final preview", "world-news", run_id,
    )
    ops.mark_article_fetched(article_id, "body")
    ops.insert_watchlist_match(article_id, entity["id"], None, "afl", run_id)
    story_id = ops.create_story()
    ops.add_article_to_story(story_id, article_id, run_id)
    ops.upsert_story_summary(story_id, "A summary.", "test-model")

    _seed_n_stories(1, entity["id"], "AFL", "https://www.nine.com.au/world-news/topic-search-nomatch")

    resp_unfiltered = client.get("/stories")
    assert ">AFL (2)<" in resp_unfiltered.text

    resp_filtered = client.get("/stories?q=grand+final")
    assert ">AFL (1)<" in resp_filtered.text
