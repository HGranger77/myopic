from fastapi.testclient import TestClient

from database import operations as ops
from main import app

client = TestClient(app)


def _seed_matched_story(sentiment_label: str = "POSITIVE", entity_name: str = "Joe Biden", url: str = None):
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
    ops.upsert_article_sentiment(article_id, story_id, sentiment_label, 0.9, "test-model")
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


def test_stories_page_common_entities_shown_outside_dropdown_only_when_shared_by_all_sources():
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
