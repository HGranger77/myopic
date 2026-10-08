from fastapi.testclient import TestClient

from database import operations as ops
from main import app

client = TestClient(app)


def _seed_matched_story(sentiment_label: str = "POSITIVE"):
    entity = ops.create_watchlist_entity("Joe Biden")
    run_id = ops.create_scrape_run()
    article_id, _ = ops.upsert_article(
        "nine_com_au", "https://www.nine.com.au/world-news/biden-story.html", "Biden announces policy", "world-news", run_id
    )
    ops.mark_article_fetched(article_id, "Joe Biden announced a new policy today.")
    ops.insert_watchlist_match(article_id, entity["id"], None, "joe biden", run_id)

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
