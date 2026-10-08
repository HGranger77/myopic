from pathlib import Path

import run_batch
from database import operations as ops
from nlp import embedding as embedding_module
from nlp import ner as ner_module
from nlp import sentiment as sentiment_module
from nlp import summarize as summarize_module
from scraper import fetch

FIXTURES = Path(__file__).parent / "fixtures"
FRONT_PAGE_HTML = (FIXTURES / "smoke_front_page.html").read_text()
ARTICLE_HTML = (FIXTURES / "article_sample.html").read_text()
ARTICLE_URL = "https://www.nine.com.au/australia-news/solar-farm-fight-20261006-p613cq.html"

FAKE_SOURCES = {
    "sources": [
        {
            "name": "nine.com.au",
            "module": "nine_com_au",
            "base_url": "https://www.nine.com.au",
            "front_pages": [{"slug": "home", "path": "/"}],
        }
    ]
}


def _fake_get(url, timeout=15):
    if url == "https://www.nine.com.au/":
        return FRONT_PAGE_HTML
    raise AssertionError(f"unexpected sync fetch url in smoke test: {url}")


async def _fake_get_many(urls, concurrency=8, timeout=15):
    assert urls == [ARTICLE_URL]
    return {ARTICLE_URL: (ARTICLE_HTML, None)}


def _fake_extract_entities(text):
    return [
        {"text": "Joe Biden", "normalized_text": "joe biden", "label": "PERSON", "start_char": 0, "end_char": 9}
    ]


def _fake_embed(text):
    return [0.1] * 384


def _fake_summarize(text):
    return "A stubbed summary of the story."


def _fake_analyze_sentiment(text):
    return "POSITIVE", 0.95


def _patch_models(monkeypatch):
    # run_batch imports each nlp.* module LOCALLY inside the function that
    # needs it (so lightweight stages don't pay for torch/spacy imports they
    # never use) - patch the source modules, not run_batch's own namespace,
    # since `from nlp.x import y` re-resolves nlp.x.y fresh on every call.
    monkeypatch.setattr(fetch, "get", _fake_get)
    monkeypatch.setattr(fetch, "get_many", _fake_get_many)
    monkeypatch.setattr(run_batch, "load_sources", lambda: FAKE_SOURCES)
    monkeypatch.setattr(ner_module, "extract_entities", _fake_extract_entities)
    monkeypatch.setattr(embedding_module, "embed", _fake_embed)
    monkeypatch.setattr(summarize_module, "summarize", _fake_summarize)
    monkeypatch.setattr(sentiment_module, "analyze", _fake_analyze_sentiment)


def test_pipeline_end_to_end(monkeypatch, tmp_path):
    entity = ops.create_watchlist_entity("Joe Biden")
    ops.add_alias(entity["id"], "Biden")
    _patch_models(monkeypatch)

    # Mirrors the full Argo DAG: start-run -> discover -> fetch-article-bodies
    # -> filter-article -> embed-article -> cluster-stories -> the three
    # parallel branches -> finalize.
    assert run_batch.start_run(tmp_path) == 0
    run_id = int((tmp_path / "run_id").read_text())

    assert run_batch.discover(run_id, "nine_com_au") == 0
    assert run_batch.fetch_article_bodies() == 0
    assert run_batch.filter_article(run_id) == 0
    assert run_batch.embed_article() == 0
    assert run_batch.cluster_stories(run_id) == 0
    assert run_batch.summarize_story(run_id) == 0
    assert run_batch.sentiment_article(run_id) == 0
    assert run_batch.story_entities(run_id) == 0
    assert run_batch.finalize(run_id) == 0

    touched = ops.get_touched_story_ids(run_id)
    assert len(touched) == 1
    story_id = touched[0]

    articles = ops.get_story_articles(story_id)
    assert len(articles) == 1
    assert articles[0]["is_relevant"] is True

    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, articles_matched, stories_touched, summaries_generated FROM scrape_runs WHERE id = %s", (run_id,))
            run_row = cur.fetchone()
            cur.execute("SELECT summary_text FROM story_summaries WHERE story_id = %s", (story_id,))
            summary_row = cur.fetchone()
            cur.execute("SELECT sentiment_label FROM article_sentiment WHERE article_id = %s", (articles[0]["id"],))
            sentiment_row = cur.fetchone()
            cur.execute("SELECT count(*) AS n FROM story_entities WHERE story_id = %s", (story_id,))
            entity_count = cur.fetchone()["n"]

    assert run_row["status"] == "success"
    assert run_row["articles_matched"] == 1
    assert run_row["stories_touched"] == 1
    assert run_row["summaries_generated"] == 1
    assert summary_row["summary_text"] == "A stubbed summary of the story."
    assert sentiment_row["sentiment_label"] == "POSITIVE"
    assert entity_count == 1


def test_filter_article_worklist_excludes_already_filtered(monkeypatch, tmp_path):
    """Retry-safety for the new single-task-per-stage design comes from the
    worklist query itself (is_relevant IS NULL), not a delete-then-reinsert
    per item - a retried task just finds nothing left to do for articles it
    already finished."""
    entity = ops.create_watchlist_entity("Joe Biden")
    ops.add_alias(entity["id"], "Biden")
    _patch_models(monkeypatch)

    run_batch.start_run(tmp_path)
    run_id = int((tmp_path / "run_id").read_text())
    run_batch.discover(run_id, "nine_com_au")
    run_batch.fetch_article_bodies()

    run_batch.filter_article(run_id)
    run_batch.filter_article(run_id)  # simulate a retried task - should find nothing left to do

    with ops.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM articles LIMIT 1")
            article_id = cur.fetchone()["id"]
            cur.execute("SELECT count(*) AS n FROM article_entities WHERE article_id = %s", (article_id,))
            entity_count = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM watchlist_matches WHERE article_id = %s", (article_id,))
            match_count = cur.fetchone()["n"]

    assert entity_count == 1  # not duplicated by the second pass
    assert match_count == 1
