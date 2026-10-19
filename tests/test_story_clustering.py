from datetime import datetime, timedelta, timezone

from database import get_connection
from database import operations as ops

T0 = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
NEAREST, CENTROID, HOURS = 0.25, 0.35, 24


def _vec(*head):
    """384-dim unit vector whose first components are `head` (rest zero-padded)."""
    v = list(head) + [0.0] * (384 - len(head))
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def _add_to_story(story_id, vec, when, run_id, url):
    article_id, _ = ops.upsert_article("src", url, "h", "home", run_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET embedding = %s::vector, discovered_at = %s WHERE id = %s",
                (vec, when, article_id),
            )
    ops.add_article_to_story(story_id, article_id, run_id)


def _find(vec, when):
    return ops.find_similar_story(vec, when, HOURS, NEAREST, CENTROID)


def test_joins_close_story_within_window():
    run_id = ops.create_scrape_run()
    story = ops.create_story()
    _add_to_story(story, _vec(1, 0), T0, run_id, "u1")
    assert _find(_vec(1, 0.1), T0 + timedelta(hours=3)) == story


def test_unrelated_article_starts_new_story():
    run_id = ops.create_scrape_run()
    story = ops.create_story()
    _add_to_story(story, _vec(1, 0), T0, run_id, "u1")
    assert _find(_vec(0, 1), T0) is None


def test_window_applies_on_both_sides():
    run_id = ops.create_scrape_run()
    story = ops.create_story()
    _add_to_story(story, _vec(1, 0), T0, run_id, "u1")
    assert _find(_vec(1, 0), T0 + timedelta(hours=23)) == story
    assert _find(_vec(1, 0), T0 - timedelta(hours=23)) == story
    assert _find(_vec(1, 0), T0 + timedelta(hours=25)) is None
    assert _find(_vec(1, 0), T0 - timedelta(hours=25)) is None


def test_centroid_check_blocks_drift():
    # Story = {A, B, C}: A is close to the new article, but B and C pull the
    # centroid far away - nearest passes, centroid fails.
    run_id = ops.create_scrape_run()
    story = ops.create_story()
    _add_to_story(story, _vec(1, 0), T0, run_id, "a")
    _add_to_story(story, _vec(0, 1), T0, run_id, "b")
    _add_to_story(story, _vec(0, 1), T0, run_id, "c")
    assert _find(_vec(1, 0), T0) is None


def test_picks_best_of_several_stories():
    run_id = ops.create_scrape_run()
    far = ops.create_story()
    near = ops.create_story()
    _add_to_story(far, _vec(1, 0.5), T0, run_id, "far")
    _add_to_story(near, _vec(1, 0.05), T0, run_id, "near")
    assert _find(_vec(1, 0), T0) == near


def test_embedding_text_is_headline_plus_short_lead():
    import run_batch

    body = "First   line.\n\nSecond paragraph. " + "x" * 1000
    text = run_batch.embedding_text("Headline", body)
    assert text.startswith("Headline. First line. Second paragraph.")
    assert len(text) == len("Headline. ") + run_batch.EMBED_LEAD_CHARS
    assert run_batch.embedding_text("Headline", None) == "Headline"


def _story_articles(n, body_len=3000):
    return [
        {"source": f"s{i}", "headline": f"headline {i}", "body_text": f"fact{i} " + "x" * body_len,
         "discovered_at": T0 + timedelta(hours=i)}
        for i in range(n)
    ]


def test_combined_story_text_gives_every_article_a_share():
    import run_batch

    text = run_batch._combined_story_text(_story_articles(3))
    assert len(text) <= run_batch.SUMMARY_INPUT_CHARS + 2 * len("\n\n")
    for i in range(3):  # the old plain concatenation cut the last article off entirely
        assert f"fact{i}" in text
    assert text.index("fact2") < text.index("fact0")  # newest first


def test_combined_story_text_short_articles_donate_unused_share():
    import run_batch

    arts = _story_articles(3, body_len=3000)
    arts[0]["body_text"] = "tiny"
    text = run_batch._combined_story_text(arts)
    assert "tiny" in text
    assert len(text) > run_batch.SUMMARY_INPUT_CHARS * 0.95


def test_combined_story_text_drops_oldest_when_too_many():
    import run_batch

    arts = _story_articles(60, body_len=500)
    text = run_batch._combined_story_text(arts)
    assert "headline 59" in text and "headline 0\n" not in text
    assert text.count("[s") <= run_batch.SUMMARY_INPUT_CHARS // run_batch.MIN_ARTICLE_CHARS


def test_window_accepts_float_hours():
    # run_batch passes MYOPIC_STORY_WINDOW_HOURS as a float
    run_id = ops.create_scrape_run()
    story = ops.create_story()
    _add_to_story(story, _vec(1, 0), T0, run_id, "u1")
    assert ops.find_similar_story(_vec(1, 0), T0, 24.0, NEAREST, CENTROID) == story
    assert ops.find_similar_story(_vec(1, 0), T0 + timedelta(hours=13), 12.5, NEAREST, CENTROID) is None
