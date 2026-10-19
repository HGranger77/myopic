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
