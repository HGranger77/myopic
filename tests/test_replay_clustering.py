from datetime import datetime, timedelta, timezone

import numpy as np

from database import get_connection
from database import operations as ops
from scripts.replay_clustering import Article, cluster_new, cluster_old, diff_stories, group

T0 = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


def _vec(*head):
    v = np.array(list(head) + [0.0] * (384 - len(head)), dtype=float)
    return v / np.linalg.norm(v)


def _art(i, vec, hours=0.0):
    return Article(i, "src", f"headline {i}", T0 + timedelta(hours=hours), vec)


def test_old_rule_links_across_days_new_rule_does_not():
    arts = [_art(1, _vec(1, 0)), _art(2, _vec(1, 0), hours=72)]
    assert len(group(cluster_old(arts))) == 1
    assert len(group(cluster_new(arts))) == 2


def test_new_rule_blocks_drift_old_rule_allows():
    # chain of articles each 20 degrees from the last (distance ~0.06 per hop):
    # single-link follows it all the way round, the centroid check cuts it off
    arts = [
        _art(i, _vec(np.cos(np.radians(20 * i)), np.sin(np.radians(20 * i))), hours=i) for i in range(8)
    ]
    assert len(group(cluster_old(arts))) == 1
    assert len(group(cluster_new(arts))) > 1


def test_diff_reports_split_stories():
    arts = [_art(1, _vec(1, 0)), _art(2, _vec(1, 0), hours=72)]
    lines = diff_stories(cluster_old(arts), cluster_new(arts), {a.id: a for a in arts})
    assert any("split into 2" in line for line in lines)


def test_replay_matches_database_rule():
    """cluster_new must make the same decisions as ops.find_similar_story."""
    rng = np.random.default_rng(0)
    base = [_vec(*rng.normal(size=4)) for _ in range(3)]
    arts = []
    for i in range(40):
        v = base[i % 3] + rng.normal(scale=0.15, size=384) * (1 if i % 5 else 3)
        arts.append(_art(i + 1, v / np.linalg.norm(v), hours=float(rng.uniform(0, 60))))
    arts.sort(key=lambda a: a.discovered_at)
    expected = cluster_new(arts)

    run_id = ops.create_scrape_run()
    db_story, replay_to_db = {}, {}
    for art in arts:
        story = ops.find_similar_story(art.embedding.tolist(), art.discovered_at, 24, 0.25, 0.35)
        if story is None:
            story = ops.create_story()
        article_id, _ = ops.upsert_article("src", f"u{art.id}", "h", "home", run_id)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE articles SET embedding = %s::vector, discovered_at = %s WHERE id = %s",
                    (art.embedding.tolist(), art.discovered_at, article_id),
                )
        ops.add_article_to_story(story, article_id, run_id)
        db_story[art.id] = story
    assert len(set(expected.values())) > 1, "test data should produce several stories"
    assert {frozenset(s) for s in group(expected)} == {frozenset(s) for s in group(db_story)}
