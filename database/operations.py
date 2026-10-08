"""Raw-SQL CRUD operations. Each function opens and commits its own connection."""
from datetime import datetime, timezone
from typing import Iterable, Optional

from database import get_connection


# ---------------------------------------------------------------------------
# watchlist_entities / watchlist_aliases
# ---------------------------------------------------------------------------

def create_watchlist_entity(
    canonical_name: str, entity_type: Optional[str] = None, notes: Optional[str] = None
) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO watchlist_entities (canonical_name, entity_type, notes)
                VALUES (%s, %s, %s)
                RETURNING *
                """,
                (canonical_name, entity_type, notes),
            )
            entity = cur.fetchone()
            _insert_alias(cur, entity["id"], canonical_name)
    return get_watchlist_entity(entity["id"])


def get_watchlist_entity(entity_id: int) -> Optional[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM watchlist_entities WHERE id = %s", (entity_id,))
            entity = cur.fetchone()
            if entity is None:
                return None
            cur.execute(
                "SELECT * FROM watchlist_aliases WHERE watchlist_entity_id = %s ORDER BY id",
                (entity_id,),
            )
            entity = dict(entity)
            entity["aliases"] = cur.fetchall()
            return entity


def list_watchlist_entities(active_only: bool = False) -> list[dict]:
    query = "SELECT * FROM watchlist_entities"
    if active_only:
        query += " WHERE is_active = TRUE"
    query += " ORDER BY id"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            entities = [dict(row) for row in cur.fetchall()]
            for entity in entities:
                cur.execute(
                    "SELECT * FROM watchlist_aliases WHERE watchlist_entity_id = %s ORDER BY id",
                    (entity["id"],),
                )
                entity["aliases"] = cur.fetchall()
    return entities


def update_watchlist_entity(entity_id: int, **fields) -> Optional[dict]:
    if not fields:
        return get_watchlist_entity(entity_id)
    allowed = {"canonical_name", "entity_type", "notes", "is_active"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    set_clause = ", ".join(f"{k} = %s" for k in fields)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE watchlist_entities SET {set_clause}, updated_at = now() "
                "WHERE id = %s RETURNING id",
                (*fields.values(), entity_id),
            )
            if cur.fetchone() is None:
                return None
    return get_watchlist_entity(entity_id)


def delete_watchlist_entity(entity_id: int) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM watchlist_entities WHERE id = %s RETURNING id", (entity_id,)
            )
            return cur.fetchone() is not None


def add_alias(entity_id: int, alias: str) -> Optional[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM watchlist_entities WHERE id = %s", (entity_id,))
            if cur.fetchone() is None:
                return None
            return _insert_alias(cur, entity_id, alias)


def remove_alias(alias_id: int) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM watchlist_aliases WHERE id = %s RETURNING id", (alias_id,)
            )
            return cur.fetchone() is not None


def _insert_alias(cur, entity_id: int, alias: str) -> dict:
    from nlp.matching import normalize

    cur.execute(
        """
        INSERT INTO watchlist_aliases (watchlist_entity_id, alias, normalized_alias)
        VALUES (%s, %s, %s)
        ON CONFLICT (watchlist_entity_id, normalized_alias) DO UPDATE SET alias = EXCLUDED.alias
        RETURNING *
        """,
        (entity_id, alias, normalize(alias)),
    )
    return cur.fetchone()


def get_active_watchlist_with_aliases() -> list[dict]:
    """Returns [{watchlist_entity_id, canonical_name, aliases: [normalized_alias, ...]}]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT we.id AS watchlist_entity_id, we.canonical_name, wa.normalized_alias
                FROM watchlist_entities we
                JOIN watchlist_aliases wa ON wa.watchlist_entity_id = we.id
                WHERE we.is_active = TRUE
                ORDER BY we.id
                """
            )
            rows = cur.fetchall()
    grouped: dict[int, dict] = {}
    for row in rows:
        entry = grouped.setdefault(
            row["watchlist_entity_id"],
            {
                "watchlist_entity_id": row["watchlist_entity_id"],
                "canonical_name": row["canonical_name"],
                "aliases": [],
            },
        )
        entry["aliases"].append(row["normalized_alias"])
    return list(grouped.values())


# ---------------------------------------------------------------------------
# scrape_runs
# ---------------------------------------------------------------------------

def create_scrape_run() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO scrape_runs (status) VALUES ('running') RETURNING id"
            )
            return cur.fetchone()["id"]


def increment_scrape_run_counts(
    run_id: int,
    front_pages_scraped: int = 0,
    front_pages_failed: int = 0,
    articles_found: int = 0,
) -> None:
    """Atomic increment, not read-modify-write - safe for concurrent per-source
    discover tasks to call against the same run_id without clobbering each other."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE scrape_runs
                SET front_pages_scraped = front_pages_scraped + %s,
                    front_pages_failed = front_pages_failed + %s,
                    articles_found = articles_found + %s
                WHERE id = %s
                """,
                (front_pages_scraped, front_pages_failed, articles_found, run_id),
            )


def finish_scrape_run(
    run_id: int,
    status: str,
    articles_fetched: int = 0,
    articles_matched: int = 0,
    stories_touched: int = 0,
    summaries_generated: int = 0,
    error_message: Optional[str] = None,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE scrape_runs
                SET finished_at = %s, status = %s, articles_fetched = %s,
                    articles_matched = %s, stories_touched = %s,
                    summaries_generated = %s, error_message = %s
                WHERE id = %s
                """,
                (
                    datetime.now(timezone.utc),
                    status,
                    articles_fetched,
                    articles_matched,
                    stories_touched,
                    summaries_generated,
                    error_message,
                    run_id,
                ),
            )


def get_scrape_run(run_id: int) -> Optional[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM scrape_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
# articles
# ---------------------------------------------------------------------------

def upsert_article(
    source: str, url: str, headline: str, front_page_slug: str, run_id: int
) -> tuple[int, bool]:
    """Returns (article_id, is_new)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO articles (source, url, headline, front_page_slug, first_seen_run_id, last_seen_run_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET last_seen_run_id = EXCLUDED.last_seen_run_id
                RETURNING id, (xmax = 0) AS is_new
                """,
                (source, url, headline, front_page_slug, run_id, run_id),
            )
            row = cur.fetchone()
            return row["id"], row["is_new"]


def get_pending_articles() -> list[dict]:
    """Articles never successfully fetched yet (any run) - fetch-article-bodies'
    worklist. Permanently-failed articles are not retried."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM articles WHERE fetch_status = 'pending'")
            return [dict(row) for row in cur.fetchall()]


def get_article(article_id: int) -> Optional[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def mark_article_fetched(article_id: int, body_text: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE articles
                SET body_text = %s, fetch_status = 'fetched', fetched_at = now()
                WHERE id = %s
                """,
                (body_text, article_id),
            )


def mark_article_failed(article_id: int, error: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET fetch_status = 'failed', fetch_error = %s WHERE id = %s",
                (error, article_id),
            )


# ---------------------------------------------------------------------------
# filter-article: topic-of-interest relevance + per-article entities
# ---------------------------------------------------------------------------

def get_articles_needing_filtering() -> list[dict]:
    """Fetched articles not yet run through filter-article."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM articles WHERE fetch_status = 'fetched' AND is_relevant IS NULL"
            )
            return [dict(row) for row in cur.fetchall()]


def set_article_relevance(article_id: int, is_relevant: bool) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET is_relevant = %s WHERE id = %s", (is_relevant, article_id)
            )


def delete_article_entities(article_id: int) -> None:
    """Clears prior entities/matches for an article before reprocessing it -
    makes filter-article safe to retry without duplicating rows."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM article_entities WHERE article_id = %s", (article_id,))
            cur.execute("DELETE FROM watchlist_matches WHERE article_id = %s", (article_id,))


def insert_article_entities(article_id: int, entities: Iterable[dict]) -> list[dict]:
    """entities: iterable of {text, normalized_text, label, start_char, end_char}."""
    inserted = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            for ent in entities:
                cur.execute(
                    """
                    INSERT INTO article_entities
                        (article_id, entity_text, normalized_text, entity_label, start_char, end_char)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        article_id,
                        ent["text"],
                        ent["normalized_text"],
                        ent["label"],
                        ent.get("start_char"),
                        ent.get("end_char"),
                    ),
                )
                inserted.append(cur.fetchone())
    return inserted


def insert_watchlist_match(
    article_id: int,
    watchlist_entity_id: int,
    article_entity_id: Optional[int],
    matched_alias: str,
    scrape_run_id: int,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO watchlist_matches
                    (article_id, watchlist_entity_id, article_entity_id, matched_alias, scrape_run_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (article_id, watchlist_entity_id) DO NOTHING
                """,
                (article_id, watchlist_entity_id, article_entity_id, matched_alias, scrape_run_id),
            )


def count_matched_articles(scrape_run_id: int) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(DISTINCT article_id) AS n FROM watchlist_matches WHERE scrape_run_id = %s",
                (scrape_run_id,),
            )
            return cur.fetchone()["n"]


# ---------------------------------------------------------------------------
# embed-article
# ---------------------------------------------------------------------------

def get_articles_needing_embedding() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM articles WHERE is_relevant = TRUE AND embedding IS NULL"
            )
            return [dict(row) for row in cur.fetchall()]


def set_article_embedding(article_id: int, embedding: list[float]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET embedding = %s WHERE id = %s", (embedding, article_id)
            )


# ---------------------------------------------------------------------------
# stories / story_articles (cluster-stories)
# ---------------------------------------------------------------------------

def get_articles_needing_clustering() -> list[dict]:
    """Relevant, embedded articles not yet assigned to a story."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.* FROM articles a
                LEFT JOIN story_articles sa ON sa.article_id = a.id
                WHERE a.is_relevant = TRUE AND a.embedding IS NOT NULL AND sa.id IS NULL
                """
            )
            return [dict(row) for row in cur.fetchall()]


def find_similar_story(
    embedding: list[float], window_days: int, max_distance: float
) -> Optional[int]:
    """Nearest existing story (by its most similar member article) within the
    rolling window, if under max_distance. None means "start a new story"."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sa.story_id, a.embedding <=> %s::vector AS distance
                FROM story_articles sa
                JOIN articles a ON a.id = sa.article_id
                JOIN stories s ON s.id = sa.story_id
                WHERE s.updated_at > now() - (%s || ' days')::interval
                ORDER BY distance ASC
                LIMIT 1
                """,
                (embedding, window_days),
            )
            row = cur.fetchone()
            if row is None or row["distance"] > max_distance:
                return None
            return row["story_id"]


def create_story() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO stories DEFAULT VALUES RETURNING id")
            return cur.fetchone()["id"]


def add_article_to_story(story_id: int, article_id: int, scrape_run_id: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO story_articles (story_id, article_id, scrape_run_id) VALUES (%s, %s, %s)",
                (story_id, article_id, scrape_run_id),
            )
            cur.execute("UPDATE stories SET updated_at = now() WHERE id = %s", (story_id,))


def get_touched_story_ids(scrape_run_id: int) -> list[int]:
    """Stories that gained a member this run - new stories and ones that grew."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT story_id FROM story_articles WHERE scrape_run_id = %s",
                (scrape_run_id,),
            )
            return [row["story_id"] for row in cur.fetchall()]


def count_summaries_for_stories(story_ids: list[int]) -> int:
    if not story_ids:
        return 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM story_summaries WHERE story_id = ANY(%s)", (story_ids,)
            )
            return cur.fetchone()["n"]


def count_fetched_for_run(scrape_run_id: int) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM articles WHERE last_seen_run_id = %s AND fetch_status = 'fetched'",
                (scrape_run_id,),
            )
            return cur.fetchone()["n"]


def get_new_story_memberships_for_run(scrape_run_id: int) -> list[dict]:
    """This run's new story memberships - sentiment-article's worklist (one-shot
    per article, unlike summarize-story/story-entities which redo the whole story).
    Returns [{article_id, story_id}, ...]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT article_id, story_id FROM story_articles WHERE scrape_run_id = %s",
                (scrape_run_id,),
            )
            return [dict(row) for row in cur.fetchall()]


def get_story_articles(story_id: int) -> list[dict]:
    """Every member article of a story, across all runs - summarize-story and
    story-entities regenerate from the complete set, not just this run's delta."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.* FROM articles a
                JOIN story_articles sa ON sa.article_id = a.id
                WHERE sa.story_id = %s
                ORDER BY a.discovered_at
                """,
                (story_id,),
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# story_summaries
# ---------------------------------------------------------------------------

def upsert_story_summary(story_id: int, summary_text: str, model_name: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO story_summaries (story_id, summary_text, model_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (story_id) DO UPDATE
                    SET summary_text = EXCLUDED.summary_text, model_name = EXCLUDED.model_name,
                        updated_at = now()
                """,
                (story_id, summary_text, model_name),
            )


# ---------------------------------------------------------------------------
# story_entities
# ---------------------------------------------------------------------------

def delete_story_entities(story_id: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM story_entities WHERE story_id = %s", (story_id,))


def insert_story_entities(story_id: int, entities: Iterable[dict], model_name: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            for ent in entities:
                cur.execute(
                    """
                    INSERT INTO story_entities
                        (story_id, entity_text, normalized_text, entity_label, model_name)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (story_id, ent["text"], ent["normalized_text"], ent["label"], model_name),
                )


# ---------------------------------------------------------------------------
# article_sentiment
# ---------------------------------------------------------------------------

def upsert_article_sentiment(
    article_id: int, story_id: int, label: str, score: float, model_name: str
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO article_sentiment (article_id, story_id, sentiment_label, sentiment_score, model_name)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (article_id) DO UPDATE
                    SET sentiment_label = EXCLUDED.sentiment_label,
                        sentiment_score = EXCLUDED.sentiment_score,
                        model_name = EXCLUDED.model_name
                """,
                (article_id, story_id, label, score, model_name),
            )


# ---------------------------------------------------------------------------
# story viewer (web page)
# ---------------------------------------------------------------------------

def get_recent_stories_with_summaries(limit: int = 50) -> list[dict]:
    """Stories with a summary, newest-updated first, for the web viewer."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    s.id AS story_id, s.updated_at, ss.summary_text,
                    array_agg(DISTINCT a.source) AS sources,
                    array_agg(DISTINCT we.canonical_name) AS matched_entities
                FROM stories s
                JOIN story_summaries ss ON ss.story_id = s.id
                JOIN story_articles sa ON sa.story_id = s.id
                JOIN articles a ON a.id = sa.article_id
                LEFT JOIN watchlist_matches wm ON wm.article_id = a.id
                LEFT JOIN watchlist_entities we ON we.id = wm.watchlist_entity_id
                GROUP BY s.id, s.updated_at, ss.summary_text
                ORDER BY s.updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]


def get_story_source_articles(story_id: int) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id, a.source, a.url, a.headline, asent.sentiment_label, asent.sentiment_score
                FROM articles a
                JOIN story_articles sa ON sa.article_id = a.id
                LEFT JOIN article_sentiment asent ON asent.article_id = a.id
                WHERE sa.story_id = %s
                ORDER BY a.discovered_at
                """,
                (story_id,),
            )
            return [dict(row) for row in cur.fetchall()]
