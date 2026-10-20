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
    """Returns [{watchlist_entity_id, canonical_name, aliases: [alias, ...]}].

    `aliases` holds the ORIGINAL-case alias text (`wa.alias`), not
    `normalized_alias` - matters more than it looks. nlp.tagging's zero-shot
    classifier treats whichever alias it's given as a literal candidate
    label, and casing changes its score: measured directly, "AFL" scored
    0.01-0.40 on unrelated stories while the lowercased "afl" scored
    0.64-0.76 on the SAME stories - a false-positive regression introduced
    by an earlier version of this function that returned normalized_alias.
    Since each normalized alias is unique per entity (the table's UNIQUE
    constraint), there's exactly one original-case spelling to pick."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT we.id AS watchlist_entity_id, we.canonical_name, wa.alias
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
        entry["aliases"].append(row["alias"])
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
# translate-article: non-English sources only (see config/sources.yaml)
# ---------------------------------------------------------------------------

def get_articles_needing_translation(
    source_modules: list[str], limit: Optional[int] = None
) -> list[dict]:
    """Fetched articles from a non-English source not yet translated, oldest
    first. source_modules: the subset of config/sources.yaml's modules whose
    `language` isn't "en" - empty means nothing to do, no query needed.
    limit: run_batch.py's TRANSLATE_MAX_ARTICLES caps the per-run cost of
    this autoregressive, not-yet-rate-limited stage - leftover articles just
    wait for a later run. None means no cap (e.g. for tests)."""
    if not source_modules:
        return []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM articles
                WHERE fetch_status = 'fetched' AND source = ANY(%s) AND original_body_text IS NULL
                ORDER BY discovered_at
                LIMIT %s
                """,
                (source_modules, limit),
            )
            return [dict(row) for row in cur.fetchall()]


def set_article_translation(
    article_id: int, translated_headline: str, translated_body_text: str
) -> None:
    """Moves the as-scraped headline/body_text into original_headline/
    original_body_text, then replaces them with the English translation -
    every stage downstream just reads headline/body_text either way."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE articles
                SET original_headline = headline, original_body_text = body_text,
                    headline = %s, body_text = %s
                WHERE id = %s
                """,
                (translated_headline, translated_body_text, article_id),
            )


# ---------------------------------------------------------------------------
# tag-relevance: story-level topic classification, stamped onto member articles
# ---------------------------------------------------------------------------

def set_article_relevance(article_id: int, is_relevant: bool) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET is_relevant = %s WHERE id = %s", (is_relevant, article_id)
            )


def delete_watchlist_matches(article_id: int) -> None:
    """Clears prior matches for an article before reprocessing it - makes
    tag-relevance safe to retry without duplicating rows."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM watchlist_matches WHERE article_id = %s", (article_id,))


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

def get_articles_needing_embedding(non_english_modules: Iterable[str] = ()) -> list[dict]:
    """Every fetched article, not just watchlist-relevant ones - relevance
    isn't decided until tag-relevance, downstream of clustering/summarizing
    now (see that function's docstring in run_batch.py for why).

    non_english_modules: source modules translate-article treats as
    non-English (same list that stage computes from config/sources.yaml).
    An article from one of these sources is excluded here until it's
    actually been translated (original_headline IS NOT NULL) - without this,
    embed-article was feeding RAW CHINESE TEXT into an English-only sentence
    embedding model for any article still waiting behind
    translate-article's TRANSLATE_MAX_ARTICLES-per-run cap. Confirmed this
    is exactly what happened to 49 chinanews articles that piled up
    untranslated and all got clustered into one story based on meaningless
    embeddings, which then fed a mush of raw Chinese into the English-only
    summarizer and produced garbled nonsense - see nlp/summarize.py's
    docstring for the measured garbage output."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM articles
                WHERE fetch_status = 'fetched' AND embedding IS NULL
                AND (source != ALL(%(non_english)s) OR original_headline IS NOT NULL)
                """,
                {"non_english": list(non_english_modules)},
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
    """Every embedded article not yet assigned to a story - clustering runs
    on all daily news now, not just watchlist-relevant articles."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.* FROM articles a
                LEFT JOIN story_articles sa ON sa.article_id = a.id
                WHERE a.embedding IS NOT NULL AND sa.id IS NULL
                """
            )
            return [dict(row) for row in cur.fetchall()]


def find_similar_story(
    embedding: list[float],
    discovered_at,
    window_hours: float,
    max_nearest_distance: float,
    max_centroid_distance: float,
) -> Optional[int]:
    """Best existing story for an article, or None for "start a new story".

    Only member articles discovered within +/- window_hours of
    `discovered_at` are considered, so stories never link across more than
    that gap. A story qualifies only if BOTH hold:
      - its nearest in-window member is within max_nearest_distance (the
        article closely matches at least one report), AND
      - the centroid of its in-window members is within
        max_centroid_distance (the article fits the story as a whole, not
        just one outlier member - this stops a story drifting from, say,
        Gulf conflict into Saudi football through a chain of loose links).
    Qualifying stories are ranked by the sum of the two distances.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sa.story_id,
                       MIN(a.embedding <=> %(emb)s::vector) AS nearest,
                       AVG(a.embedding) <=> %(emb)s::vector AS centroid
                FROM story_articles sa
                JOIN articles a ON a.id = sa.article_id
                WHERE a.discovered_at BETWEEN %(t)s - (%(h)s * interval '1 hour')
                                          AND %(t)s + (%(h)s * interval '1 hour')
                GROUP BY sa.story_id
                HAVING MIN(a.embedding <=> %(emb)s::vector) <= %(max_nearest)s
                   AND AVG(a.embedding) <=> %(emb)s::vector <= %(max_centroid)s
                ORDER BY MIN(a.embedding <=> %(emb)s::vector) + (AVG(a.embedding) <=> %(emb)s::vector) ASC
                LIMIT 1
                """,
                {
                    "emb": embedding, "t": discovered_at, "h": window_hours,
                    "max_nearest": max_nearest_distance, "max_centroid": max_centroid_distance,
                },
            )
            row = cur.fetchone()
            return row["story_id"] if row else None


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
    """Stories that gained a member this run - new stories and ones that grew.
    tag-relevance's worklist: every touched story gets classified, regardless
    of relevance (relevance isn't known yet)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT story_id FROM story_articles WHERE scrape_run_id = %s",
                (scrape_run_id,),
            )
            return [row["story_id"] for row in cur.fetchall()]


def get_touched_relevant_story_ids(scrape_run_id: int) -> list[int]:
    """Touched stories that tag-relevance has since marked relevant (member
    articles' is_relevant = TRUE) - used by finalize to report the
    watchlist-relevant subset rather than every story from every source."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT sa.story_id
                FROM story_articles sa
                JOIN articles a ON a.id = sa.article_id
                WHERE sa.scrape_run_id = %s AND a.is_relevant = TRUE
                """,
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


def get_relevant_articles_missing_entities() -> list[dict]:
    """Every watchlist-relevant article article-entities hasn't checked yet
    - article-entities' worklist (one-shot per article, unlike
    summarize-story which redoes the whole story). Returns
    [{article_id}, ...].

    Keys off articles.entities_checked_at IS NULL, not "no article_entities
    rows exist" - those aren't the same thing. A NOT-EXISTS-on-rows version
    of this query can't tell "never processed" apart from "processed, found
    zero entities" (common for short articles), so it kept reprocessing the
    same zero-entity articles forever - see insert_article_entities's
    docstring for the fix.

    Also gap-based rather than scoped to "this run's new story
    memberships" the way this used to be - that scoping assumed relevance
    is only ever established at discovery time, which broke in practice:
    reclassifying the whole watchlist after adding a topic or fixing a
    classification label (see nlp.tagging's docstring for why that's
    sometimes necessary) can retroactively flip is_relevant on an article
    from a much older run, and the old run-scoped query would never pick
    that article up - it silently stayed without NER forever. This version
    is self-healing regardless of how or when relevance was decided."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT a.id AS article_id FROM articles a WHERE a.is_relevant = TRUE AND a.entities_checked_at IS NULL"
            )
            return [dict(row) for row in cur.fetchall()]


def get_relevant_articles_missing_sentiment() -> list[dict]:
    """Every watchlist-relevant article without an article_sentiment row yet
    - sentiment-article's worklist. Gap-based for the same reason as
    get_relevant_articles_missing_entities() above. Returns
    [{article_id, story_id}, ...]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id AS article_id, sa.story_id FROM articles a
                JOIN story_articles sa ON sa.article_id = a.id
                WHERE a.is_relevant = TRUE
                AND NOT EXISTS (SELECT 1 FROM article_sentiment asent WHERE asent.article_id = a.id)
                """
            )
            return [dict(row) for row in cur.fetchall()]


def get_story_articles(story_id: int) -> list[dict]:
    """Every member article of a story, across all runs - summarize-story
    regenerates from the complete set, not just this run's delta."""
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


def get_story_summary_text(story_id: int) -> Optional[str]:
    """tag-relevance classifies this - the short summary, not the raw
    combined article text - see nlp/tagging.py for why."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT summary_text FROM story_summaries WHERE story_id = %s", (story_id,)
            )
            row = cur.fetchone()
            return row["summary_text"] if row else None


# ---------------------------------------------------------------------------
# article_sentiment
# ---------------------------------------------------------------------------

def upsert_article_sentiment(
    article_id: int,
    story_id: int,
    label: str,
    score: float,
    model_name: str,
    emotion_label: Optional[str] = None,
    emotion_score: Optional[float] = None,
    emotion_model_name: Optional[str] = None,
) -> None:
    """emotion_* is optional - sentiment_article() always passes it (both
    classifiers run in the same stage), but callers that only care about
    sentiment (tests, one-off backfills) aren't forced to thread emotion
    data through for no reason."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO article_sentiment
                    (article_id, story_id, sentiment_label, sentiment_score, model_name,
                     emotion_label, emotion_score, emotion_model_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (article_id) DO UPDATE
                    SET sentiment_label = EXCLUDED.sentiment_label,
                        sentiment_score = EXCLUDED.sentiment_score,
                        model_name = EXCLUDED.model_name,
                        emotion_label = EXCLUDED.emotion_label,
                        emotion_score = EXCLUDED.emotion_score,
                        emotion_model_name = EXCLUDED.emotion_model_name
                """,
                (article_id, story_id, label, score, model_name, emotion_label, emotion_score, emotion_model_name),
            )


# ---------------------------------------------------------------------------
# article_entities - one-shot NER per article (an article's own text never
# changes once fetched, so there's nothing to redo for it)
# ---------------------------------------------------------------------------

def delete_article_entities(article_id: int) -> None:
    """Makes article-entities safe to retry without duplicating rows - see
    sentiment_article()/upsert_article_sentiment for why this stage's
    worklist (relevant articles missing entities) can hand back an article
    it already processed on a retried task."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM article_entities WHERE article_id = %s", (article_id,))


def insert_article_entities(article_id: int, entities: Iterable[dict]) -> None:
    """entities: iterable of {text, normalized_text, label, start_char, end_char}
    (nlp.ner.extract_entities' return shape).

    Always stamps articles.entities_checked_at, even when `entities` is
    empty - that's the signal article-entities' worklist actually keys off
    (see get_relevant_articles_missing_entities), precisely so a genuinely
    zero-entity article doesn't look indistinguishable from "never
    checked" and get reprocessed every run forever."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            for ent in entities:
                cur.execute(
                    """
                    INSERT INTO article_entities
                        (article_id, entity_text, normalized_text, entity_label, start_char, end_char)
                    VALUES (%s, %s, %s, %s, %s, %s)
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
            cur.execute("UPDATE articles SET entities_checked_at = now() WHERE id = %s", (article_id,))


def get_article_entities(article_id: int) -> dict:
    """People/organizations/locations significant to one article, for the
    /stories page - PERSON/ORG/{GPE,LOC} respectively (spaCy's other labels -
    DATE, MONEY, CARDINAL, etc. - aren't shown). "Significant" filters out
    one-off noise like a byline or an incidental aside - see
    nlp.matching.filter_significant_entities for exactly what's kept and why."""
    from nlp.matching import filter_significant_entities

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT entity_text, entity_label, normalized_text, start_char FROM article_entities
                WHERE article_id = %s AND entity_label IN ('PERSON', 'ORG', 'GPE', 'LOC')
                ORDER BY start_char
                """,
                (article_id,),
            )
            rows = cur.fetchall()
    buckets = {"people": [], "organizations": [], "locations": []}
    for row in rows:
        ent = {"text": row["entity_text"], "normalized_text": row["normalized_text"], "start_char": row["start_char"]}
        if row["entity_label"] == "PERSON":
            buckets["people"].append(ent)
        elif row["entity_label"] == "ORG":
            buckets["organizations"].append(ent)
        else:  # GPE or LOC
            buckets["locations"].append(ent)
    return {category: filter_significant_entities(entities) for category, entities in buckets.items()}


# ---------------------------------------------------------------------------
# article_entity_sentiment - aspect-based sentiment per significant entity
# (see nlp/entity_sentiment.py), one-shot per article like article_entities
# ---------------------------------------------------------------------------

def get_relevant_articles_missing_entity_sentiment() -> list[dict]:
    """Every watchlist-relevant article entity-sentiment hasn't checked yet.
    Gap-based on entity_sentiment_checked_at, not on whether
    article_entity_sentiment rows exist - same reasoning as
    get_relevant_articles_missing_entities(), since an article with zero
    significant entities would otherwise look permanently unprocessed.
    Returns [{article_id}, ...]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT a.id AS article_id FROM articles a "
                "WHERE a.is_relevant = TRUE AND a.entity_sentiment_checked_at IS NULL"
            )
            return [dict(row) for row in cur.fetchall()]


def delete_article_entity_sentiment(article_id: int) -> None:
    """Makes entity-sentiment safe to retry without duplicating rows - same
    reasoning as delete_article_entities()."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM article_entity_sentiment WHERE article_id = %s", (article_id,))


def insert_article_entity_sentiment(article_id: int, scored_entities: Iterable[dict], model_name: str) -> None:
    """scored_entities: iterable of {entity_text, entity_category, sentiment_label,
    sentiment_score} - entity_category is one of people/organizations/locations
    (get_article_entities()'s bucket names). Always stamps
    articles.entity_sentiment_checked_at, even when scored_entities is empty -
    see get_relevant_articles_missing_entity_sentiment()'s docstring for why."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            for ent in scored_entities:
                cur.execute(
                    """
                    INSERT INTO article_entity_sentiment
                        (article_id, entity_text, entity_category, sentiment_label, sentiment_score, model_name)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id, entity_category, entity_text) DO UPDATE
                        SET sentiment_label = EXCLUDED.sentiment_label,
                            sentiment_score = EXCLUDED.sentiment_score,
                            model_name = EXCLUDED.model_name
                    """,
                    (
                        article_id,
                        ent["entity_text"],
                        ent["entity_category"],
                        ent["sentiment_label"],
                        ent["sentiment_score"],
                        model_name,
                    ),
                )
            cur.execute("UPDATE articles SET entity_sentiment_checked_at = now() WHERE id = %s", (article_id,))


def get_article_entity_sentiment(article_id: int) -> dict:
    """{(category, entity_text_lower): {"label": ..., "score": ...}} for one
    article - a lookup the /stories page uses to annotate each entity tag
    with its own sentiment, keyed case-insensitively since display text
    elsewhere (get_article_entities) is matched the same way."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT entity_text, entity_category, sentiment_label, sentiment_score
                FROM article_entity_sentiment WHERE article_id = %s
                """,
                (article_id,),
            )
            rows = cur.fetchall()
    return {
        (row["entity_category"], row["entity_text"].lower()): {
            "label": row["sentiment_label"],
            "score": row["sentiment_score"],
        }
        for row in rows
    }


# ---------------------------------------------------------------------------
# story viewer (web page)
# ---------------------------------------------------------------------------

def get_recent_stories_with_summaries(
    limit: int = 50,
    topic: Optional[str] = None,
    date: Optional[str] = None,
    q: Optional[str] = None,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Watchlist-relevant stories with a summary, newest-updated first, for
    the web viewer. Returns (page_of_stories, total_matching_count) - the
    count is a COUNT(*) OVER() window, computed from the same query in one
    round trip rather than a separate count query, and reflects every
    matching story (before LIMIT/OFFSET), for the pagination controls.

    The HAVING clause is load-bearing, not cosmetic: every touched story
    gets a summary now (tag-relevance runs after summarize-story, not
    before - see run_batch.py), so without it this would list every
    clustered story from every source, relevant or not, rather than just
    the ones that matched something on the watchlist.

    topic: an exact watchlist_entities.canonical_name to filter to (the
    /stories page's topic filter) - applied as a WHERE/EXISTS check, not a
    HAVING on the aggregated matched_entities array, so it narrows which
    stories are considered (and therefore which hit the LIMIT) rather than
    just which rows get filtered out of an already-limited set.

    date: an exact 'YYYY-MM-DD' string to filter to stories last updated on
    that calendar day (the /stories page's calendar sidepane) - before this
    existed, a day where tag-relevance touched 50+ stories would silently
    push every older story off the single unpaginated page with no way to
    reach them; see the incident that motivated adding this.

    q: a free-text search term, matched case-insensitively (ILIKE) against
    the story's summary OR any member article's headline - a plain
    substring scan, not full-text search (ts_vector/GIN), since this is a
    few thousand rows in a home-lab DB, not a search-engine-scale corpus."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    s.id AS story_id, s.updated_at, ss.summary_text,
                    array_agg(DISTINCT a.source) AS sources,
                    array_agg(DISTINCT we.canonical_name) AS matched_entities,
                    sl.location_text, (sl.lat IS NOT NULL) AS located,
                    count(*) OVER() AS total_count
                FROM stories s
                JOIN story_summaries ss ON ss.story_id = s.id
                LEFT JOIN story_locations sl ON sl.story_id = s.id
                JOIN story_articles sa ON sa.story_id = s.id
                JOIN articles a ON a.id = sa.article_id
                LEFT JOIN watchlist_matches wm ON wm.article_id = a.id
                LEFT JOIN watchlist_entities we ON we.id = wm.watchlist_entity_id
                WHERE (%(topic)s IS NULL OR EXISTS (
                    SELECT 1 FROM story_articles sa2
                    JOIN watchlist_matches wm2 ON wm2.article_id = sa2.article_id
                    JOIN watchlist_entities we2 ON we2.id = wm2.watchlist_entity_id
                    WHERE sa2.story_id = s.id AND we2.canonical_name = %(topic)s
                ))
                AND (%(date)s IS NULL OR s.updated_at::date = %(date)s::date)
                AND (%(q)s IS NULL OR ss.summary_text ILIKE '%%' || %(q)s || '%%' OR EXISTS (
                    SELECT 1 FROM story_articles sa3
                    JOIN articles a3 ON a3.id = sa3.article_id
                    WHERE sa3.story_id = s.id AND a3.headline ILIKE '%%' || %(q)s || '%%'
                ))
                GROUP BY s.id, s.updated_at, ss.summary_text, sl.location_text, sl.lat
                HAVING count(wm.id) > 0
                ORDER BY s.updated_at DESC
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                {"topic": topic, "date": date, "q": q, "limit": limit, "offset": offset},
            )
            rows = [dict(row) for row in cur.fetchall()]
    total_count = rows[0]["total_count"] if rows else 0
    for row in rows:
        del row["total_count"]
    return rows, total_count


def get_story_dates(topic: Optional[str] = None, q: Optional[str] = None) -> list[dict]:
    """Every calendar day (by story updated_at) with at least one
    displayable relevant story, with a per-day count - powers the
    /stories page's calendar sidepane (which days are clickable, and their
    counts). Scoped to the current topic/search filters if active,
    consistent with the topic pills' own counts. Returns
    [{"day": date, "story_count": n}, ...]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.updated_at::date AS day, count(DISTINCT s.id) AS story_count
                FROM stories s
                JOIN story_summaries ss ON ss.story_id = s.id
                JOIN story_articles sa ON sa.story_id = s.id
                JOIN articles a ON a.id = sa.article_id
                JOIN watchlist_matches wm ON wm.article_id = a.id
                JOIN watchlist_entities we ON we.id = wm.watchlist_entity_id
                WHERE (%(topic)s IS NULL OR we.canonical_name = %(topic)s)
                AND (%(q)s IS NULL OR ss.summary_text ILIKE '%%' || %(q)s || '%%' OR EXISTS (
                    SELECT 1 FROM story_articles sa3
                    JOIN articles a3 ON a3.id = sa3.article_id
                    WHERE sa3.story_id = s.id AND a3.headline ILIKE '%%' || %(q)s || '%%'
                ))
                GROUP BY s.updated_at::date
                ORDER BY day
                """,
                {"topic": topic, "q": q},
            )
            return [dict(row) for row in cur.fetchall()]


def get_topic_story_counts(date: Optional[str] = None, q: Optional[str] = None) -> list[dict]:
    """Every active watchlist topic with how many displayable stories (has a
    summary, counted once even if several of its member articles matched)
    currently match it - the /stories page's filter-pill list and counts.
    LEFT JOINed from watchlist_entities, not an INNER JOIN from
    watchlist_matches, so a topic with zero matches so far still appears
    (with story_count 0) rather than silently disappearing as a filter
    option. Queried live on every request - any watchlist entity added via
    the API shows up here (and so on the page) on the very next load, no
    caching or redeploy involved.

    date/q: the same calendar-day and free-text filters as
    get_recent_stories_with_summaries() - when active, counts reflect only
    stories matching them, so the pills stay consistent with what's
    actually on screen rather than showing stale all-time totals. Applied
    in the JOIN conditions, not WHERE - putting them in WHERE would turn
    the LEFT JOIN into a de facto INNER JOIN (a topic with zero stories
    under the filter would vanish instead of showing "(0)")."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT we.canonical_name, count(DISTINCT ss.story_id) AS story_count
                FROM watchlist_entities we
                LEFT JOIN watchlist_matches wm ON wm.watchlist_entity_id = we.id
                LEFT JOIN story_articles sa ON sa.article_id = wm.article_id
                LEFT JOIN stories s ON s.id = sa.story_id
                    AND (%(date)s IS NULL OR s.updated_at::date = %(date)s::date)
                LEFT JOIN story_summaries ss ON ss.story_id = s.id
                    AND (%(q)s IS NULL OR ss.summary_text ILIKE '%%' || %(q)s || '%%' OR EXISTS (
                        SELECT 1 FROM story_articles sa3
                        JOIN articles a3 ON a3.id = sa3.article_id
                        WHERE sa3.story_id = s.id AND a3.headline ILIKE '%%' || %(q)s || '%%'
                    ))
                WHERE we.is_active = TRUE
                GROUP BY we.id, we.canonical_name
                ORDER BY lower(we.canonical_name)
                """,
                {"date": date, "q": q},
            )
            return [dict(row) for row in cur.fetchall()]


def get_story_source_articles(story_id: int) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id, a.source, a.url, a.headline,
                    asent.sentiment_label, asent.sentiment_score, asent.emotion_label, asent.emotion_score
                FROM articles a
                JOIN story_articles sa ON sa.article_id = a.id
                LEFT JOIN article_sentiment asent ON asent.article_id = a.id
                WHERE sa.story_id = %s
                ORDER BY a.discovered_at
                """,
                (story_id,),
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# geolocation / map view
# ---------------------------------------------------------------------------

def get_relevant_stories_missing_location() -> list[int]:
    """Relevant stories geolocate-story hasn't checked yet - its worklist.
    Gap-based (a story_locations row, even an empty one, marks it done)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT sa.story_id
                FROM story_articles sa
                JOIN articles a ON a.id = sa.article_id
                WHERE a.is_relevant = TRUE
                AND NOT EXISTS (SELECT 1 FROM story_locations sl WHERE sl.story_id = sa.story_id)
                """
            )
            return [row["story_id"] for row in cur.fetchall()]


def get_story_place_candidates(story_id: int) -> list[dict]:
    """Every place (GPE/LOC/FAC) NER found in a story's articles, with the
    counts nlp.placeselect needs. No model runs here. Returns
    [{text, key, significant_articles, mentions, person_mentions,
    org_mentions, headline_hits}, ...] ordered by significance + headline hits, strongest
    first.

    significant_articles: how many member articles pass
    nlp.matching.filter_significant_entities for this place (see
    get_article_entities). person_mentions / org_mentions: how many PERSON /
    ORG entities in the story contain this place's text as a word - only
    for one-word names ("Kabila" in "Joseph Kabila", "Sydney" in "Sydney
    Swans"), 0 otherwise. nlp.placeselect drops on the first and only
    penalises on the second."""
    import re

    from nlp.placeselect import place_key

    articles = get_story_articles(story_id)
    ids = [a["id"] for a in articles]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT entity_text, entity_label FROM article_entities WHERE article_id = ANY(%s)",
                (ids,),
            )
            rows = cur.fetchall()
    places: dict[str, dict] = {}
    person_words: dict[str, int] = {}
    org_words: dict[str, int] = {}
    for row in rows:
        text, label = row["entity_text"], row["entity_label"]
        if label in ("GPE", "LOC", "FAC"):
            p = places.setdefault(place_key(text), {"text": text, "mentions": 0, "significant_articles": 0})
            p["mentions"] += 1
        if label in ("PERSON", "ORG"):
            bucket = person_words if label == "PERSON" else org_words
            for word in re.findall(r"[a-z]{3,}", text.lower()):
                bucket[word] = bucket.get(word, 0) + 1
    for article in articles:
        for text in get_article_entities(article["id"])["locations"]:
            p = places.setdefault(place_key(text), {"text": text, "mentions": 0, "significant_articles": 0})
            p["significant_articles"] += 1
    headlines = " ".join(a["headline"] for a in articles).lower()
    out = []
    for key, p in places.items():
        words = re.findall(r"[a-z]{3,}", key)
        out.append({
            "key": key, "text": p["text"],
            "significant_articles": p["significant_articles"], "mentions": p["mentions"],
            "person_mentions": person_words.get(words[0], 0) if len(words) == 1 else 0,
            "org_mentions": org_words.get(words[0], 0) if len(words) == 1 else 0,
            "headline_hits": len(re.findall(r"(?<![a-z])" + re.escape(key) + r"(?![a-z])", headlines)),
        })
    # only places the pipeline's own significance filter or the headline vouches for
    out = [c for c in out if c["significant_articles"] > 0 or c["headline_hits"] > 0]
    out.sort(key=lambda c: -(c["significant_articles"] + c["headline_hits"]))
    return out


def get_cached_geocode(query: str) -> Optional[dict]:
    """None if never looked up; otherwise {lat, lon, display_name}, with
    lat/lon None for a cached "not found"."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT lat, lon, display_name, place_rank, radius_km FROM geocode_cache WHERE query = %s", (query.lower(),)
            )
            row = cur.fetchone()
            return dict(row) if row else None


def set_cached_geocode(query: str, result: Optional[dict]) -> None:
    result = result or {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO geocode_cache (query, lat, lon, display_name, place_rank, radius_km)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (query) DO UPDATE
                SET lat = EXCLUDED.lat, lon = EXCLUDED.lon, display_name = EXCLUDED.display_name,
                    place_rank = EXCLUDED.place_rank, radius_km = EXCLUDED.radius_km
                """,
                (query.lower(), result.get("lat"), result.get("lon"), result.get("display_name"),
                 result.get("place_rank"), result.get("radius_km")),
            )


def upsert_story_location(
    story_id: int, location_text: Optional[str], lat: Optional[float], lon: Optional[float],
    precise: Optional[bool] = None, radius_km: Optional[float] = None,
    confidence: Optional[float] = None,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO story_locations (story_id, location_text, lat, lon, precise, radius_km, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (story_id) DO UPDATE
                SET location_text = EXCLUDED.location_text, lat = EXCLUDED.lat, lon = EXCLUDED.lon,
                    precise = EXCLUDED.precise, radius_km = EXCLUDED.radius_km,
                    confidence = EXCLUDED.confidence
                """,
                (story_id, location_text, lat, lon, precise, radius_km, confidence),
            )


def get_map_stories(date: Optional[str] = None) -> list[dict]:
    """Located, relevant, summarized stories (optionally last updated on one
    'YYYY-MM-DD' day) for the /map page. [{story_id, summary_text,
    location_text, lat, lon, updated_at, topics, headline}, ...] - headline
    is the story's earliest-discovered article's."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id AS story_id, s.updated_at, ss.summary_text,
                       sl.location_text, sl.lat, sl.lon, sl.precise, sl.radius_km,
                       (SELECT a.headline FROM story_articles sa
                        JOIN articles a ON a.id = sa.article_id
                        WHERE sa.story_id = s.id ORDER BY a.discovered_at LIMIT 1) AS headline,
                       ARRAY(SELECT DISTINCT we.canonical_name FROM story_articles sa
                             JOIN watchlist_matches wm ON wm.article_id = sa.article_id
                             JOIN watchlist_entities we ON we.id = wm.watchlist_entity_id
                             WHERE sa.story_id = s.id) AS topics
                FROM stories s
                JOIN story_summaries ss ON ss.story_id = s.id
                JOIN story_locations sl ON sl.story_id = s.id
                WHERE sl.lat IS NOT NULL AND sl.lon IS NOT NULL
                AND EXISTS (SELECT 1 FROM story_articles sa
                            JOIN watchlist_matches wm ON wm.article_id = sa.article_id
                            WHERE sa.story_id = s.id)
                AND (%(date)s IS NULL OR s.updated_at::date = %(date)s::date)
                ORDER BY s.updated_at DESC
                """,
                {"date": date},
            )
            return [dict(row) for row in cur.fetchall()]


def get_latest_map_date():
    """Most recent day with a mappable story, or None."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT max(s.updated_at::date) AS day FROM stories s
                JOIN story_locations sl ON sl.story_id = s.id
                WHERE sl.lat IS NOT NULL
                """
            )
            return cur.fetchone()["day"]
