CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS watchlist_entities (
    id              SERIAL PRIMARY KEY,
    canonical_name  TEXT NOT NULL UNIQUE,
    entity_type     TEXT,
    notes           TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS watchlist_aliases (
    id                   SERIAL PRIMARY KEY,
    watchlist_entity_id  INTEGER NOT NULL REFERENCES watchlist_entities(id) ON DELETE CASCADE,
    alias                TEXT NOT NULL,
    normalized_alias     TEXT NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (watchlist_entity_id, normalized_alias)
);
CREATE INDEX IF NOT EXISTS idx_aliases_normalized ON watchlist_aliases(normalized_alias);

CREATE TABLE IF NOT EXISTS scrape_runs (
    id                     SERIAL PRIMARY KEY,
    started_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at            TIMESTAMPTZ,
    status                 TEXT NOT NULL DEFAULT 'running'
                           CHECK (status IN ('running', 'success', 'partial', 'failed')),
    front_pages_scraped    INTEGER DEFAULT 0,
    front_pages_failed     INTEGER DEFAULT 0,
    articles_found         INTEGER DEFAULT 0,
    articles_fetched       INTEGER DEFAULT 0,
    articles_matched       INTEGER DEFAULT 0,
    stories_touched        INTEGER DEFAULT 0,
    summaries_generated    INTEGER DEFAULT 0,
    error_message          TEXT
);

-- Each source's `discover` task increments these fields directly and
-- concurrently (one fan-out task per source) - updates must be atomic
-- (SET x = x + %s), never read-modify-write, or concurrent sources would
-- clobber each other's counts.

CREATE TABLE IF NOT EXISTS articles (
    id                 SERIAL PRIMARY KEY,
    source             TEXT NOT NULL,
    url                TEXT NOT NULL UNIQUE,
    headline           TEXT NOT NULL,
    front_page_slug    TEXT,
    body_text          TEXT,
    fetch_status       TEXT NOT NULL DEFAULT 'pending'
                       CHECK (fetch_status IN ('pending', 'fetched', 'failed')),
    fetch_error        TEXT,
    -- filled in by filter-article: does this article match a topic of interest at all?
    is_relevant        BOOLEAN,
    -- filled in by embed-article; all-MiniLM-L6-v2 (see nlp/embedding.py)
    embedding          vector(384),
    first_seen_run_id  INTEGER REFERENCES scrape_runs(id),
    last_seen_run_id   INTEGER REFERENCES scrape_runs(id),
    discovered_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    fetched_at         TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
-- HNSW supports incremental inserts well, unlike IVFFlat (which wants a
-- representative sample up front) - a better fit for a table that grows by
-- a few hundred rows a day rather than being bulk-loaded once.
CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS article_entities (
    id               SERIAL PRIMARY KEY,
    article_id       INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    entity_text      TEXT NOT NULL,
    normalized_text  TEXT NOT NULL,
    entity_label     TEXT NOT NULL,
    start_char       INTEGER,
    end_char         INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_article_entities_article ON article_entities(article_id);
CREATE INDEX IF NOT EXISTS idx_article_entities_normalized ON article_entities(normalized_text);

CREATE TABLE IF NOT EXISTS watchlist_matches (
    id                    SERIAL PRIMARY KEY,
    article_id            INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    watchlist_entity_id   INTEGER NOT NULL REFERENCES watchlist_entities(id) ON DELETE CASCADE,
    article_entity_id     INTEGER REFERENCES article_entities(id),
    matched_alias         TEXT NOT NULL,
    scrape_run_id         INTEGER REFERENCES scrape_runs(id),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, watchlist_entity_id)
);

-- A story is a cluster of articles (possibly from different sources, possibly
-- discovered on different days) judged to be reporting on the same underlying
-- event. cluster-stories compares each newly-embedded relevant article
-- against stories touched within a rolling window (see MYOPIC_STORY_WINDOW_DAYS)
-- and either attaches it to an existing story or starts a new one.
CREATE TABLE IF NOT EXISTS stories (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS story_articles (
    id             SERIAL PRIMARY KEY,
    story_id       INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    article_id     INTEGER NOT NULL UNIQUE REFERENCES articles(id) ON DELETE CASCADE,
    -- which run added this member - lets downstream stages ask "which
    -- stories did THIS run touch" without threading an id list through Argo
    scrape_run_id  INTEGER NOT NULL REFERENCES scrape_runs(id),
    added_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_story_articles_story ON story_articles(story_id);
CREATE INDEX IF NOT EXISTS idx_story_articles_run ON story_articles(scrape_run_id);

-- Regenerated (not appended) every time a story gains a new member - the
-- summary is written from every source article the story has, not just the
-- new one, so it's an UPSERT keyed on story_id rather than a growing log.
CREATE TABLE IF NOT EXISTS story_summaries (
    id            SERIAL PRIMARY KEY,
    story_id      INTEGER NOT NULL UNIQUE REFERENCES stories(id) ON DELETE CASCADE,
    summary_text  TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Also regenerated per touched story (deleted + reinserted, like
-- article_entities), from a fresh NER pass over every source article's
-- combined text - deliberately not just an aggregation of article_entities,
-- since combining sources can surface entities no single article emphasized.
CREATE TABLE IF NOT EXISTS story_entities (
    id               SERIAL PRIMARY KEY,
    story_id         INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    entity_text      TEXT NOT NULL,
    normalized_text  TEXT NOT NULL,
    entity_label     TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_story_entities_story ON story_entities(story_id);

-- One-shot per article (not regenerated when a story grows) - this is each
-- source's own framing of the story, so an existing member's sentiment
-- doesn't change just because another source joined the cluster.
CREATE TABLE IF NOT EXISTS article_sentiment (
    id               SERIAL PRIMARY KEY,
    article_id       INTEGER NOT NULL UNIQUE REFERENCES articles(id) ON DELETE CASCADE,
    story_id         INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    sentiment_label  TEXT NOT NULL,
    sentiment_score  REAL,
    model_name       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_article_sentiment_story ON article_sentiment(story_id);
