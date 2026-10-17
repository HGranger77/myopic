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
    -- translate-article's originals, for a non-English source (see
    -- config/sources.yaml's per-source `language`) - NULL for English
    -- sources, where headline/body_text already ARE the original. Kept
    -- separately rather than overwritten so the original text isn't lost -
    -- every English-only stage downstream (embed, summarize, sentiment,
    -- NER, tag-relevance) reads headline/body_text either way, translated
    -- or not, and doesn't need to know the difference.
    original_headline  TEXT,
    original_body_text TEXT,
    -- filled in by tag-relevance: does this article's story match a topic of interest at all?
    is_relevant        BOOLEAN,
    -- filled in by embed-article; all-MiniLM-L6-v2 (see nlp/embedding.py)
    embedding          vector(384),
    -- Stamped by article-entities whenever it runs NER on this article,
    -- regardless of how many entities (if any) were found - NOT the same
    -- as "has article_entities rows". article-entities' worklist is
    -- "relevant AND entities_checked_at IS NULL", not "relevant AND no
    -- article_entities rows exist": a NOT-EXISTS-on-article_entities check
    -- can't tell "never processed" apart from "processed, found zero
    -- entities" (common for short articles, and for every untranslated
    -- Chinese article NER was mistakenly run on before translate-article
    -- gated it - see get_articles_needing_embedding's docstring), so that
    -- version kept reprocessing the same zero-entity articles every run.
    entities_checked_at TIMESTAMPTZ,
    -- Same gap-tracking fix as entities_checked_at above, for
    -- entity-sentiment's worklist - an article can legitimately have zero
    -- significant entities (and so zero article_entity_sentiment rows)
    -- without that meaning "never checked".
    entity_sentiment_checked_at TIMESTAMPTZ,
    first_seen_run_id  INTEGER REFERENCES scrape_runs(id),
    last_seen_run_id   INTEGER REFERENCES scrape_runs(id),
    discovered_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    fetched_at         TIMESTAMPTZ
);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS entity_sentiment_checked_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
-- HNSW supports incremental inserts well, unlike IVFFlat (which wants a
-- representative sample up front) - a better fit for a table that grows by
-- a few hundred rows a day rather than being bulk-loaded once.
CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles
    USING hnsw (embedding vector_cosine_ops);

-- Filled in once per article (like article_sentiment, not redone on every
-- run) by the article-entities stage - a fresh NER pass over just that one
-- article's own text. Powers the per-source entity breakdown on the
-- /stories page.
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

-- One row per SIGNIFICANT entity per article (see
-- nlp.matching.filter_significant_entities - the same already-filtered set
-- the /stories page shows as entity tags), not every raw NER span in
-- article_entities above. Aspect-based sentiment (nlp/entity_sentiment.py)
-- scores how the article frames that specific entity, separate from
-- article_sentiment's one whole-article read.
CREATE TABLE IF NOT EXISTS article_entity_sentiment (
    id               SERIAL PRIMARY KEY,
    article_id       INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    entity_text      TEXT NOT NULL,
    entity_category  TEXT NOT NULL CHECK (entity_category IN ('people', 'organizations', 'locations')),
    sentiment_label  TEXT NOT NULL,
    sentiment_score  REAL,
    model_name       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, entity_category, entity_text)
);
CREATE INDEX IF NOT EXISTS idx_article_entity_sentiment_article ON article_entity_sentiment(article_id);

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

-- One-shot per article (not regenerated when a story grows) - this is each
-- source's own framing of the story, so an existing member's sentiment
-- doesn't change just because another source joined the cluster.
CREATE TABLE IF NOT EXISTS article_sentiment (
    id                  SERIAL PRIMARY KEY,
    article_id          INTEGER NOT NULL UNIQUE REFERENCES articles(id) ON DELETE CASCADE,
    story_id            INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    sentiment_label     TEXT NOT NULL,
    sentiment_score     REAL,
    model_name          TEXT NOT NULL,
    -- Filled in alongside sentiment, same stage/pod, same worklist - a
    -- second, finer-grained read (anger/disgust/fear/joy/neutral/sadness/
    -- surprise - see nlp/emotion.py) shown next to the POSITIVE/NEGATIVE
    -- badge, not instead of it. Nullable: rows written before this was
    -- added have no emotion data and that's fine, nothing backfills them
    -- automatically.
    emotion_label       TEXT,
    emotion_score       REAL,
    emotion_model_name  TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- CREATE TABLE IF NOT EXISTS doesn't add columns to a table that already
-- exists, so databases created before emotion was added need these.
ALTER TABLE article_sentiment ADD COLUMN IF NOT EXISTS emotion_label TEXT;
ALTER TABLE article_sentiment ADD COLUMN IF NOT EXISTS emotion_score REAL;
ALTER TABLE article_sentiment ADD COLUMN IF NOT EXISTS emotion_model_name TEXT;
CREATE INDEX IF NOT EXISTS idx_article_sentiment_story ON article_sentiment(story_id);

-- Cache of place-name -> coordinates lookups (nlp/geocode.py), keyed by the
-- normalized query text. Negative results (lat/lon NULL) are cached too, so
-- a place the geocoder can't resolve isn't re-queried on every run.
CREATE TABLE IF NOT EXISTS geocode_cache (
    id            SERIAL PRIMARY KEY,
    query         TEXT NOT NULL UNIQUE,
    lat           DOUBLE PRECISION,
    lon           DOUBLE PRECISION,
    display_name  TEXT,
    place_rank    INTEGER,
    radius_km     DOUBLE PRECISION,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE geocode_cache ADD COLUMN IF NOT EXISTS place_rank INTEGER;
ALTER TABLE geocode_cache ADD COLUMN IF NOT EXISTS radius_km DOUBLE PRECISION;

-- Where a story took place - one row per story, stamped by the
-- geolocate-story stage even when no location could be found (lat/lon
-- NULL), so "never checked" and "checked, nothing found" are distinguishable
-- (same reasoning as articles.entities_checked_at). Powers the /map page.
CREATE TABLE IF NOT EXISTS story_locations (
    id             SERIAL PRIMARY KEY,
    story_id       INTEGER NOT NULL UNIQUE REFERENCES stories(id) ON DELETE CASCADE,
    location_text  TEXT,
    lat            DOUBLE PRECISION,
    lon            DOUBLE PRECISION,
    -- precise: a specific settlement; otherwise the map shows an approximate
    -- area of radius_km (a country, state or continent)
    precise        BOOLEAN,
    radius_km      DOUBLE PRECISION,
    -- the NLI model's score that the story happened here (nlp/placeselect.py)
    confidence     REAL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE story_locations ADD COLUMN IF NOT EXISTS confidence REAL;
ALTER TABLE story_locations ADD COLUMN IF NOT EXISTS precise BOOLEAN;
ALTER TABLE story_locations ADD COLUMN IF NOT EXISTS radius_km DOUBLE PRECISION;
