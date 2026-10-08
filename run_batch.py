"""Batch pipeline, one Argo task per DAG stage - each loads its model (if any)
once and loops over everything that stage owns, rather than fanning out one
pod per article. The only fan-out left is `discover`, one task per source.

Every stage after `start-run` takes only --run-id; nothing else threads
through Argo parameters. Each stage finds its own worklist by querying
Postgres directly (e.g. "fetched articles not yet filtered"), so there's no
explicit article/story-id list to pass between tasks.
"""
import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path

import yaml
from dotenv import load_dotenv

from database import operations as ops
from nlp.matching import build_watchlist_index, match_entities
from scraper import fetch
from scraper.registry import get_source

# nlp.ner/embedding/summarize/sentiment are NOT imported here - each pulls in
# a heavy library (spacy, sentence_transformers, torch/transformers) that a
# stage without a model (start-run, discover, fetch-article-bodies,
# cluster-stories, finalize) would otherwise pay for at import time
# regardless of needing it. Each model-bearing function below imports its
# own module locally instead, right where it's used.

CONFIG_PATH = Path(__file__).parent / "config" / "sources.yaml"

STORY_WINDOW_DAYS = int(os.environ.get("MYOPIC_STORY_WINDOW_DAYS", "14"))
STORY_MAX_DISTANCE = float(os.environ.get("MYOPIC_STORY_MAX_DISTANCE", "0.3"))
FETCH_CONCURRENCY = int(os.environ.get("MYOPIC_FETCH_CONCURRENCY", "8"))
MAX_STORY_TEXT_CHARS = 8000  # keeps a many-source story's combined text bounded


def load_sources() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _combined_story_text(articles: list[dict]) -> str:
    parts = [f"[{a['source']}] {a['headline']}\n{a['body_text']}" for a in articles]
    return "\n\n".join(parts)[:MAX_STORY_TEXT_CHARS]


# ---------------------------------------------------------------------------
# start-run
# ---------------------------------------------------------------------------

def start_run(out_dir: Path) -> int:
    run_id = ops.create_scrape_run()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_id").write_text(str(run_id))
    print(f"[start-run] created run {run_id}")
    return 0


# ---------------------------------------------------------------------------
# discover (fan-out per source)
# ---------------------------------------------------------------------------

def discover(run_id: int, source_module: str) -> int:
    sources = load_sources()["sources"]
    config = next((s for s in sources if s["module"] == source_module), None)
    if config is None:
        print(f"[discover] unknown source module {source_module!r}", file=sys.stderr)
        return 1

    source_fns = get_source(source_module)
    base_url = config["base_url"]
    front_pages_scraped = 0
    front_pages_failed = 0
    articles_found = 0

    for front_page in config["front_pages"]:
        url = base_url.rstrip("/") + front_page["path"]
        try:
            html = fetch.get(url)
        except Exception as exc:  # noqa: BLE001 - one bad front page shouldn't abort discovery
            print(f"[discover:{source_module}] failed to fetch front page {url}: {exc}", file=sys.stderr)
            front_pages_failed += 1
            continue
        front_pages_scraped += 1
        headlines = source_fns["extract_headlines"](html, base_url, front_page["slug"])
        articles_found += len(headlines)
        for item in headlines:
            ops.upsert_article(source_module, item["url"], item["headline"], item["section_slug"], run_id)

    ops.increment_scrape_run_counts(
        run_id,
        front_pages_scraped=front_pages_scraped,
        front_pages_failed=front_pages_failed,
        articles_found=articles_found,
    )
    print(f"[discover:{source_module}] {front_pages_scraped} front pages ok, "
          f"{front_pages_failed} failed, {articles_found} headlines found")
    return 0


# ---------------------------------------------------------------------------
# fetch-article-bodies (one task, concurrent fetch, no model)
# ---------------------------------------------------------------------------

def fetch_article_bodies() -> int:
    pending = ops.get_pending_articles()
    if not pending:
        print("[fetch-article-bodies] nothing pending")
        return 0

    by_url = {a["url"]: a for a in pending}
    results = asyncio.run(fetch.get_many(list(by_url), concurrency=FETCH_CONCURRENCY))

    fetched = 0
    failed = 0
    for url, article in by_url.items():
        html, error = results[url]
        if html is None:
            ops.mark_article_failed(article["id"], error or "fetch failed")
            failed += 1
            continue
        try:
            extract_body = get_source(article["source"])["extract_body"]
            body_text = extract_body(html, url)
            if not body_text:
                raise ValueError("no body text extracted")
            ops.mark_article_fetched(article["id"], body_text)
            fetched += 1
        except Exception as exc:  # noqa: BLE001 - one bad article shouldn't abort the batch
            ops.mark_article_failed(article["id"], str(exc))
            failed += 1

    print(f"[fetch-article-bodies] fetched {fetched}, failed {failed} (of {len(pending)} pending)")
    return 0


# ---------------------------------------------------------------------------
# filter-article (one task, NER model, pure CPU loop)
# ---------------------------------------------------------------------------

def filter_article(run_id: int) -> int:
    from nlp.ner import extract_entities

    watchlist = ops.get_active_watchlist_with_aliases()
    watchlist_index = build_watchlist_index(watchlist)

    to_filter = ops.get_articles_needing_filtering()
    relevant = 0
    for article in to_filter:
        ops.delete_article_entities(article["id"])
        entities = extract_entities(article["body_text"])
        ops.insert_article_entities(article["id"], entities)

        matches = []
        if watchlist_index:
            normalized_entities = [ent["normalized_text"] for ent in entities]
            matches = match_entities(normalized_entities, watchlist_index)
            entity_lookup = {ent["normalized_text"]: ent for ent in entities}
            for normalized_text, watchlist_entity_id, matched_alias in matches:
                article_entity = entity_lookup.get(normalized_text)
                ops.insert_watchlist_match(
                    article["id"], watchlist_entity_id,
                    article_entity.get("id") if article_entity else None,
                    matched_alias, run_id,
                )

        is_relevant = len(matches) > 0
        ops.set_article_relevance(article["id"], is_relevant)
        relevant += int(is_relevant)

    print(f"[filter-article] filtered {len(to_filter)}, {relevant} relevant")
    return 0


# ---------------------------------------------------------------------------
# embed-article (one task, embedding model, pure CPU loop)
# ---------------------------------------------------------------------------

def embed_article() -> int:
    from nlp.embedding import embed

    to_embed = ops.get_articles_needing_embedding()
    for article in to_embed:
        vector = embed(article["body_text"])
        ops.set_article_embedding(article["id"], vector)
    print(f"[embed-article] embedded {len(to_embed)} articles")
    return 0


# ---------------------------------------------------------------------------
# cluster-stories (one task, pgvector similarity, no model)
# ---------------------------------------------------------------------------

def cluster_stories(run_id: int) -> int:
    to_cluster = ops.get_articles_needing_clustering()
    new_stories = 0
    joined_stories = 0
    for article in to_cluster:
        story_id = ops.find_similar_story(article["embedding"], STORY_WINDOW_DAYS, STORY_MAX_DISTANCE)
        if story_id is None:
            story_id = ops.create_story()
            new_stories += 1
        else:
            joined_stories += 1
        ops.add_article_to_story(story_id, article["id"], run_id)

    print(f"[cluster-stories] {len(to_cluster)} articles clustered: "
          f"{new_stories} new stories, {joined_stories} joined existing ones")
    return 0


# ---------------------------------------------------------------------------
# summarize-story (one task, summarizer model, loop over touched stories)
# ---------------------------------------------------------------------------

def summarize_story(run_id: int) -> int:
    from nlp.summarize import MODEL_NAME as SUMMARIZER_MODEL_NAME
    from nlp.summarize import summarize

    touched = ops.get_touched_story_ids(run_id)
    done = 0
    for story_id in touched:
        try:
            articles = ops.get_story_articles(story_id)
            summary_text = summarize(_combined_story_text(articles))
            ops.upsert_story_summary(story_id, summary_text, SUMMARIZER_MODEL_NAME)
            done += 1
        except Exception as exc:  # noqa: BLE001 - one bad story shouldn't abort the rest
            print(f"[summarize-story] failed for story {story_id}: {exc}", file=sys.stderr)

    print(f"[summarize-story] summarized {done}/{len(touched)} touched stories")
    return 0


# ---------------------------------------------------------------------------
# sentiment-article (one task, sentiment model, loop over new memberships)
# ---------------------------------------------------------------------------

def sentiment_article(run_id: int) -> int:
    from nlp.sentiment import MODEL_NAME as SENTIMENT_MODEL_NAME
    from nlp.sentiment import analyze as analyze_sentiment

    new_memberships = ops.get_new_story_memberships_for_run(run_id)
    done = 0
    for membership in new_memberships:
        try:
            article = ops.get_article(membership["article_id"])
            label, score = analyze_sentiment(article["body_text"])
            ops.upsert_article_sentiment(
                article["id"], membership["story_id"], label, score, SENTIMENT_MODEL_NAME
            )
            done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[sentiment-article] failed for article {membership['article_id']}: {exc}", file=sys.stderr)

    print(f"[sentiment-article] scored {done}/{len(new_memberships)} new articles")
    return 0


# ---------------------------------------------------------------------------
# story-entities (one task, NER model, fresh pass over combined story text)
# ---------------------------------------------------------------------------

def story_entities(run_id: int) -> int:
    from nlp.ner import MODEL_NAME as NER_MODEL_NAME
    from nlp.ner import extract_entities

    touched = ops.get_touched_story_ids(run_id)
    done = 0
    for story_id in touched:
        try:
            articles = ops.get_story_articles(story_id)
            entities = extract_entities(_combined_story_text(articles))
            ops.delete_story_entities(story_id)
            ops.insert_story_entities(story_id, entities, NER_MODEL_NAME)
            done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[story-entities] failed for story {story_id}: {exc}", file=sys.stderr)

    print(f"[story-entities] processed {done}/{len(touched)} touched stories")
    return 0


# ---------------------------------------------------------------------------
# finalize (one task, no model, DB accounting)
# ---------------------------------------------------------------------------

def finalize(run_id: int) -> int:
    run = ops.get_scrape_run(run_id)
    touched = ops.get_touched_story_ids(run_id)

    articles_fetched = ops.count_fetched_for_run(run_id)
    articles_matched = ops.count_matched_articles(run_id)
    summaries_generated = ops.count_summaries_for_stories(touched)

    status = "partial" if run["front_pages_failed"] > 0 else "success"
    ops.finish_scrape_run(
        run_id, status,
        articles_fetched=articles_fetched,
        articles_matched=articles_matched,
        stories_touched=len(touched),
        summaries_generated=summaries_generated,
    )
    print(f"[finalize] run {run_id} finished: {status} fetched={articles_fetched} "
          f"matched={articles_matched} stories_touched={len(touched)} summaries={summaries_generated}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start-run")
    p_start.add_argument("--out-dir", type=Path, default=Path("/tmp/outputs"))

    p_discover = sub.add_parser("discover")
    p_discover.add_argument("--run-id", type=int, required=True)
    p_discover.add_argument("--source", type=str, required=True, help="source module name")

    for name in ["fetch-article-bodies", "embed-article"]:
        sub.add_parser(name)

    for name in ["filter-article", "cluster-stories", "summarize-story", "sentiment-article", "story-entities", "finalize"]:
        p = sub.add_parser(name)
        p.add_argument("--run-id", type=int, required=True)

    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = _parse_args()

    try:
        if args.command == "start-run":
            return start_run(args.out_dir)
        if args.command == "discover":
            return discover(args.run_id, args.source)
        if args.command == "fetch-article-bodies":
            return fetch_article_bodies()
        if args.command == "filter-article":
            return filter_article(args.run_id)
        if args.command == "embed-article":
            return embed_article()
        if args.command == "cluster-stories":
            return cluster_stories(args.run_id)
        if args.command == "summarize-story":
            return summarize_story(args.run_id)
        if args.command == "sentiment-article":
            return sentiment_article(args.run_id)
        if args.command == "story-entities":
            return story_entities(args.run_id)
        if args.command == "finalize":
            return finalize(args.run_id)
        raise AssertionError(f"unhandled command: {args.command}")
    except Exception as exc:  # noqa: BLE001 - total stage failure
        traceback.print_exc()
        run_id = getattr(args, "run_id", None)
        if run_id is not None:
            ops.finish_scrape_run(run_id, "failed", error_message=f"{args.command}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
