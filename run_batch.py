"""Batch pipeline, one Argo task per DAG stage - each loads its model (if any)
once and loops over everything that stage owns, rather than fanning out one
pod per article. The only fan-out left is `discover`, one task per source.

Every stage after `start-run` takes only --run-id; nothing else threads
through Argo parameters. Each stage finds its own worklist by querying
Postgres directly (e.g. "embedded articles not yet clustered"), so there's
no explicit article/story-id list to pass between tasks.

Relevance filtering runs AFTER clustering/summarizing, not before: every
fetched article gets embedded and clustered into a story regardless of
watchlist relevance, every touched story gets a summary, and only THEN does
tag-relevance classify that short summary against the watchlist and stamp
the verdict onto the story's member articles. This moved deliberately - see
tag_relevance()'s docstring below for why checking the raw article text
directly was both slower and less accurate than checking the summary.
sentiment-article and article-entities, the two stages after tag-relevance,
are scoped back down to the relevant subset, since there's no point paying
for either on stories nobody's watching for.
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
from scraper import fetch
from scraper.registry import get_source

# nlp.relevance/ner/embedding/summarize/sentiment are NOT imported here - each
# pulls in a heavy library (transformers/torch, spacy, sentence_transformers)
# that a stage without a model (start-run, discover, fetch-article-bodies,
# cluster-stories, finalize) would otherwise pay for at import time
# regardless of needing it. Each model-bearing function below imports its
# own module locally instead, right where it's used.

CONFIG_PATH = Path(__file__).parent / "config" / "sources.yaml"

STORY_WINDOW_DAYS = int(os.environ.get("MYOPIC_STORY_WINDOW_DAYS", "14"))
STORY_MAX_DISTANCE = float(os.environ.get("MYOPIC_STORY_MAX_DISTANCE", "0.3"))
FETCH_CONCURRENCY = int(os.environ.get("MYOPIC_FETCH_CONCURRENCY", "8"))
MAX_STORY_TEXT_CHARS = 8000  # keeps a many-source story's combined text bounded
# translate-article is autoregressive (see nlp/translate.py) - at ~60s/article
# this caps a single run's cost rather than translating a whole backlog (e.g.
# a source's full homepage on first discovery) in one go. Untranslated
# articles just wait for a later run - set low (5) for initial validation of
# the stage itself, not tuned for throughput yet.
TRANSLATE_MAX_ARTICLES = int(os.environ.get("MYOPIC_TRANSLATE_MAX_ARTICLES", "5"))


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
    for i, (url, article) in enumerate(by_url.items(), start=1):
        html, error = results[url]
        if html is None:
            ops.mark_article_failed(article["id"], error or "fetch failed")
            failed += 1
            print(f"[fetch-article-bodies] {i}/{len(by_url)} failed: {url}")
            continue
        try:
            extract_body = get_source(article["source"])["extract_body"]
            body_text = extract_body(html, url)
            if not body_text:
                raise ValueError("no body text extracted")
            ops.mark_article_fetched(article["id"], body_text)
            fetched += 1
            print(f"[fetch-article-bodies] {i}/{len(by_url)} fetched: {url}")
        except Exception as exc:  # noqa: BLE001 - one bad article shouldn't abort the batch
            ops.mark_article_failed(article["id"], str(exc))
            failed += 1
            print(f"[fetch-article-bodies] {i}/{len(by_url)} failed: {url} ({exc})")

    print(f"[fetch-article-bodies] fetched {fetched}, failed {failed} (of {len(pending)} pending)")
    return 0


# ---------------------------------------------------------------------------
# translate-article (one task, translation model, non-English sources only)
#
# Every stage after this one is English-only (embedding, summarization,
# sentiment, NER, zero-shot tagging) - chinanews.com.cn (see
# config/sources.yaml's per-source `language`) is this project's first
# non-English source, so this is where its text becomes usable by the rest
# of the pipeline. Skips the model import entirely (not just the model
# load) when every configured source is English - the common case for
# anyone who hasn't added a non-English source, and the same "stages
# without a model pay nothing" discipline as every other stage.
# ---------------------------------------------------------------------------

def translate_article() -> int:
    sources = load_sources()
    non_english_modules = [
        s["module"] for s in sources["sources"] if s.get("language", "en") != "en"
    ]
    if not non_english_modules:
        print("[translate-article] no non-English sources configured, nothing to do")
        return 0

    from nlp.translate import translate_to_english

    to_translate = ops.get_articles_needing_translation(non_english_modules, TRANSLATE_MAX_ARTICLES)
    for i, article in enumerate(to_translate, start=1):
        translated_headline = translate_to_english(article["headline"])
        translated_body = translate_to_english(article["body_text"])
        ops.set_article_translation(article["id"], translated_headline, translated_body)
        print(f"[translate-article] {i}/{len(to_translate)} article {article['id']} translated")

    print(f"[translate-article] translated {len(to_translate)} articles "
          f"(capped at {TRANSLATE_MAX_ARTICLES}/run)")
    return 0


# ---------------------------------------------------------------------------
# embed-article (one task, embedding model, pure CPU loop)
# ---------------------------------------------------------------------------

def embed_article() -> int:
    from nlp.embedding import embed

    to_embed = ops.get_articles_needing_embedding()
    for i, article in enumerate(to_embed, start=1):
        vector = embed(article["body_text"])
        ops.set_article_embedding(article["id"], vector)
        print(f"[embed-article] {i}/{len(to_embed)} article {article['id']} embedded")
    print(f"[embed-article] embedded {len(to_embed)} articles")
    return 0


# ---------------------------------------------------------------------------
# cluster-stories (one task, pgvector similarity, no model)
# ---------------------------------------------------------------------------

def cluster_stories(run_id: int) -> int:
    to_cluster = ops.get_articles_needing_clustering()
    new_stories = 0
    joined_stories = 0
    for i, article in enumerate(to_cluster, start=1):
        story_id = ops.find_similar_story(article["embedding"], STORY_WINDOW_DAYS, STORY_MAX_DISTANCE)
        if story_id is None:
            story_id = ops.create_story()
            new_stories += 1
            print(f"[cluster-stories] {i}/{len(to_cluster)} article {article['id']}: new story {story_id}")
        else:
            joined_stories += 1
            print(f"[cluster-stories] {i}/{len(to_cluster)} article {article['id']}: joined story {story_id}")
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
    for i, story_id in enumerate(touched, start=1):
        try:
            articles = ops.get_story_articles(story_id)
            summary_text = summarize(_combined_story_text(articles))
            ops.upsert_story_summary(story_id, summary_text, SUMMARIZER_MODEL_NAME)
            done += 1
            print(f"[summarize-story] {i}/{len(touched)} story {story_id} summarized")
        except Exception as exc:  # noqa: BLE001 - one bad story shouldn't abort the rest
            print(f"[summarize-story] {i}/{len(touched)} failed for story {story_id}: {exc}", file=sys.stderr)

    print(f"[summarize-story] summarized {done}/{len(touched)} touched stories")
    return 0


# ---------------------------------------------------------------------------
# tag-relevance (one task, zero-shot classifier, loop over touched stories)
#
# Runs downstream of summarize-story and classifies each touched story's
# SHORT SUMMARY against the watchlist, not the raw combined article text.
# Tried on raw article bodies first: both slower (encoder cost scales with
# input length - ~52s for one long real article vs ~5s for its summary) and
# less accurate (a long article's scores compressed into an uninformative
# 0.56-0.84 band across every topic; the summary gave a clean, well-
# separated result). See nlp/tagging.py for the full writeup.
#
# The verdict is computed ONCE per story but stamped onto every current
# member article (old and new) - consistent with summarize-story, which
# likewise always reflects the story's complete current membership, not
# just what's new this run. nlp.tagging (zero-shot classification) is wired
# in; nlp.relevance (the LLM, one call per topic) is kept intact but
# disconnected - swap the import below to switch back.
# ---------------------------------------------------------------------------

def tag_relevance(run_id: int) -> int:
    from nlp.tagging import check_relevance

    watchlist = ops.get_active_watchlist_with_aliases()
    touched = ops.get_touched_story_ids(run_id)
    relevant = 0
    for i, story_id in enumerate(touched, start=1):
        summary_text = ops.get_story_summary_text(story_id)
        member_articles = ops.get_story_articles(story_id)

        matches = check_relevance(summary_text, watchlist) if watchlist and summary_text else []
        is_relevant = len(matches) > 0

        for article in member_articles:
            ops.delete_watchlist_matches(article["id"])
            for match in matches:
                ops.insert_watchlist_match(
                    article["id"], match["watchlist_entity_id"],
                    None, match["canonical_name"], run_id,
                )
            ops.set_article_relevance(article["id"], is_relevant)

        relevant += int(is_relevant)
        verdict = "RELEVANT (" + ", ".join(m["canonical_name"] for m in matches) + ")" if matches else "not relevant"
        print(f"[tag-relevance] {i}/{len(touched)} story {story_id} "
              f"({len(member_articles)} articles): {verdict}")

    print(f"[tag-relevance] tagged {len(touched)} stories, {relevant} relevant")
    return 0


# ---------------------------------------------------------------------------
# sentiment-article (one task, sentiment model, loop over new memberships)
# ---------------------------------------------------------------------------

def sentiment_article(run_id: int) -> int:
    from nlp.sentiment import MODEL_NAME as SENTIMENT_MODEL_NAME
    from nlp.sentiment import analyze as analyze_sentiment

    new_memberships = ops.get_new_story_memberships_for_run(run_id)
    done = 0
    for i, membership in enumerate(new_memberships, start=1):
        try:
            article = ops.get_article(membership["article_id"])
            label, score = analyze_sentiment(article["body_text"])
            ops.upsert_article_sentiment(
                article["id"], membership["story_id"], label, score, SENTIMENT_MODEL_NAME
            )
            done += 1
            print(f"[sentiment-article] {i}/{len(new_memberships)} article {article['id']}: {label} ({score:.3f})")
        except Exception as exc:  # noqa: BLE001
            print(f"[sentiment-article] {i}/{len(new_memberships)} failed for article "
                  f"{membership['article_id']}: {exc}", file=sys.stderr)

    print(f"[sentiment-article] scored {done}/{len(new_memberships)} new articles")
    return 0


# ---------------------------------------------------------------------------
# article-entities (one task, NER model, loop over new memberships)
#
# Same worklist as sentiment-article - one-shot per article, not redone
# every run. Powers the per-article/per-source entity breakdown on the
# /stories page.
# ---------------------------------------------------------------------------

def article_entities(run_id: int) -> int:
    from nlp.ner import extract_entities

    new_memberships = ops.get_new_story_memberships_for_run(run_id)
    done = 0
    for i, membership in enumerate(new_memberships, start=1):
        try:
            article = ops.get_article(membership["article_id"])
            entities = extract_entities(article["body_text"])
            ops.delete_article_entities(article["id"])
            ops.insert_article_entities(article["id"], entities)
            done += 1
            print(f"[article-entities] {i}/{len(new_memberships)} article {article['id']}: {len(entities)} entities")
        except Exception as exc:  # noqa: BLE001
            print(f"[article-entities] {i}/{len(new_memberships)} failed for article "
                  f"{membership['article_id']}: {exc}", file=sys.stderr)

    print(f"[article-entities] processed {done}/{len(new_memberships)} new articles")
    return 0


# ---------------------------------------------------------------------------
# finalize (one task, no model, DB accounting)
# ---------------------------------------------------------------------------

def finalize(run_id: int) -> int:
    # stories_touched/summaries_generated report the watchlist-relevant
    # subset, not every story from every source - summarize-story now runs
    # on all daily news unconditionally (see tag-relevance's docstring), so
    # the all-touched-stories count would just be total news volume, not a
    # useful "how much of this run mattered" metric.
    run = ops.get_scrape_run(run_id)
    touched = ops.get_touched_relevant_story_ids(run_id)

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

    for name in ["fetch-article-bodies", "translate-article", "embed-article"]:
        sub.add_parser(name)

    for name in ["cluster-stories", "summarize-story", "tag-relevance", "sentiment-article", "article-entities", "finalize"]:
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
        if args.command == "translate-article":
            return translate_article()
        if args.command == "embed-article":
            return embed_article()
        if args.command == "cluster-stories":
            return cluster_stories(args.run_id)
        if args.command == "summarize-story":
            return summarize_story(args.run_id)
        if args.command == "tag-relevance":
            return tag_relevance(args.run_id)
        if args.command == "sentiment-article":
            return sentiment_article(args.run_id)
        if args.command == "article-entities":
            return article_entities(args.run_id)
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
