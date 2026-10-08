"""Per-source scraper registry.

Each entry maps a source's `module` name (from config/sources.yaml) to the
pair of functions that know how to parse *that* source's markup. Adding a
second source means writing a new extract_headlines/extract_body pair and
registering it here - nothing upstream (discover's fan-out, the DB schema,
every later DAG stage) needs to change, since they all key off the source
name, not its implementation.
"""
from scraper import abc_news
from scraper import article as nine_com_au_article
from scraper import front_page as nine_com_au_front_page

SOURCES = {
    "nine_com_au": {
        "extract_headlines": nine_com_au_front_page.extract_headlines,
        "extract_body": nine_com_au_article.extract_body,
    },
    "abc_net_au": {
        "extract_headlines": abc_news.extract_headlines,
        "extract_body": abc_news.extract_body,
    },
}


def get_source(module: str) -> dict:
    if module not in SOURCES:
        raise ValueError(f"unknown source module: {module!r} (known: {sorted(SOURCES)})")
    return SOURCES[module]
