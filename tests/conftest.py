import pytest
from dotenv import load_dotenv

load_dotenv()

from database import get_connection, init_db  # noqa: E402

_TABLES = [
    "article_sentiment",
    "story_summaries",
    "story_articles",
    "stories",
    "watchlist_matches",
    "article_entities",
    "articles",
    "scrape_runs",
    "watchlist_aliases",
    "watchlist_entities",
]


@pytest.fixture(scope="session", autouse=True)
def _schema():
    init_db()


@pytest.fixture(autouse=True)
def _clean_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE")
    yield
