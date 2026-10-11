from pathlib import Path

from scraper.news_com_au import extract_body, extract_headlines

FIXTURES = Path(__file__).parent / "fixtures"
FRONT_PAGE = (FIXTURES / "newscomau_front_page_sample.html").read_text()
ARTICLE = (FIXTURES / "newscomau_article_sample.html").read_text()


def test_extracts_headline_and_resolved_url():
    results = extract_headlines(FRONT_PAGE, "https://www.news.com.au", "home")
    assert {
        "headline": "Parramatta Eels sign new prop for 2027 season",
        "url": "https://www.news.com.au/sport/nrl/parramatta-eels-new-prop-signing/news-story/aaa",
        "section_slug": "home",
    } in results


def test_skips_blocks_without_title_link():
    results = extract_headlines(FRONT_PAGE, "https://www.news.com.au", "home")
    assert not any(r["url"].endswith("/ccc") for r in results)


def test_dedupes_duplicate_entries():
    results = extract_headlines(FRONT_PAGE, "https://www.news.com.au", "home")
    urls = [r["url"] for r in results]
    assert len(urls) == len(set(urls)) == 1


def test_extract_body_strips_scripts_and_related_tiles():
    body = extract_body(ARTICLE, "https://www.news.com.au/sport/nrl/parramatta-eels-new-prop-signing/news-story/aaa")
    assert body == (
        "The Eels have signed a new front-rower ahead of the 2027 season.\n\n"
        "The club confirmed the signing on Thursday."
    )
    assert "tracking pixel" not in body
    assert "unrelated" not in body.lower()
