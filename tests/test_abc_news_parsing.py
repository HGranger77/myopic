from pathlib import Path

from scraper.abc_news import extract_body, extract_headlines

FIXTURES = Path(__file__).parent / "fixtures"
FRONT_PAGE = (FIXTURES / "abc_front_page_sample.html").read_text()
ARTICLE = (FIXTURES / "abc_article_sample.html").read_text()


def test_extracts_headline_and_resolved_url():
    results = extract_headlines(FRONT_PAGE, "https://www.abc.net.au", "home")
    assert {
        "headline": "Parramatta Eels sign new prop for 2027 season",
        "url": "https://www.abc.net.au/news/2026-10-09/parramatta-eels-new-prop-signing/107200001",
        "section_slug": "home",
    } in results


def test_skips_items_without_title():
    results = extract_headlines(FRONT_PAGE, "https://www.abc.net.au", "home")
    assert not any(r["url"].endswith("/no-headline/107200002") for r in results)


def test_dedupes_across_topstories_and_morenews():
    results = extract_headlines(FRONT_PAGE, "https://www.abc.net.au", "home")
    urls = [r["url"] for r in results]
    assert len(urls) == len(set(urls)) == 2


def test_extract_body_joins_paragraphs_and_skips_embeds():
    body = extract_body(ARTICLE, "https://www.abc.net.au/news/x")
    assert body == (
        "The Eels have signed a new front-rower ahead of the 2027 season.\n\n"
        "The club confirmed the signing on Thursday."
    )
