from pathlib import Path

from scraper.aljazeera import extract_body, extract_headlines

FIXTURES = Path(__file__).parent / "fixtures"
FRONT_PAGE = (FIXTURES / "aljazeera_front_page_sample.html").read_text()
ARTICLE = (FIXTURES / "aljazeera_article_sample.html").read_text()


def test_extracts_headline_and_resolved_url():
    results = extract_headlines(FRONT_PAGE, "https://www.aljazeera.com", "home")
    assert {
        "headline": "Parramatta Eels sign new prop for 2027 season",
        "url": "https://www.aljazeera.com/sport/2026/10/9/parramatta-eels-new-prop-signing",
        "section_slug": "home",
    } in results


def test_skips_non_post_types_and_missing_titles():
    results = extract_headlines(FRONT_PAGE, "https://www.aljazeera.com", "home")
    urls = [r["url"] for r in results]
    assert not any("eels-training" in u for u in urls)
    assert not any("no-headline" in u for u in urls)


def test_dedupes_duplicate_entries():
    results = extract_headlines(FRONT_PAGE, "https://www.aljazeera.com", "home")
    urls = [r["url"] for r in results]
    assert len(urls) == len(set(urls)) == 1


def test_extract_body_strips_scripts_and_picks_matching_post():
    body = extract_body(ARTICLE, "https://www.aljazeera.com/sport/2026/10/9/parramatta-eels-new-prop-signing")
    assert body == (
        "The Eels have signed a new front-rower ahead of the 2027 season.\n\n"
        "The club confirmed the signing on Thursday."
    )
    assert "tracking pixel" not in body
    assert "unrelated" not in body.lower()
