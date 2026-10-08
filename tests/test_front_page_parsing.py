from pathlib import Path

from scraper.front_page import extract_headlines

FIXTURE = (Path(__file__).parent / "fixtures" / "front_page_sample.html").read_text()


def test_extracts_headline_and_resolved_url():
    results = extract_headlines(FIXTURE, "https://www.nine.com.au", "australia-news")
    assert {
        "headline": "Solar farm fight divides small town",
        "url": "https://www.nine.com.au/australia-news/solar-farm-fight-20261006-p613cq.html",
        "section_slug": "australia-news",
    } in results


def test_skips_assets_without_headline():
    results = extract_headlines(FIXTURE, "https://www.nine.com.au", "australia-news")
    assert not any(r["url"].endswith("/no-headline.html") for r in results)


def test_dedupes_and_preserves_count():
    results = extract_headlines(FIXTURE, "https://www.nine.com.au", "australia-news")
    urls = [r["url"] for r in results]
    assert len(urls) == len(set(urls)) == 2
