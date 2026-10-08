from pathlib import Path

from scraper.article import extract_body

FIXTURE = (Path(__file__).parent / "fixtures" / "article_sample.html").read_text()
URL = "https://www.nine.com.au/australia-news/solar-farm-fight-20261006-p613cq.html"


def test_extracts_body_in_block_order():
    body = extract_body(FIXTURE, URL)
    assert body == (
        "Michelle was looking forward to a quiet retirement.\n\n"
        "Then the solar farm proposal arrived in the mail."
    )


def test_does_not_pick_up_related_story_asset():
    body = extract_body(FIXTURE, URL)
    assert "unrelated related-story tile" not in body
