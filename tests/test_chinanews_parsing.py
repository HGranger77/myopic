from pathlib import Path

from scraper.chinanews import extract_body, extract_headlines

FIXTURES = Path(__file__).parent / "fixtures"
FRONT_PAGE = (FIXTURES / "chinanews_front_page_sample.html").read_text(encoding="utf-8")
ARTICLE = (FIXTURES / "chinanews_article_sample.html").read_text(encoding="utf-8")


def test_extracts_headline_and_resolves_protocol_relative_and_absolute_urls():
    results = extract_headlines(FRONT_PAGE, "https://www.chinanews.com.cn", "home")
    assert {
        "headline": "中国航天事业创建70周年",
        "url": "https://www.chinanews.com.cn/sp/2026/10-09/10709999.shtml",
        "section_slug": "home",
    } in results


def test_skips_links_without_text_and_non_article_paths():
    results = extract_headlines(FRONT_PAGE, "https://www.chinanews.com.cn", "home")
    urls = [r["url"] for r in results]
    assert not any("1207309" in u for u in urls)
    assert not any("not-an-article-link" in u for u in urls)


def test_dedupes_duplicate_entries():
    results = extract_headlines(FRONT_PAGE, "https://www.chinanews.com.cn", "home")
    urls = [r["url"] for r in results]
    assert len(urls) == len(set(urls)) == 1


def test_extract_body_only_direct_child_paragraphs_of_left_zw():
    body = extract_body(ARTICLE, "https://www.chinanews.com.cn/sp/2026/10-09/10709999.shtml")
    assert body == "中国航天事业创建70周年。\n\n从无到有，从弱到强。"
    assert "图片说明" not in body
    assert "不相关" not in body
