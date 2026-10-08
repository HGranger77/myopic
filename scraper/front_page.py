"""Extract (headline, url) pairs from a front page's embedded Apollo state."""
from urllib.parse import urljoin

from scraper.apollo import article_assets, extract_apollo_state


def extract_headlines(html: str, base_url: str, section_slug: str) -> list[dict]:
    """Returns [{headline, url, section_slug}, ...], deduped by resolved url."""
    state = extract_apollo_state(html)
    results = []
    seen_urls = set()
    for asset in article_assets(state):
        path = asset.get("urls", {}).get("canonical", {}).get("path")
        headline = asset.get("headlines", {}).get("headline")
        if not path or not headline:
            continue
        url = urljoin(base_url, path)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append({"headline": headline, "url": url, "section_slug": section_slug})
    return results
