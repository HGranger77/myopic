"""Extract full article body text from an article page.

Primary path: the page's own Apollo state contains an ArticleAsset for this
exact article (plus a handful of unrelated "related story" assets) with a
`body.blocks[]` list of typed blocks (MARKUP/IMAGE/VIDEOS/...) — we render
just the MARKUP blocks' HTML to plain text, in order. If that structure isn't
found (template change, non-article page), fall back to trafilatura's
general-purpose boilerplate removal.
"""
from bs4 import BeautifulSoup

from scraper.apollo import article_assets, extract_apollo_state


def extract_body(html: str, url: str) -> str:
    state = extract_apollo_state(html)
    assets = article_assets(state)
    target = _find_matching_asset(assets, url)
    if target is not None:
        body_text = _render_body(target.get("body", {}))
        if body_text:
            return body_text
    return _fallback_extract(html)


def _find_matching_asset(assets: list[dict], url: str) -> dict | None:
    for asset in assets:
        path = asset.get("urls", {}).get("canonical", {}).get("path", "")
        if path and path in url:
            return asset
    return None


def _render_body(body: dict) -> str:
    paragraphs = []
    for block in body.get("blocks", []):
        if block.get("type") == "MARKUP":
            text = BeautifulSoup(block.get("markup", ""), "lxml").get_text(" ", strip=True)
            if text:
                paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _fallback_extract(html: str) -> str:
    import trafilatura

    return trafilatura.extract(html) or ""
