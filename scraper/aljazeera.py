"""aljazeera.com scraper. Front page and article content both live in
window.__APOLLO_STATE__="<base64 JSON>" - base64-encoded, unlike
nine.com.au's and ABC's unencoded state. Once decoded, it's Apollo entities
keyed like "Post:<id>"; each Post already holds its own full article body as
one HTML string in `content` (no separate per-paragraph block list to walk,
unlike nine.com.au's MARKUP blocks).

Only Posts with type == "post" are treated as standalone articles - the same
state also holds "episode" (video) and "liveblog" entries, and some Post
entries that are just liveblog updates with no `link` of their own."""
import base64
import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

_APOLLO_STATE_RE = re.compile(r'window\.__APOLLO_STATE__="([^"]+)"')


def _extract_apollo_state(html: str) -> dict:
    match = _APOLLO_STATE_RE.search(html)
    if not match:
        return {}
    try:
        decoded = base64.b64decode(match.group(1)).decode("utf-8")
        return json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return {}


def extract_headlines(html: str, base_url: str, section_slug: str) -> list[dict]:
    state = _extract_apollo_state(html)
    results = []
    seen_urls = set()
    for value in state.values():
        if not isinstance(value, dict) or value.get("__typename") != "Post":
            continue
        if value.get("type") != "post":
            continue
        title = value.get("title")
        link = value.get("link")
        if not title or not link:
            continue
        url = urljoin(base_url, link)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append({"headline": title, "url": url, "section_slug": section_slug})
    return results


def extract_body(html: str, url: str) -> str:
    state = _extract_apollo_state(html)
    for value in state.values():
        if not isinstance(value, dict) or value.get("__typename") != "Post":
            continue
        link = value.get("link") or ""
        if link and link in url:
            body = _render_content(value.get("content", ""))
            if body:
                return body
            break
    if trafilatura is not None:
        extracted = trafilatura.extract(html, url=url)
        if extracted:
            return extracted
    return ""


def _render_content(content_html: str) -> str:
    soup = BeautifulSoup(content_html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    paragraphs = []
    for el in soup.find_all(["p", "h2", "h3"]):
        text = el.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)
