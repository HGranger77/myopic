"""abc.net.au scraper. Scope deliberately limited to the homepage's
topStories + moreNews modules (no section pages) - this is a second source
added for testing the multi-source pipeline mechanism, not a fully general
ABC scraper."""
from urllib.parse import urljoin

from scraper.abc_state import collect_text, extract_next_data

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None


def extract_headlines(html: str, base_url: str, section_slug: str) -> list[dict]:
    data = extract_next_data(html)
    home = data.get("props", {}).get("pageProps", {}).get("home", {})

    results = []
    seen_urls = set()
    for module_key in ("topStories", "moreNews"):
        items = home.get(module_key, {}).get("items", [])
        for item in items:
            title = item.get("title")
            link = item.get("link")
            if not title or not link:
                continue
            url = urljoin(base_url, link)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            results.append({"headline": title, "url": url, "section_slug": section_slug})
    return results


def extract_body(html: str, url: str) -> str:
    data = extract_next_data(html)
    document = data.get("props", {}).get("pageProps", {}).get("document", {})
    text = document.get("loaders", {}).get("articledetail", {}).get("text", {})
    descriptor = text.get("descriptor", {})

    paragraphs = []
    for child in descriptor.get("children", []):
        if child.get("type") == "tagname":
            paragraph = collect_text(child).strip()
            if paragraph:
                paragraphs.append(paragraph)

    body = "\n\n".join(paragraphs)
    if body:
        return body

    if trafilatura is not None:
        extracted = trafilatura.extract(html, url=url)
        if extracted:
            return extracted
    return ""
