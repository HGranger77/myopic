"""chinanews.com.cn scraper (Chinese-language, simplified). Plain
server-rendered HTML, no embedded JSON state - headlines are plain <a> tags
whose href matches the article URL pattern (/<section>/<yyyy>/<mm-dd>/<id>.shtml),
and the body lives in div.left_zw inside div.content_maincontent_content.
Body text comes back in Chinese - see nlp/translate.py, which run_batch.py
runs on every article before anything English-only (embedding, summarize,
sentiment, NER, tagging) touches it."""
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

_ARTICLE_PATH_RE = re.compile(r"/\d{4}/\d{2}-\d{2}/\d+\.shtml$")


def extract_headlines(html: str, base_url: str, section_slug: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    results = []
    seen_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not _ARTICLE_PATH_RE.search(href):
            continue
        headline = a.get_text(strip=True)
        if not headline:
            continue
        url = urljoin(base_url, href)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append({"headline": headline, "url": url, "section_slug": section_slug})
    return results


def extract_body(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one("div.left_zw")
    if container is not None:
        for tag in container.find_all(["script", "style"]):
            tag.decompose()
        paragraphs = [p.get_text(strip=True) for p in container.find_all("p", recursive=False)]
        paragraphs = [p for p in paragraphs if p]
        body = "\n\n".join(paragraphs)
        if body:
            return body

    if trafilatura is not None:
        extracted = trafilatura.extract(html, url=url)
        if extracted:
            return extracted
    return ""
