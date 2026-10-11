"""news.com.au scraper. Plain server-rendered HTML (no Apollo/Next.js state
to decode, unlike nine.com.au/ABC/Al Jazeera) - headlines come straight off
each front-page storyblock's own title link, and the article body lives in
`div.story-body-nodes`, which also embeds "Others you may like" promo tiles
as nested <article> elements - those get decomposed before reading <p> text,
the same "page has several article-shaped things, only one is real" problem
nine.com.au's scraper handles, just with the junk nested one level deeper."""
from urllib.parse import urljoin

from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None


def extract_headlines(html: str, base_url: str, section_slug: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    results = []
    seen_urls = set()
    for block in soup.find_all("article", class_="storyblock"):
        title_link = block.select_one("h3.storyblock_title a")
        if title_link is None:
            continue
        headline = title_link.get_text(strip=True)
        href = title_link.get("href")
        if not headline or not href:
            continue
        url = urljoin(base_url, href)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append({"headline": headline, "url": url, "section_slug": section_slug})
    return results


def extract_body(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one("div.story-body-nodes")
    if container is not None:
        for tag in container.find_all(["article", "script", "style"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        paragraphs = [p for p in paragraphs if p]
        body = "\n\n".join(paragraphs)
        if body:
            return body

    if trafilatura is not None:
        extracted = trafilatura.extract(html, url=url)
        if extracted:
            return extracted
    return ""
