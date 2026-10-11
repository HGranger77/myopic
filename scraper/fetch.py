"""Shared HTTP session for all scraping."""
import asyncio

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_session = None


def get_session() -> requests.Session:
    global _session
    if _session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": _USER_AGENT})
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        session.mount("http://", HTTPAdapter(max_retries=retries))
        _session = session
    return _session


def get(url: str, timeout: int = 15) -> str:
    response = get_session().get(url, timeout=timeout)
    response.raise_for_status()
    # requests falls back to ISO-8859-1 (the old RFC default) whenever a
    # server's Content-Type header omits a charset - chinanews.com.cn does
    # exactly that, and requests ignores the page's own <meta charset=
    # "UTF-8"> entirely, silently mangling every non-ASCII character into
    # double-encoded garbage. Invisible on every English/ASCII source (ASCII
    # round-trips identically through either encoding), which is why this
    # went unnoticed until the first non-English source. apparent_encoding
    # sniffs the actual bytes instead of trusting the header.
    response.encoding = response.apparent_encoding
    return response.text


async def get_many(urls: list[str], concurrency: int = 8, timeout: int = 15) -> dict[str, tuple[str | None, str | None]]:
    """Fetch many URLs concurrently, bounded by `concurrency` in-flight requests.

    Returns {url: (html_or_None, error_or_None)} - one entry per input URL,
    never raises for an individual failure so one bad URL doesn't sink the batch.
    """
    results: dict[str, tuple[str | None, str | None]] = {}
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, timeout=timeout, follow_redirects=True) as client:
        async def fetch_one(url: str) -> None:
            async with semaphore:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    results[url] = (response.text, None)
                except Exception as exc:  # noqa: BLE001 - recorded per-url, not raised
                    results[url] = (None, str(exc))

        await asyncio.gather(*(fetch_one(url) for url in urls))
    return results
