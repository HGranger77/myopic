"""nine.com.au server-renders an Apollo GraphQL cache into every page as
`window.__APOLLO_STATE__ = {...};`. That JSON blob is a far more reliable
source of headlines/URLs/body text than scraping rendered markup, since it
isn't entangled with unrelated widgets (related-story tiles, ads, nav)."""
import json
import re

_APOLLO_RE = re.compile(r"window\.__APOLLO_STATE__\s*=\s*(\{.*?\});</script>", re.DOTALL)


def extract_apollo_state(html: str) -> dict:
    match = _APOLLO_RE.search(html)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def article_assets(state: dict) -> list[dict]:
    return [
        v for v in state.values()
        if isinstance(v, dict) and v.get("__typename") == "ArticleAsset"
    ]
