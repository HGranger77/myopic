"""abc.net.au is a Next.js site - every page embeds its server-rendered props
as plain JSON in <script id="__NEXT_DATA__" type="application/json">, no JS
assignment wrapper (unlike nine.com.au's __APOLLO_STATE__), so it's just
json.loads-able directly."""
import json
import re

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL
)


def extract_next_data(html: str) -> dict:
    match = _NEXT_DATA_RE.search(html)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def collect_text(node) -> str:
    """Recursively joins every {"type": "text"} node's content under `node` -
    walks straight past "embed" (image/video) nodes, which carry no text."""
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("content", "")
    return "".join(collect_text(child) for child in node.get("children", []))
