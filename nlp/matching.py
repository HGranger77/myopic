"""Normalization and alias-aware matching between extracted entities and the watchlist."""
import re

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def build_watchlist_index(watchlist: list[dict]) -> dict[str, int]:
    """watchlist: [{watchlist_entity_id, canonical_name, aliases: [normalized_alias, ...]}].

    Returns {normalized_alias: watchlist_entity_id}.
    """
    index: dict[str, int] = {}
    for entry in watchlist:
        for alias in entry["aliases"]:
            index[alias] = entry["watchlist_entity_id"]
    return index


def match_entities(normalized_entities: list[str], watchlist_index: dict[str, int]) -> list[tuple[str, int, str]]:
    """Returns [(normalized_entity_text, watchlist_entity_id, matched_alias), ...].

    Matches on exact normalized equality, or whole-word substring containment in
    either direction (so alias "biden" matches extracted "joe biden" and vice versa).
    """
    matches = []
    for entity_text in normalized_entities:
        for alias, watchlist_entity_id in watchlist_index.items():
            if _contains_whole(entity_text, alias) or _contains_whole(alias, entity_text):
                matches.append((entity_text, watchlist_entity_id, alias))
    return matches


def _contains_whole(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return bool(re.search(rf"\b{re.escape(needle)}\b", haystack))
