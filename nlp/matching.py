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
            if contains_whole_word(entity_text, alias) or contains_whole_word(alias, entity_text):
                matches.append((entity_text, watchlist_entity_id, alias))
    return matches


def contains_whole_word(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return bool(re.search(rf"\b{re.escape(needle)}\b", haystack))


# A story's real subject and location almost always get introduced in the
# lede paragraph even on a single mention, while an author byline or an
# incidental aside is both late in the body AND never repeated - so "early
# OR repeated" is a better significance signal than either alone.
_EARLY_POSITION_CHARS = 200
_MIN_MENTIONS = 2


def filter_significant_entities(
    entities: list[dict],
    min_mentions: int = _MIN_MENTIONS,
    early_position_chars: int = _EARLY_POSITION_CHARS,
) -> list[str]:
    """Takes every occurrence of ONE entity category (e.g. all PERSON rows, or
    all GPE+LOC rows pooled together - call once per UI bucket) from
    nlp.ner.extract_entities (or DB rows re-hydrated into the same shape) and
    returns just the ones worth showing as significant to the article, as
    canonical display text ordered by first mention.

    Lives here rather than in nlp.ner because it's pure text/position logic
    with no model dependency - it's called from database/operations.py
    (get_article_entities), which runs in the lightweight API image that
    doesn't have spaCy installed, so it must not import anything that pulls
    spaCy in.

    Two things happen:

    1. Variant merging - occurrences whose normalized text is a whole-word
       substring of each other ("Biden" / "Joe Biden") are treated as the same
       entity, so a name isn't undercounted just because the article
       shortens it on later mentions. The longest variant seen is used as the
       canonical display text.
    2. Significance filtering - a merged entity is kept only if it's
       mentioned `min_mentions`+ times, or its first mention falls within
       `early_position_chars` of the start of the text. This is what drops a
       one-off byline name or incidental aside (late, single mention) while
       still keeping a short brief's actual who/where even if it's mentioned
       only once (early mention).

    This is deliberately simple substring/position heuristics, not real
    coreference resolution - "Biden" and "the President" won't merge, and a
    repeated-but-irrelevant aside mentioned twice in the lede would still
    pass. Good enough for a visual "what's this story about" signal without
    adding another model to the pipeline.
    """
    groups: list[dict] = []
    for ent in sorted(entities, key=lambda e: e.get("start_char") or 0):
        text = ent["text"]
        norm = ent.get("normalized_text") or normalize(text)
        start = ent.get("start_char") or 0
        group = next(
            (
                g
                for g in groups
                if any(contains_whole_word(norm, v[1]) or contains_whole_word(v[1], norm) for v in g["variants"])
            ),
            None,
        )
        if group is None:
            group = {"variants": []}
            groups.append(group)
        group["variants"].append((text, norm, start))

    significant = []
    for group in groups:
        variants = group["variants"]
        earliest = min(start for _, _, start in variants)
        if len(variants) < min_mentions and earliest >= early_position_chars:
            continue
        canonical = max(variants, key=lambda v: len(v[0]))[0]
        significant.append((earliest, canonical))

    significant.sort(key=lambda pair: pair[0])
    return [text for _, text in significant]
