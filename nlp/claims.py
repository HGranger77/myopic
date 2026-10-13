"""Claim-candidate extraction and cross-source claim grouping.

No model of its own: sentences come from the spaCy pipeline nlp.ner already
loads, and grouping reuses the MiniLM sentence embeddings from nlp.embedding.

A "claim candidate" is a sentence a second source could plausibly confirm or
contradict - one carrying a figure (MONEY/PERCENT/QUANTITY/CARDINAL) or a
direct quotation. DATE is deliberately NOT a trigger: nearly every news
sentence says "on Tuesday", so it would flag everything. That is a heuristic,
not claim detection in any strict sense - it misses claims stated without
numbers or quotes, and some candidates ("two men were arrested") are barely
claims at all.

Grouping measures agreement between outlets, NOT truth: two outlets
repeating the same wire copy count as "corroborated".
"""
from collections import defaultdict
from typing import Optional

_FIGURE_LABELS = {"MONEY", "PERCENT", "QUANTITY", "CARDINAL"}
_QUOTE_CHARS = ('"', "“", "”")
MIN_CHARS = 30
MAX_CHARS = 400
MAX_PER_ARTICLE = 15


def is_claim_candidate(sentence: str, entity_labels: set[str]) -> bool:
    text = sentence.strip()
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    return bool(entity_labels & _FIGURE_LABELS) or any(q in text for q in _QUOTE_CHARS)


def extract_claim_sentences(text: str, max_claims: int = MAX_PER_ARTICLE) -> list[str]:
    if not text:
        return []
    from nlp.ner import _get_model

    doc = _get_model()(text)
    claims: list[str] = []
    for sent in doc.sents:
        labels = {ent.label_ for ent in sent.ents}
        if is_claim_candidate(sent.text, labels):
            claims.append(sent.text.strip())
            if len(claims) >= max_claims:
                break
    return claims


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def group_claims(claims: list[dict], max_distance: float) -> list[dict]:
    """claims: [{id, article_id, source, embedding}, ...] for ONE story, with
    unit-length embeddings (embed() normalizes), so cosine distance is
    1 - dot. Two claims are linked when they come from DIFFERENT articles and
    are within max_distance; groups are connected components (union-find).

    Returns [{claim_ids, source_count, status}, ...], status 'corroborated'
    (claims from 2+ distinct sources) or 'single_source'. A claim linked to
    nothing is a one-member single_source group. Sentences within one article
    never link to each other - an article repeating itself isn't agreement."""
    parent = list(range(len(claims)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]
            if a["article_id"] == b["article_id"]:
                continue
            if 1.0 - _dot(a["embedding"], b["embedding"]) <= max_distance:
                parent[find(i)] = find(j)

    members: dict[int, list[dict]] = defaultdict(list)
    for i, claim in enumerate(claims):
        members[find(i)].append(claim)

    groups = []
    for group in members.values():
        source_count = len({c["source"] for c in group})
        groups.append({
            "claim_ids": [c["id"] for c in group],
            "source_count": source_count,
            "status": "corroborated" if source_count >= 2 else "single_source",
        })
    return groups
