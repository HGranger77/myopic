"""Fluff detection: flags stories that carry little real information
(celebrity gossip, listicles, lifestyle/horoscope/shopping content, puff
pieces). Same zero-shot BART-MNLI classifier as nlp/tagging.py, run on the
story SUMMARY for the same speed/accuracy reasons documented there.

Two label sets are scored in a single multi-label call; a story is fluff
when its best fluff label beats its best substantive label AND clears
_FLUFF_THRESHOLD. Requiring both avoids flagging borderline hard news that
merely scores weakly on everything.
"""
from nlp.tagging import _MAX_INPUT_CHARS, _get_classifier

FLUFF_LABELS = [
    "celebrity gossip or entertainment trivia",
    "lifestyle, horoscope, or shopping content",
    "light-hearted human interest or viral social media story",
    "listicle or product promotion",
]
SUBSTANTIVE_LABELS = [
    "politics, government, or policy",
    "business, economics, or finance",
    "crime, courts, or law enforcement",
    "science, technology, or health research",
    "war, international affairs, or disasters",
    "sports results or analysis",
]
_FLUFF_THRESHOLD = 0.6


def is_fluff_from_scores(scores: dict[str, float]) -> bool:
    fluff = max((scores.get(label, 0.0) for label in FLUFF_LABELS), default=0.0)
    substantive = max((scores.get(label, 0.0) for label in SUBSTANTIVE_LABELS), default=0.0)
    return fluff >= _FLUFF_THRESHOLD and fluff > substantive


def is_fluff(summary_text: str) -> bool:
    """summary_text: a story's short summary, not the raw article body."""
    if not summary_text or not summary_text.strip():
        return False
    result = _get_classifier()(
        summary_text[:_MAX_INPUT_CHARS],
        candidate_labels=FLUFF_LABELS + SUBSTANTIVE_LABELS,
        multi_label=True,
    )
    return is_fluff_from_scores(dict(zip(result["labels"], result["scores"])))
