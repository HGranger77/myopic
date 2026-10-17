"""Chooses the one place a story happened in, from the places its articles mention.

Used by the geolocate-story stage. The stage's steps, and what does each one:

  1. Candidates     NER places (GPE/LOC/FAC) already extracted by article-entities.
                    No model runs here - a database read.            (ops.get_story_place_candidates)
  2. Name screen    Drops a candidate whose text is mostly used as a PERSON/ORG
                    name in the same story (the surname "Kabila", the cricketer
                    "Litchfield"). Plain counting.                   (screen_candidates)
  3. Geocode        Looks each place up on OpenStreetMap's Nominatim; drops
                    anything that isn't a real place. No ML - an HTTP
                    call, cached in Postgres.                        (nlp.geocode)
  4. Place scoring  THE ML STEP. facebook/bart-large-mnli, the same
                    zero-shot NLI model tag-relevance uses, is asked how well
                    "This event happened in <place>." follows from the story's
                    headline + summary, once per candidate.          (nli_scores)
  5. Choice         Picks the best-scoring place, steps down to a more specific
                    place inside it if one is nearly as good, and returns nothing
                    if the best place is too weakly supported.      (choose_place)

Why NLI: ranking by how often a place is mentioned, how early, or whether it's in
the headline all plateaued at the same accuracy, because the places a story
*mentions* most are often not where it *happened* (Yemen, repeated, vs. the
Riyadh a missile was aimed at). Entailment reads the sentence rather than
counting words. The constants below were tuned on 261 hand-audited stories
(see docs/geolocation-audit.md); on held-out stories this cut wrong pins from
16 to 12 of 130 versus the best non-ML ranking.
"""
import re

MODEL_NAME = "facebook/bart-large-mnli"  # shared with nlp/tagging.py
HYPOTHESIS = "This event happened in {}."

MAX_CANDIDATES = 8        # geocoded places sent to the model, per story
MIN_NLI = 0.4             # best place's entailment must reach this, else no pin
SIGNIFICANT_WEIGHT = 0.1  # small bonus per article that finds the place significant (max 3)
CHILD_MARGIN = 0.2        # a place inside the winner replaces it if within this score
PRECISE_MIN_RANK = 12     # Nominatim place_rank at/above which a pin is "precise"
_MAX_PREMISE_CHARS = 1500


def place_key(text: str) -> str:
    return re.sub(r"^the ", "", text.strip().lower())


def screen_candidates(candidates: list[dict]) -> list[dict]:
    """Step 2. Each candidate: {key, mentions, person_mentions, ...}. Drops a
    candidate when words of that exact text appear in PERSON/ORG entities at
    least as often as the text appears as a place."""
    return [c for c in candidates if c["person_mentions"] < max(c["mentions"], 1)]


def is_inside(child: dict, parent: dict) -> bool:
    """True if the child's geocoded address lists the parent as a component
    ("Sydney, New South Wales, Australia" is inside Australia)."""
    if child["key"] == parent["key"]:
        return False
    parts = [p.strip().lower() for p in (child["geo"]["display_name"] or "").split(",")]
    parent_name = (parent["geo"]["display_name"] or "").split(",")[0].strip().lower()
    return parent_name in parts[1:]


def nli_scores(premise: str, places: list[str]) -> dict[str, float]:
    """Step 4, the only model call. Returns {place: P(story happened in place)}."""
    from nlp.tagging import _get_classifier  # one shared bart-large-mnli per process

    result = _get_classifier()(
        premise[:_MAX_PREMISE_CHARS], candidate_labels=places,
        hypothesis_template=HYPOTHESIS, multi_label=True,
    )
    return dict(zip(result["labels"], result["scores"]))


def choose_place(candidates: list[dict], scores: dict[str, float]):
    """Step 5. candidates carry {text, key, significant_articles, geo}; scores is
    nli_scores() output keyed by candidate text. Returns
    (candidate, nli_score) or None."""
    if not candidates:
        return None
    combined = {
        c["key"]: scores[c["text"]] + SIGNIFICANT_WEIGHT * min(c["significant_articles"], 3)
        for c in candidates
    }
    best = max(candidates, key=lambda c: combined[c["key"]])
    children = [c for c in candidates
                if is_inside(c, best) and combined[c["key"]] >= combined[best["key"]] - CHILD_MARGIN]
    if children:
        best = max(children, key=lambda c: combined[c["key"]])
    if scores[best["text"]] < MIN_NLI:
        return None
    return best, scores[best["text"]]
