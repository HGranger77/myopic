"""Watchlist relevance via zero-shot text classification - an alternative to
the one-LLM-call-per-topic approach in nlp/relevance.py (kept intact but not
wired into run_batch.py - see tag_relevance() there to switch back).

BART-MNLI is an *encoder* classifier: one batched forward pass scores every
candidate topic at once, rather than nlp/relevance.py's per-topic
autoregressive generation loop. That alone made it faster in a short-text
benchmark, but NOT on real article bodies: attention cost is quadratic in
input length, and measured directly against one real ~4000-char article, a
single call took ~52s with every one of 9 topics scored in an uninformative
0.56-0.84 band - no usable signal. Called on that same story's ~130-word
summary instead, the identical call took ~7s and separated cleanly (the 8
genuinely-irrelevant topics all scored under 0.05; only one bordered the
threshold). That's why tag_relevance() in run_batch.py calls this with a
story's SUMMARY, never the raw article text - this module doesn't enforce
that itself, so pass it the right thing.

It's also strictly better than embedding similarity on the one case that
experiment failed hard on: broad single-word topics like "politics" (BART-
MNLI scores a genuinely political summary 0.97 there; a raw sentence-
embedding comparison scored the same word only 0.08, barely above the noise
floor of a totally unrelated article).

Validated against the same true/false-positive pair used throughout this
project: the Parramatta Eels NRL team vs. an unrelated article about
Parramatta the *council*. True case scores 0.997 on "Parramatta Eels",
false case 0.448 - a wide enough margin for a single global threshold.

A bare acronym watchlist topic is a bad candidate label, though: zero-shot
NLI scores a label against an implicit "this example is about {label}"
hypothesis, and a 2-letter token like "AI" makes a near-meaningless one -
measured directly, an unrelated football-transfer summary scored 0.35-0.55
on the literal label "AI" (close enough to the 0.6 threshold to cross it on
some stories), and an unrelated ICE-raid story scored 0.55. Swapping in the
longest watchlist alias as the label text instead ("artificial intelligence"
if that's been added as an alias for "AI") fixed it cleanly: those same two
false positives dropped to 0.01 and 0.0002, while the genuine AI stories
stayed at 0.97-0.99. See _classification_label() below - this is exactly
what watchlist aliases are for, and nlp/relevance.py (the LLM path) already
used them; this module just hadn't been wired up to.

CASING MATTERS to this classifier - don't lowercase a label on its way in.
The first version of this fix pulled alias text from
get_active_watchlist_with_aliases(), which at the time returned
normalized_alias (always lowercase). That broke "AFL", which had no other
alias to fall back to, so its own canonical name got lowercased to "afl" -
measured directly, "AFL" scored 0.01-0.40 on two unrelated stories (a tiger
attack, a retail-earnings report) while "afl" scored 0.64-0.76 on the SAME
two stories, a false-positive regression worse than the "AI" bug this was
meant to fix. get_active_watchlist_with_aliases() now returns the
ORIGINAL-case alias text for exactly this reason.
"""
import torch
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"
_MAX_INPUT_CHARS = 4000
_RELEVANCE_THRESHOLD = 0.6

# PyTorch's own thread-count default is NOT cgroup-aware (same class of bug
# already hit twice with llama.cpp's n_threads and the GGML backend's
# OMP_NUM_THREADS - see nlp/relevance.py) - must match tag-relevance's
# cpu_request/cpu_limit in workflows/myopic_workflow.py, not be left to
# whatever this host happens to default to.
_N_THREADS = 4

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        torch.set_num_threads(_N_THREADS)
        _classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=-1)
    return _classifier


def _classification_label(entry: dict) -> str:
    """The longest alias is almost always the most descriptive form of a
    watchlist entity ("artificial intelligence" vs "AI", "Parramatta Eels"
    vs a shorter alias if one existed) - see this module's docstring for why
    that specifically matters to a zero-shot label. Falls back to
    canonical_name only if aliases is empty, which shouldn't normally happen
    since create_watchlist_entity always seeds canonical_name as an alias.

    Known limitation, not handled: if two different watchlist entities end
    up with the identical longest alias text, the classifier call still
    works, but only one of them will be resolvable from the returned scores
    dict in check_relevance() below - a non-issue for a small personal
    watchlist, but not something this guards against."""
    return max(entry["aliases"], key=len) if entry["aliases"] else entry["canonical_name"]


def check_relevance(text: str, watchlist: list[dict]) -> list[dict]:
    """text: a story's short summary, NOT the raw article body - see this
    module's docstring for why that distinction is load-bearing here.
    watchlist: [{watchlist_entity_id, canonical_name, aliases: [...]}, ...].

    Returns the subset judged relevant, each as
    {"watchlist_entity_id": ..., "canonical_name": ...}. Same return shape
    as nlp.relevance.check_relevance, so tag_relevance() can swap between
    the two by changing one import.
    """
    if not watchlist:
        return []

    classifier = _get_classifier()
    label_by_id = {entry["watchlist_entity_id"]: _classification_label(entry) for entry in watchlist}
    result = classifier(text[:_MAX_INPUT_CHARS], candidate_labels=list(label_by_id.values()), multi_label=True)
    scores = dict(zip(result["labels"], result["scores"]))

    matches = []
    for entry in watchlist:
        if scores.get(label_by_id[entry["watchlist_entity_id"]], 0.0) >= _RELEVANCE_THRESHOLD:
            matches.append({
                "watchlist_entity_id": entry["watchlist_entity_id"],
                "canonical_name": entry["canonical_name"],
            })
    return matches
