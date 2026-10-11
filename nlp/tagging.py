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
Unlike nlp/relevance.py, this has no natural place to fold in aliases (a
candidate label is just a short phrase, not a sentence) - matching is on
canonical_name only.
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
    labels = [entry["canonical_name"] for entry in watchlist]
    result = classifier(text[:_MAX_INPUT_CHARS], candidate_labels=labels, multi_label=True)
    scores = dict(zip(result["labels"], result["scores"]))

    matches = []
    for entry in watchlist:
        if scores.get(entry["canonical_name"], 0.0) >= _RELEVANCE_THRESHOLD:
            matches.append({
                "watchlist_entity_id": entry["watchlist_entity_id"],
                "canonical_name": entry["canonical_name"],
            })
    return matches
