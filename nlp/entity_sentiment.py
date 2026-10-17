"""Aspect-based sentiment: how an article frames one specific entity, not the
article as a whole. nlp/sentiment.py gives one POSITIVE/NEGATIVE read for the
entire article - that collapses a story where one party is praised and
another is criticized into a single, misleading number. This scores each
significant entity (see nlp.matching.filter_significant_entities) on its own.

Validated directly against real news text, not assumed: in a Houthi-missile
story, "Houthis" scored Negative (0.879) while "Saudi Arabia"/"Riyadh" (the
affected party) scored Neutral; in a mixed story, a praised scientist scored
Positive while a criticized company scored Negative (0.991) - genuine
per-entity discrimination within the same sentence, which neither
nlp.sentiment nor nlp.emotion can produce (both score the whole text once).
"""
import torch
from transformers import pipeline

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
_MAX_INPUT_TOKENS = 512

# Same defensive pin as nlp/sentiment.py and nlp/emotion.py - must match
# entity-sentiment's cpu_request/cpu_limit in workflows/myopic_workflow.py.
_N_THREADS = 2

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        torch.set_num_threads(_N_THREADS)
        _classifier = pipeline("text-classification", model=MODEL_NAME, device=-1)
    return _classifier


def analyze(text: str, entity: str) -> tuple[str, float]:
    """Returns (label, score) - label is one of Positive, Negative, Neutral,
    scoped to how `text` frames `entity` specifically, not the text overall.

    truncation/max_length must be passed directly to the pipeline call, not
    pre-applied via a separate tokenizer.encode/decode pass the way
    nlp.sentiment.analyze() does it - this is a sentence-PAIR input (text +
    entity), and pre-truncating just the text leaves no budget accounted
    for the entity's own tokens once the pair is re-encoded together.
    Confirmed directly: passing truncation=True alone (no max_length) is
    silently a no-op here - the tokenizer warns "model has no predefined
    maximum length" and returns the FULL untruncated input - so max_length
    must be given explicitly, every call."""
    classifier = _get_classifier()
    result = classifier(text, text_pair=entity, truncation=True, max_length=_MAX_INPUT_TOKENS)[0]
    return result["label"], result["score"]
