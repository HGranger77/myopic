"""Per-article emotion, alongside (not instead of) the POSITIVE/NEGATIVE
sentiment in nlp/sentiment.py. Positive/negative alone collapses two very
different stories into the same badge - a corruption scandal and a natural
disaster both just read as NEGATIVE - this adds the finer-grained read
(anger, disgust, fear, joy, neutral, sadness, surprise) a reader actually
wants from a news story.
"""
import torch
from transformers import pipeline

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
_MAX_INPUT_TOKENS = 512

# Same defensive pin as nlp/sentiment.py, and for the same reason: this
# loads in the SAME process/pod as nlp.sentiment (sentiment-article runs
# both back to back on every article), so the combined thread budget must
# still match sentiment-article's cpu_request/cpu_limit in
# workflows/myopic_workflow.py, not double it.
_N_THREADS = 2

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        torch.set_num_threads(_N_THREADS)
        _classifier = pipeline("text-classification", model=MODEL_NAME, device=-1)
    return _classifier


def analyze(text: str) -> tuple[str, float]:
    """Returns (label, score) - label is one of anger, disgust, fear, joy,
    neutral, sadness, surprise."""
    classifier = _get_classifier()
    tokenizer = classifier.tokenizer
    truncated = tokenizer.decode(
        tokenizer.encode(text, truncation=True, max_length=_MAX_INPUT_TOKENS),
        skip_special_tokens=True,
    )
    result = classifier(truncated)[0]
    return result["label"], result["score"]
