"""Per-article sentiment, scoped to one source's framing of a story."""
from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_MAX_INPUT_TOKENS = 512

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)
    return _classifier


def analyze(text: str) -> tuple[str, float]:
    """Returns (label, score) - label is POSITIVE or NEGATIVE."""
    classifier = _get_classifier()
    tokenizer = classifier.tokenizer
    truncated = tokenizer.decode(
        tokenizer.encode(text, truncation=True, max_length=_MAX_INPUT_TOKENS),
        skip_special_tokens=True,
    )
    result = classifier(truncated)[0]
    return result["label"], result["score"]
