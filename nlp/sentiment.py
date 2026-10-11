"""Per-article sentiment, scoped to one source's framing of a story."""
import torch
from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_MAX_INPUT_TOKENS = 512

# Same defensive pin as nlp/summarize.py and nlp/tagging.py - PyTorch's
# thread-count default isn't cgroup-aware. Lower than those two since
# sentiment-article runs CONCURRENTLY with article-entities (both depend on
# tag-relevance), not alone - must match sentiment-article's
# cpu_request/cpu_limit in workflows/myopic_workflow.py.
_N_THREADS = 2

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        torch.set_num_threads(_N_THREADS)
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
