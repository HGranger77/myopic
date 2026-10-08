"""Local abstractive summarization via a distilled BART model."""
from transformers import pipeline

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
_MAX_INPUT_TOKENS = 1024

_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
    return _summarizer


def summarize(text: str) -> str:
    summarizer = _get_summarizer()
    tokenizer = summarizer.tokenizer
    truncated = tokenizer.decode(
        tokenizer.encode(text, truncation=True, max_length=_MAX_INPUT_TOKENS),
        skip_special_tokens=True,
    )
    result = summarizer(truncated, max_length=130, min_length=30, do_sample=False)
    return result[0]["summary_text"].strip()
