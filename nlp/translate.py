"""Translates non-English article text to English via a small MarianMT
model - needed because every other model in this pipeline (embedding,
summarization, sentiment, NER, zero-shot tagging) is English-only, and
chinanews.com.cn (see config/sources.yaml's per-source `language`) is this
project's first non-English source.

This is autoregressive, the same cost class as nlp/relevance.py's LLM, NOT
nlp/tagging.py's one-shot classifier - measured directly on a real scraped
article (3546 chars, 49 paragraphs): ~70-135s for the full text depending on
batching strategy (noisy measurements, but consistently in the
minutes-per-article range, never seconds). Per-paragraph batching (one
`translator(list_of_paragraphs, batch_size=8)` call) measured faster than
combining paragraphs into fewer, longer chunks first - plausible explanation
is that total decode steps scales with total output length either way, so
there's no free lunch from combining; a bigger batch of shorter sequences
just parallelizes better. Input is truncated to _MAX_INPUT_CHARS for the
same reason nlp/tagging.py and nlp/summarize.py bound their own inputs -
nothing downstream reads more of the body than that anyway.

n_threads is pinned the same defensive way as nlp/summarize.py and
nlp/tagging.py (see those modules) - PyTorch's thread-count default isn't
cgroup-aware, and this is the fourth model to need the same fix, not the
first."""
import torch
from transformers import pipeline

MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
_MAX_INPUT_CHARS = 2000
_N_THREADS = 4  # must match translate-article's cpu_request/cpu_limit in workflows/myopic_workflow.py

_translator = None


def _get_translator():
    global _translator
    if _translator is None:
        torch.set_num_threads(_N_THREADS)
        _translator = pipeline("translation", model=MODEL_NAME, device=-1)
    return _translator


def translate_to_english(text: str) -> str:
    if not text:
        return ""
    translator = _get_translator()
    paragraphs = [p for p in text[:_MAX_INPUT_CHARS].split("\n") if p.strip()]
    if not paragraphs:
        return ""
    results = translator(paragraphs, max_length=512, batch_size=8)
    return "\n\n".join(r["translation_text"] for r in results)
