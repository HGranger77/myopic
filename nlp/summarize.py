"""Local abstractive summarization via a distilled BART model.

Greedy decoding (do_sample=False, the default num_beams=1) has no built-in
defense against getting stuck repeating a short phrase - a known failure
mode of autoregressive generation. Hit this for real: one story's summary
came out as "...the first and the first-and've'I've been a member of the
public and the public has been the first of the..." on otherwise-clean
English input that re-summarizes fine on retry, so it reads as the greedy
search occasionally tipping into a repetition loop rather than anything
wrong with the input - see no_repeat_ngram_size below, the standard
mitigation. (A second, much larger source of garbled summaries turned out
to be a separate bug entirely - untranslated non-English articles reaching
this English-only model at all; see
database.operations.get_articles_needing_embedding's docstring. This
repetition guard doesn't fix that class - fed raw Chinese text, it still
produces nonsense, just not a repetition loop - the real fix there is never
letting that text arrive here untranslated.)
"""
import torch
from transformers import pipeline

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
_MAX_INPUT_TOKENS = 1024

# PyTorch's thread-count default is NOT cgroup-aware (hit twice already with
# llama.cpp's n_threads/OMP_NUM_THREADS and nlp/tagging.py's torch default -
# see those modules). Confirmed here too: with summarize-story's old 1-core
# limit, this spawned 19 OS threads fighting over that single core, and
# summarize-story only ever ran on a handful of relevant stories before the
# DAG reorder - cheap enough that the oversubscription cost never surfaced.
# Running on every clustered story now, it did: 22/414 stories summarized in
# 12 minutes (~3.7h projected) before this fix. Must match summarize-story's
# cpu_request/cpu_limit in workflows/myopic_workflow.py.
_N_THREADS = 4

_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        torch.set_num_threads(_N_THREADS)
        _summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
    return _summarizer


def summarize(text: str) -> str:
    summarizer = _get_summarizer()
    tokenizer = summarizer.tokenizer
    truncated = tokenizer.decode(
        tokenizer.encode(text, truncation=True, max_length=_MAX_INPUT_TOKENS),
        skip_special_tokens=True,
    )
    result = summarizer(
        truncated, max_length=130, min_length=30, do_sample=False, no_repeat_ngram_size=3
    )
    return result[0]["summary_text"].strip()
