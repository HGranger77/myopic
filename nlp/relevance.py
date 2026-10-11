"""Watchlist relevance via a lightweight local instruct LLM, replacing NER +
substring matching. One YES/NO question per (article, watchlist entity) pair -
a model this small can't reliably free-form list matches across the whole
watchlist in a single call, but a single yes/no question is parseable.

Model history, for anyone tuning this later:
- Qwen2.5-0.5B-Instruct (full precision): rejected. On the exact
  false-positive case this replaces (an article about Parramatta *council*,
  not the Parramatta Eels NRL team, matched by the old substring matcher) it
  answered YES, and flipped its answer on the genuine positive once asked to
  reason before answering - too unreliable for this judgment.
- Qwen2.5-1.5B-Instruct via transformers (fp32, then bf16): correctly judges
  both cases, but fp32 weights alone are ~6GB - more than this project's
  6GB-capped minikube VM has, and even bf16 only cuts the *idle* footprint;
  generation on this CPU backend upcasts internally and peak RSS still hit
  ~4GB, leaving almost nothing for Postgres/Argo/Grafana running alongside
  it. Loading the full model during the Docker build once pushed the whole
  VM (and nearly the host) into swap thrashing.
- Qwen2.5-1.5B-Instruct, Q4_K_M GGUF via llama.cpp (current): same weights,
  same judgment quality, but quantized and run through an engine built for
  CPU inference - peak RSS ~1.9GB and flat from load through generation,
  comfortably inside the pod's memory budget.
"""
import os

# n_threads defaults to the HOST's full core count (via llama.cpp's own
# hardware-concurrency detection), which has no idea what the pod's cgroup
# actually grants it. Left unset with a 1-core cpu_limit, it spawned 15
# threads fighting over that single core's quota, and the resulting
# scheduler throttling made per-article latency wildly erratic (seconds for
# most articles, minutes for others). This MUST match tag-relevance's
# cpu_request/cpu_limit in workflows/myopic_workflow.py - benchmarked
# locally at 4 threads: ~3.3x faster than 1 (0.527s -> 0.158s/call), real
# parallelism since it's now sized to match genuinely available cores
# instead of oversubscribing a tiny quota.
_N_THREADS = 4

# OMP_NUM_THREADS is a SEPARATE knob from n_threads/n_threads_batch below -
# those only control llama.cpp's own generation threads; the GGML backend's
# BLAS/OpenMP thread pool for matrix ops ignores them entirely and reads
# this env var instead, defaulting to the HOST's full core count regardless
# of n_threads. Must be set before llama_cpp's native library initializes
# its own thread pool (first Llama(...) construction below).
os.environ.setdefault("OMP_NUM_THREADS", str(_N_THREADS))

from huggingface_hub import hf_hub_download  # noqa: E402
from llama_cpp import Llama  # noqa: E402

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
_MAX_INPUT_CHARS = 4000

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        _llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=_N_THREADS,
            n_threads_batch=_N_THREADS,
            verbose=False,
        )
    return _llm


def _is_about(article_text: str, canonical_name: str, aliases: list[str]) -> bool:
    llm = _get_llm()
    other_aliases = [a for a in aliases if a.lower() != canonical_name.lower()]
    alias_hint = f" (also known as: {', '.join(other_aliases)})" if other_aliases else ""
    prompt = (
        f"Article:\n{article_text[:_MAX_INPUT_CHARS]}\n\n"
        f'Question: Is this article substantively about "{canonical_name}"{alias_hint}? '
        "A passing mention of an unrelated thing that merely shares part of the "
        "name does not count. Answer with exactly one word: YES or NO."
    )
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0,
    )
    reply = output["choices"][0]["message"]["content"].strip().upper()
    return reply.startswith("YES")


def check_relevance(article_text: str, watchlist: list[dict]) -> list[dict]:
    """watchlist: [{watchlist_entity_id, canonical_name, aliases: [...]}, ...].

    Returns the subset judged relevant, each as
    {"watchlist_entity_id": ..., "canonical_name": ...}.
    """
    matches = []
    for entry in watchlist:
        if _is_about(article_text, entry["canonical_name"], entry["aliases"]):
            matches.append({
                "watchlist_entity_id": entry["watchlist_entity_id"],
                "canonical_name": entry["canonical_name"],
            })
    return matches
