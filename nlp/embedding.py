"""Sentence-embedding model for cross-source story clustering."""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(text: str) -> list[float]:
    vector = _get_model().encode(text, normalize_embeddings=True)
    return vector.tolist()


_BATCH_SIZE = 16


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embeds many texts in one encode() call. Measured ~2.5x faster than a
    per-text loop on CPU (60 headline+lead texts, 2 threads: 2.2s -> 0.8s) -
    the texts are short, so per-call overhead dominates the forward pass.
    Vectors match embed() to ~1e-7. 16 beat 32 in the same benchmark."""
    if not texts:
        return []
    vectors = _get_model().encode(texts, batch_size=_BATCH_SIZE, normalize_embeddings=True)
    return vectors.tolist()
