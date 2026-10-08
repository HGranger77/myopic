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
