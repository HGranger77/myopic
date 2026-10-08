"""spaCy-based named entity extraction."""
import spacy
from nlp.matching import normalize

MODEL_NAME = "en_core_web_sm"
_nlp = None


def _get_model():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(MODEL_NAME)
    return _nlp


def extract_entities(text: str) -> list[dict]:
    """Returns [{text, normalized_text, label, start_char, end_char}, ...]."""
    if not text:
        return []
    doc = _get_model()(text)
    return [
        {
            "text": ent.text,
            "normalized_text": normalize(ent.text),
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
    ]
