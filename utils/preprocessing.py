import re

from pyvi import ViTokenizer


def normalize_text(text):
    """Normalize raw text before tokenization."""
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # URL
    text = re.sub(r"[@#]\S+", "", text)  # Mention, hashtag
    text = re.sub(r"[^\w\s]", "", text)  # Special characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_vietnamese(text):
    """Tokenize Vietnamese text with PyVi."""
    if not text:
        return []
    return ViTokenizer.tokenize(text).split()


def preprocess_pipeline(text):
    """Run the preprocessing pipeline for a single text."""
    text = normalize_text(text)
    tokens = tokenize_vietnamese(text)
    return tokens


def load_and_preprocess(texts):
    """Preprocess a batch of texts."""
    return [preprocess_pipeline(text) for text in texts]
