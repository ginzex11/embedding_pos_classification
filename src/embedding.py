"""
Embedding module for Project 4: Word embeddings with POS integration.
Author: Alex Ginzburg
Date: June 17, 2025
"""
import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import OneHotEncoder
from typing import List

def load_glove(path: str) -> KeyedVectors:
    """Loads GloVe embeddings from a raw GloVe text file.
    Args:
        path (str): Path to GloVe file (e.g., glove.6B.100d.txt).
    Returns:
        KeyedVectors: Loaded embeddings.
    """
    words = []
    vectors = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                words.append(word)
                vectors.append(vector)
            except ValueError:
                continue
    glove = KeyedVectors(vector_size=len(vectors[0]))
    glove.add_vectors(words, vectors)
    return glove

def get_word_emb(tokens: list, glove: KeyedVectors) -> np.ndarray:
    """Gets the average word embedding for a list of tokens.
    Args:
        tokens (list): List of token strings.
        glove (KeyedVectors): Preloaded GloVe embeddings.
    Returns:
        np.ndarray: Average embedding vector.
    """
    embeddings = [glove[word] if word in glove else np.zeros(glove.vector_size) for word in tokens]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(glove.vector_size)

def get_pos_emb(pos_tags: List[str], encoder: OneHotEncoder) -> np.ndarray:
    """Encodes POS tags as one-hot vectors.
    Args:
        pos_tags (List[str]): POS tags.
        encoder (OneHotEncoder): Fitted encoder.
    Returns:
        np.ndarray: One-hot encoded POS (padded to 10 tags).
    """
    if not pos_tags:
        pos_tags = ["PAD"] * 10
    pos_tags = pos_tags[:10]
    pos_tags += ["PAD"] * (10 - len(pos_tags))
    return encoder.transform([[t] for t in pos_tags])

def combine_emb(word_emb: np.ndarray, pos_emb: np.ndarray) -> np.ndarray:
    """Combines word and POS embeddings.
    Args:
        word_emb (np.ndarray): Word embedding (100D).
        pos_emb (np.ndarray): POS embedding (10 x num_categories).
    Returns:
        np.ndarray: Concatenated embedding (100 + num_categories).
    """
    return np.concatenate([word_emb, np.mean(pos_emb, axis=0)])

# Tests (TDD)
def test_load_glove():
    try:
        glove = load_glove(os.path.join("resources", "glove.6B.100d.txt"))
        assert isinstance(glove, KeyedVectors), "Not a KeyedVectors object"
        assert glove.vector_size == 100, "Incorrect vector size"
    except FileNotFoundError:
        pytest.skip("GloVe file not found")

def test_get_word_emb():
    glove = KeyedVectors(vector_size=100)
    glove.add_vectors(["test"], [np.ones(100)])
    emb = get_word_emb(["test"], glove)
    assert emb.shape == (100,), "Incorrect word embedding shape"
    emb_empty = get_word_emb([], glove)
    assert emb_empty.shape == (100,), "Empty input handling failed"

def test_get_pos_emb():
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit([["NOUN"], ["VERB"], ["ADJ"], ["ADV"], ["PAD"]])
    pos_emb = get_pos_emb(["NOUN", "VERB"], encoder)
    assert pos_emb.shape == (10, len(encoder.categories_[0])), "Incorrect POS embedding shape"
    pos_emb_empty = get_pos_emb([], encoder)
    assert pos_emb_empty.shape == (10, len(encoder.categories_[0])), "Empty POS handling failed"

def test_combine_emb():
    word_emb = np.zeros(100)
    pos_emb = np.ones((10, 5))  # Example with 5 categories
    combined = combine_emb(word_emb, pos_emb)
    assert combined.shape == (100 + 5,), "Incorrect combined shape"
    assert len(combined) == 105, "Combined length mismatch"

if __name__ == "__main__":
    import pytest
    pytest.main(["-v"])
    print("All embedding tests passed.")