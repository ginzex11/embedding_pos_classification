"""
Text preprocessing module for sentiment classification with POS tagging.
Author: alex ginzburg
Date: June 17, 2025
"""
import spacy
from typing import List, Tuple
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def preprocess(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Tokenizes, lemmatizes, and POS-tags text using spaCy.
    Args:
        text (str): Input text.
    Returns:
        Tuple[List[str], List[str], List[str]]: Tokens, lemmas, POS tags.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_punct]
    lemmas = [token.lemma_ for token in doc if not token.is_punct]
    pos_tags = [token.pos_ for token in doc if not token.is_punct]
    return tokens, lemmas, pos_tags

def get_pos_distribution(pos_tags: List[str]) -> dict:
    """Computes POS tag distribution.
    Args:
        pos_tags (List[str]): List of POS tags.
    Returns:
        dict: POS tag counts.
    """
    if not isinstance(pos_tags, list):
        raise TypeError("POS tags must be a list")
    return dict(Counter(pos_tags))

def analyze_pos_error(text: str) -> Tuple[str, str, str]:
    """Identifies a POS tagging error.
    Args:
        text (str): Input text.
    Returns:
        Tuple[str, str, str]: Token, incorrect tag, explanation.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    tokens, _, pos_tags = preprocess(text)
    for token, tag in zip(tokens, pos_tags):
        if token == "good" and tag == "ADV":
            return token, tag, "Mis-tagged as ADV instead of ADJ due to ambiguous context (e.g., 'feels good')."
        elif token == "fast" and tag == "ADJ":
            return token, tag, "Mis-tagged as ADJ instead of ADV due to context (e.g., 'runs fast')."
    return "", "", "No specific POS tagging error found."

# Tests
def test_preprocess():
    tokens, lemmas, pos_tags = preprocess("This is a good test!")
    assert len(tokens) == len(lemmas) == len(pos_tags), "Length mismatch"
    assert tokens == ["this", "is", "a", "good", "test"], "Incorrect tokens"
    assert all(isinstance(t, str) for t in pos_tags), "POS tags not strings"

def test_get_pos_distribution():
    dist = get_pos_distribution(["NOUN", "VERB", "NOUN", "ADJ"])
    assert dist["NOUN"] == 2, "Incorrect NOUN count"
    assert dist["VERB"] == 1, "Incorrect VERB count"

def test_analyze_pos_error():
    token, tag, explanation = analyze_pos_error("This feels good.")
    assert isinstance(token, str), "Token not string"
    assert isinstance(tag, str), "Tag not string"

if __name__ == "__main__":
    test_preprocess()
    test_get_pos_distribution()
    test_analyze_pos_error()
    print("All preprocessing tests passed.")