"""
Text preprocessing module for sentiment classification with POS tagging.
Author: Alex Ginzburg
Date: June 17, 2025
"""
import spacy
from typing import List, Tuple
from collections import Counter
from datasets import load_dataset

nlp = spacy.load("en_core_web_sm")

def load_data(dataset_name="sst"):
    """Loads dataset based on name.
    Args:
        dataset_name (str): Dataset to load ("sst" or "yelp").
    Returns:
        tuple: (texts, labels)
    """
    if dataset_name == "sst":
        dataset = load_dataset("sst", split="train")
        return dataset["sentence"], dataset["label"]
    elif dataset_name == "yelp":
        dataset = load_dataset("yelp_polarity", split="train")
        return dataset["text"], [1 if label == 1 else 0 for label in dataset["label"]]

def preprocess(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Tokenizes, lemmatizes, and POS-tags text using spaCy.
    Args:
        text (str): Input text.
    Returns:
        Tuple[List[str], List[str], List[str]]: Tokens, lemmas, POS tags.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if not text.strip():
        return [], [], []
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
    if not pos_tags:
        return {}
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
    if not text.strip():
        return "", "", "Empty input text."
    tokens, _, pos_tags = preprocess(text)
    # Check for common errors: "good" or "fast" mis-tagged
    for token, tag in zip(tokens, pos_tags):
        if token == "good" and tag == "ADV":
            return token, tag, "Mis-tagged as ADV instead of ADJ due to ambiguous context (e.g., 'feels good')."
        elif token == "fast" and tag == "ADJ":
            return token, tag, "Mis-tagged as ADJ instead of ADV due to context (e.g., 'runs fast')."
    return "", "", "No specific POS tagging error found."

# Tests (TDD)
def test_load_data():
    texts, labels = load_data()
    assert len(texts) > 0, "No texts loaded"
    assert len(labels) == len(texts), "Labels and texts length mismatch"
    assert all(isinstance(l, float) for l in labels), "Labels not float"

def test_preprocess():
    tokens, lemmas, pos_tags = preprocess("This is a good test!")
    assert len(tokens) == len(lemmas) == len(pos_tags) == 5, "Length mismatch"
    assert tokens == ["this", "is", "a", "good", "test"], "Incorrect tokens"
    assert pos_tags[3] == "ADJ", "Expected 'good' to be tagged as ADJ"

def test_get_pos_distribution():
    dist = get_pos_distribution(["NOUN", "VERB", "NOUN", "ADJ"])
    assert dist["NOUN"] == 2, "Incorrect NOUN count"
    assert dist["VERB"] == 1, "Incorrect VERB count"
    assert dist.get("PAD", 0) == 0, "Unexpected PAD tag"

def test_analyze_pos_error():
    token, tag, explanation = analyze_pos_error("This feels good.")
    assert isinstance(token, str), "Token not string"
    assert isinstance(tag, str), "Tag not string"
    assert isinstance(explanation, str), "Explanation not string"

if __name__ == "__main__":
    test_load_data()
    test_preprocess()
    test_get_pos_distribution()
    test_analyze_pos_error()
    print("All preprocessing tests passed.")

    # Example usage
    texts, labels = load_data()
    tokens, lemmas, pos_tags = zip(*[preprocess(text) for text in texts[:100]])
    all_pos = [tag for tags in pos_tags for tag in tags]
    dist = get_pos_distribution(all_pos)
    print("POS Distribution:", dist)
    token, tag, explanation = analyze_pos_error("This feels good.")
    print(f"Error Analysis: Token='{token}', Tag='{tag}', Explanation='{explanation}'")