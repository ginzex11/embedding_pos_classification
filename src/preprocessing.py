"""
Text preprocessing module for sentiment classification with POS tagging.
Author: Alex Ginzburg
Date: June 17, 2025
"""
import logging
import spacy
from typing import List, Tuple
from collections import Counter
from datasets import load_dataset
import pandas as pd
import ast  # For safely evaluating string lists

nlp = spacy.load("en_core_web_sm")

def convert_engagement(value: str) -> int:
    """Converts shorthand engagement values (e.g., '1K', '2M') to integers.
    Args:
        value (str): Engagement value as string.
    Returns:
        int: Converted integer value.
    """
    if not isinstance(value, str):
        return 0
    value = value.replace(',', '').upper()
    if value.endswith('K'):
        return int(float(value.replace('K', '')) * 1000)
    elif value.endswith('M'):
        return int(float(value.replace('M', '')) * 1000000)
    elif value.isdigit():
        return int(value)
    return 0

def parse_list_string(s: str) -> List:
    """Parses a string representation of a list (e.g., '[]', '[1, 2]') into a list.
    Args:
        s (str): String to parse.
    Returns:
        List: Parsed list or empty list if invalid.
    """
    if not isinstance(s, str) or s.strip() == '[]':
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

def load_data(dataset_name="sst", limit=None):
    """Loads dataset based on name.
    Args:
        dataset_name (str): Dataset to load ("sst" or "yelp" or "tweets").
        limit (int): Optional limit on dataset size.
    Returns:
        tuple: (texts, labels)
    """
    data = None
    if dataset_name == "sst":
        dataset = load_dataset("sst", split="train")
        data = dataset["sentence"], dataset["label"]
    elif dataset_name == "yelp":
        dataset = load_dataset("yelp_polarity", split="train")
        data = dataset["text"], [1 if label == 1 else 0 for label in dataset["label"]]
    if limit and data:
        data = (data[0][:limit], data[1][:limit])
    if dataset_name == "tweets":
        df = pd.read_csv(r"E:\Afeka\nlp\final_project\embedding_pos_classification\data\tweets_1-1404.csv")
        if limit:
            df = df.head(limit)
        df = df.dropna(subset=["Content"])
        for col in ["Likes", "Retweets", "Comments"]:
            df[col] = df[col].apply(convert_engagement)
        for col in ["Tags", "Mentions", "Emojis"]:
            df[col] = df[col].apply(parse_list_string)
        def get_sentiment(text):
            if not isinstance(text, str):
                return 0
            text = text.lower()
            positive_words = {"good", "great", "win", "peace", "happy", "success", "love", "amazing", "best", "joy"}
            negative_words = {"bad", "hell", "fail", "dead", "war", "hate", "worst", "tragedy", "loss", "sad"}
            if any(word in text for word in positive_words):
                return 1
            elif any(word in text for word in negative_words):
                return 0
            return 0  # Default to negative if no strong signal
        labels = [get_sentiment(c) for c in df["Content"]]
        logging.info(f"Label distribution: Positive={sum(labels)}, Negative={len(labels) - sum(labels)}")
        data = (df["Content"].tolist(), labels)
    if data is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return data

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
    assert all(isinstance(l, int) for l in labels), "Labels not integer"
    texts_limited, labels_limited = load_data(limit=10)
    assert len(texts_limited) == 10, "Limit not applied"

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