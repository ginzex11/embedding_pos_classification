"""
Test script for preprocessing.py with SST dataset.
Author: Alex Ginzburg
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing import load_data, preprocess, get_pos_distribution, analyze_pos_error
from datasets import load_dataset
import pytest

def test_preprocessing_pipeline():
    texts, labels = load_data()
    subset_size = 10
    tokens, lemmas, pos_tags = zip(*[preprocess(text) for text in texts[:subset_size]])
    assert len(tokens) == subset_size, "Incorrect number of processed texts"
    assert all(len(t) == len(l) == len(p) for t, l, p in zip(tokens, lemmas, pos_tags)), "Length mismatch"
    
    flat_pos_tags = [tag for tags in pos_tags for tag in tags]
    dist = get_pos_distribution(flat_pos_tags)
    assert isinstance(dist, dict), "Distribution not a dict"
    assert sum(dist.values()) == len(flat_pos_tags), "Distribution count mismatch"
    
    error_text = "It runs fast."
    token, tag, explanation = analyze_pos_error(error_text)
    assert isinstance(token, str), "Token not string"
    assert isinstance(explanation, str), "Explanation not string"

if __name__ == "__main__":
    dataset = load_dataset("sst", split="train[:10]")
    texts = dataset["sentence"]
    tokens, lemmas, pos_tags = zip(*[preprocess(text) for text in texts])
    print("Sample tokens:", tokens[0])
    print("Sample lemmas:", lemmas[0])
    print("Sample POS tags:", pos_tags[0])
    
    flat_pos_tags = [tag for tags in pos_tags for tag in tags]
    dist = get_pos_distribution(flat_pos_tags)
    print("POS distribution:", dist)
    
    error_text = "It runs fast."
    token, tag, explanation = analyze_pos_error(error_text)
    print(f"Error: Token '{token}' tagged as '{tag}'. Explanation: {explanation}")
    
    pytest.main(["-v"])
    print("All preprocessing tests passed.")