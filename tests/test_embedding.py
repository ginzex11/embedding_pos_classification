"""
Test script for embedding.py with SST dataset.
Author: Alex Ginzburg
Date: June 17, 2025
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing import load_data, preprocess
from embedding import load_glove, get_word_emb, get_pos_emb, combine_emb
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pytest

def test_embedding_pipeline():
    texts, _ = load_data()
    tokens, _, pos_tags = zip(*[preprocess(text) for text in texts[:10]])
    
    glove_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'glove.6B.100d.txt'))
    try:
        glove = load_glove(glove_path)
    except FileNotFoundError:
        pytest.skip(f"GloVe file not found at {glove_path}")

    unique_pos_tags = set(tag for tags in pos_tags[:10] for tag in tags)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit([[tag] for tag in unique_pos_tags])
    
    word_emb = get_word_emb(tokens[0], glove)
    assert word_emb.shape == (100,), "Incorrect word embedding shape"

    pos_emb = get_pos_emb(pos_tags[0], encoder)
    assert pos_emb.shape == (10, len(encoder.categories_[0])), f"Incorrect POS embedding shape, got {pos_emb.shape}, expected (10, {len(encoder.categories_[0])})"

    combined = combine_emb(word_emb, pos_emb)
    expected_dim = 100 + len(encoder.categories_[0])
    assert combined.shape == (expected_dim,), f"Incorrect combined shape, got {combined.shape}, expected ({expected_dim},)"
    assert len(combined) == expected_dim, f"Combined length mismatch, got {len(combined)}, expected {expected_dim}"

if __name__ == "__main__":
    pytest.main(["-v"])