"""
Test script for preprocessing.py with SST dataset.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import preprocess, get_pos_distribution, analyze_pos_error
from datasets import load_dataset

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

from preprocessing import test_preprocess, test_get_pos_distribution, test_analyze_pos_error
test_preprocess()
test_get_pos_distribution()
test_analyze_pos_error()
print("All tests passed successfully.")