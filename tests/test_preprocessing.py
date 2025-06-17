"""
Test script for preprocessing.py with SST dataset.
"""

from datasets import load_dataset
from preprocessing import preprocess, get_pos_distribution, analyze_pos_error

# Load sample SST data
dataset = load_dataset("sst", split="train[:10]")  # Sample 10 entries
texts = dataset["sentence"]

# Preprocess all texts
tokens, lemmas, pos_tags = zip(*[preprocess(text) for text in texts])
print("Sample tokens:", tokens[0])
print("Sample lemmas:", lemmas[0])
print("Sample POS tags:", pos_tags[0])

# Compute POS distribution
flat_pos_tags = [tag for tags in pos_tags for tag in tags]
dist = get_pos_distribution(flat_pos_tags)
print("POS distribution:", dist)

# Analyze POS error with a crafted example
error_text = "This feels good."  # Likely to trigger "good" mis-tagging
token, tag, explanation = analyze_pos_error(error_text)
print(f"Error: Token '{token}' tagged as '{tag}'. Explanation: {explanation}")

# Run unit tests
from preprocessing import test_preprocess, test_get_pos_distribution, test_analyze_pos_error
test_preprocess()
test_get_pos_distribution()
test_analyze_pos_error()
print("All tests passed successfully.")