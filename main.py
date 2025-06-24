"""
Main pipeline for sentiment classification with word and POS embeddings.
Author: Alex Ginzburg
Date: June 23, 2025 (Updated)

Usage:
- Run with default settings: python main.py
- Specify GloVe dimension: python main.py --glove_dim 50
- Change dataset: python main.py --dataset yelp

Dependencies: numpy, matplotlib, seaborn, tensorflow, gensim, scikit-learn, datasets, spacy
Install with: pip install numpy matplotlib seaborn tensorflow gensim scikit-learn datasets spacy
"""
import sys
import os
# Add src directory to sys.path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from preprocessing import load_data, preprocess, get_pos_distribution
from embedding import load_glove, get_word_emb, get_pos_emb, combine_emb
from model import build_model, train_model, evaluate_model
from gensim.models import KeyedVectors
import pytest
import logging

print(sys.path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_pos_weights(pos_tags: list) -> np.ndarray:
    """Calculates inverse frequency weights for POS tags.
    Args:
        pos_tags (list): List of all POS tags from dataset.
    Returns:
        np.ndarray: Weights for each POS category.
    """
    if not pos_tags:
        return np.ones(5)  # Default equal weights if empty
    dist = get_pos_distribution(pos_tags)
    total = sum(dist.values())
    weights = {tag: total / (count * len(dist)) for tag, count in dist.items()}
    from sklearn.preprocessing import OneHotEncoder  # Moved here to avoid import if not needed
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit([[tag] for tag in dist.keys()])
    return np.array([weights.get(tag, 1.0) for tag in encoder.categories_[0]])

def run_pipeline(glove_path: str, dataset_name: str = "sst", epochs: int = 10, glove_dim: int = 100) -> dict:
    """Runs the sentiment classification pipeline.
    Args:
        glove_path (str): Path to GloVe embeddings.
        dataset_name (str): Dataset name (default: 'sst', also supports 'yelp').
        epochs (int): Training epochs (default: 10).
        glove_dim (int): Dimension of GloVe embeddings (default: 100).
    Returns:
        dict: Dictionary containing evaluation metrics, labels, models, and embeddings.
    """
    logging.info("Loading data...")
    texts, labels = load_data(dataset_name)
    labels = np.array(labels)

    # Preprocess
    logging.info("Preprocessing texts...")
    tokens, _, pos_tags = zip(*[preprocess(text) for text in texts])
    flat_pos_tags = [tag for tags in pos_tags for tag in tags]

    # Load GloVe and POS encoder
    logging.info("Loading GloVe embeddings...")
    try:
        glove = load_glove(glove_path)
    except FileNotFoundError:
        logging.error("GloVe file not found")
        raise
    from sklearn.preprocessing import OneHotEncoder  # Moved here to avoid import if not needed
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit([[tag] for tag in set(flat_pos_tags)])

    # Generate embeddings
    logging.info("Generating embeddings...")
    word_embeddings = np.array([get_word_emb(t, glove) for t in tokens])
    pos_embeddings = np.array([get_pos_emb(p, encoder) for p in pos_tags])
    combined_embeddings = np.array([combine_emb(w, p) for w, p in zip(word_embeddings, pos_embeddings)])

    # POS weighting
    logging.info("Calculating POS weights...")
    pos_weights = create_pos_weights(flat_pos_tags)
    weighted_pos_embeddings = np.array([
        combine_emb(w, p * pos_weights[np.newaxis, :]) for w, p in zip(word_embeddings, pos_embeddings)
    ])

    # Train and evaluate models
    logging.info("Training and evaluating word-only model...")
    word_model = build_model(input_dim=glove_dim)
    word_model = train_model(word_model, word_embeddings, labels, epochs)
    word_metrics = evaluate_model(word_model, word_embeddings, labels)

    logging.info("Training and evaluating word+POS model...")
    pos_model = build_model(input_dim=glove_dim + len(encoder.categories_[0]))
    pos_model = train_model(pos_model, combined_embeddings, labels, epochs)
    pos_metrics = evaluate_model(pos_model, combined_embeddings, labels)

    logging.info("Training and evaluating weighted word+POS model...")
    weighted_model = build_model(input_dim=glove_dim + len(encoder.categories_[0]))
    weighted_model = train_model(weighted_model, weighted_pos_embeddings, labels, epochs)
    weighted_metrics = evaluate_model(weighted_model, weighted_pos_embeddings, labels)

    results = {
        "word_only": word_metrics,
        "word_pos": pos_metrics,
        "weighted_word_pos": weighted_metrics
    }

    # Return a dictionary with all necessary data
    return {
        "results": results,
        "labels": labels,
        "word_model": word_model,
        "word_embeddings": word_embeddings,
        "pos_model": pos_model,
        "combined_embeddings": combined_embeddings,
        "weighted_model": weighted_model,
        "weighted_pos_embeddings": weighted_pos_embeddings
    }

def plot_metrics(results: dict, glove_dim: int, dataset_name: str):
    """Plots comparison of model metrics.
    Args:
        results (dict): Dictionary of model metrics.
        glove_dim (int): GloVe dimension for filename.
        dataset_name (str): Dataset name for filename.
    """
    metrics = ["accuracy", "f1", "precision"]
    models = list(results.keys())
    data = [[results[model][metric] for metric in metrics] for model in models]
    
    fig, ax = plt.subplots()
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(models):
        ax.bar(x + i * width, data[i], width, label=model)
    
    ax.set_ylabel("Score")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/metrics_comparison_{dataset_name}_dim{glove_dim}.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title, glove_dim: int, dataset_name: str):
    """Plots confusion matrix for model predictions.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        title (str): Plot title.
        glove_dim (int): GloVe dimension for filename.
        dataset_name (str): Dataset name for filename.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/confusion_matrix_{title}_{dataset_name}_dim{glove_dim}.png")
    plt.close()

# Tests (TDD)
def test_pipeline():
    texts, labels = load_data()
    tokens, _, pos_tags = zip(*[preprocess(text) for text in texts[:10]])
    glove = KeyedVectors(vector_size=100)
    glove.add_vectors(["test"], [np.ones(100)])
    from sklearn.preprocessing import OneHotEncoder  # Moved here to avoid import if not needed
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit([["NOUN"], ["VERB"], ["ADJ"], ["ADV"], ["PAD"]])

    word_emb = np.array([get_word_emb(t, glove) for t in tokens])
    pos_emb = np.array([get_pos_emb(p, encoder) for p in pos_tags])
    combined_emb = np.array([combine_emb(w, p) for w, p in zip(word_emb, pos_emb)])
    weights = create_pos_weights([tag for tags in pos_tags for tag in tags])

    assert word_emb.shape[1] == 100, "Word embedding dimension incorrect"
    assert pos_emb.shape[1] == len(encoder.categories_[0]), "POS embedding dimension incorrect"
    assert combined_emb.shape[1] == 100 + len(encoder.categories_[0]), "Combined embedding dimension incorrect"
    assert len(weights) == len(encoder.categories_[0]), "POS weights length mismatch"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sentiment classification pipeline")
    parser.add_argument("--glove_dim", type=int, choices=[50, 100, 200, 300], default=100, help="GloVe dimension")
    parser.add_argument("--dataset", type=str, choices=["sst", "yelp"], default="sst", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Dynamically set glove_path based on glove_dim
    glove_paths = {
        50: "resources/glove.6B.50d.txt",
        100: "resources/glove.6B.100d.txt",
        200: "resources/glove.6B.200d.txt",
        300: "resources/glove.6B.300d.txt"
    }
    args.glove_path = glove_paths.get(args.glove_dim, "resources/glove.6B.100d.txt")

    # Run the pipeline and get the dictionary
    pipeline_data = run_pipeline(args.glove_path, args.dataset, args.epochs, args.glove_dim)
    
    # Extract variables from the dictionary
    results = pipeline_data["results"]
    labels = pipeline_data["labels"]
    word_model = pipeline_data["word_model"]
    word_embeddings = pipeline_data["word_embeddings"]
    pos_model = pipeline_data["pos_model"]
    combined_embeddings = pipeline_data["combined_embeddings"]
    weighted_model = pipeline_data["weighted_model"]
    weighted_pos_embeddings = pipeline_data["weighted_pos_embeddings"]

    # Plot metrics with glove_dim and dataset_name in filename
    plot_metrics(results, args.glove_dim, args.dataset)

    # Plot confusion matrices with glove_dim and dataset_name in filename
    y_test_binary = (labels > 0.5).astype(int)
    for model_name, (model, X) in [("word_only", (word_model, word_embeddings)), 
                                   ("word_pos", (pos_model, combined_embeddings)), 
                                   ("weighted_word_pos", (weighted_model, weighted_pos_embeddings))]:
        y_pred = (model.predict(X) > 0.5).astype(int)
        plot_confusion_matrix(y_test_binary, y_pred, model_name, args.glove_dim, args.dataset)

    logging.info("Results: %s", results)

    pytest.main(["-v"])
    logging.info("All pipeline tests passed.")