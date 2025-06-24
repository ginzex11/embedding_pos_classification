"""
Neural network classifier for sentiment analysis with word and POS embeddings.
Author: Alex Ginzburg
Date: June 17, 2025 (Updated)
"""
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
from typing import Tuple, Dict

def build_model(input_dim: int) -> tf.keras.Model:
    """Builds a feedforward neural network for sentiment classification.
    Args:
        input_dim (int): Input dimension (e.g., 50, 100, 200, or 300 for GloVe embeddings).
    Returns:
        tf.keras.Model: Compiled model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_dim=input_dim),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, X, y, epochs=20):
    """Trains the model with class weighting for imbalance.
    Args:
        model: Keras model.
        X (np.ndarray): Input embeddings.
        y (np.ndarray): Labels.
        epochs (int): Number of epochs.
    Returns:
        model: Trained model.
    """
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    history = model.fit(X, y, epochs=epochs, validation_split=0.2, class_weight=class_weight_dict, verbose=1)
    return model

def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluates model performance.
    Args:
        model (tf.keras.Model): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels (continuous).
    Returns:
        Dict[str, float]: Accuracy, F1, precision.
    """
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_test_binary = (y_test > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test_binary, y_pred),
        "f1": f1_score(y_test_binary, y_pred, zero_division=0),
        "precision": precision_score(y_test_binary, y_pred, zero_division=0)
    }

# Tests (TDD)
def test_build_model():
    model = build_model(100)
    assert len(model.layers) == 3, "Incorrect number of layers"
    assert model.input_shape == (None, 100), "Incorrect input shape"

def test_train_model():
    model = build_model(100)
    X = np.zeros((10, 100))
    y = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.1, 0.9, 0.5, 0.4])
    trained_model = train_model(model, X, y, epochs=1)
    assert trained_model is model, "Training did not return same model"

def test_evaluate_model():
    model = build_model(100)
    X_test = np.zeros((10, 100))
    y_test = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.1, 0.9, 0.5, 0.4])
    metrics = evaluate_model(model, X_test, y_test)
    assert all(k in metrics for k in ["accuracy", "f1", "precision"]), "Missing metrics"
    assert all(0 <= v <= 1 for v in metrics.values()), "Invalid metric values"

if __name__ == "__main__":
    test_build_model()
    test_train_model()
    test_evaluate_model()
    print("All model tests passed.")