"""
Main pipeline for sentiment classification with word and POS embeddings.
Usage:
- Run with default settings: python main.py
- Specify GloVe dimension: python main.py --glove_dim 50
- Change dataset: python main.py --dataset yelp
- Expected files in resources/: glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt

Dependencies: numpy, matplotlib, seaborn, tensorflow, gensim, scikit-learn, datasets, spacy
Install with: pip install numpy matplotlib seaborn tensorflow gensim scikit-learn datasets spacy
"""

results : 
sst:
    Results Summary
    50D:
        word_only: Accuracy: 0.752, F1: 0.739, Precision: 0.785
        word_pos: Accuracy: 0.749, F1: 0.709, Precision: 0.851
        weighted_word_pos: Accuracy: 0.734, F1: 0.776, Precision: 0.673
    200D:
        word_only: Accuracy: 0.758, F1: 0.740, Precision: 0.805
        word_pos: Accuracy: 0.756, F1: 0.753, Precision: 0.770
        weighted_word_pos: Accuracy: 0.762, F1: 0.763, Precision: 0.763
    300D:
        word_only: Accuracy: 0.772, F1: 0.759, Precision: 0.812
        word_pos: Accuracy: 0.769, F1: 0.754, Precision: 0.814
        weighted_word_pos: Accuracy: 0.760, F1: 0.755, Precision: 0.777
    100D (from earlier):
        word_only: Accuracy: 0.733, F1: 0.710, Precision: 0.784
        word_pos: Accuracy: 0.736, F1: 0.716, Precision: 0.780
        weighted_word_pos: Accuracy: 0.714, F1: 0.699, Precision: 0.744
    Observations:
        Performance generally improves with higher dimensions (300D shows the best accuracy at 0.772 for word_only).
        Weighted word+POS models show varied precision, suggesting POS weighting might need tuning.