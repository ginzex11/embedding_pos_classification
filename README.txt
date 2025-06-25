# Sentiment Classification with Word and POS Embeddings
ID: 208839613
Name: Alexander Ginzburg
Github: https://github.com/ginzex11/embedding_pos_classification

- Scraper included also in the project under folder -x_tweets_scraper/
- Plots included in plots + backup folder.
- Word/Pdf file of the report included in the project folder named - final_report_208839613_project_4
- tests/ folder was mostly for internal use and testing while wokring on the project, althought all the tests are passing no matter the data yelp or tweets or tts.


small summary:

## Datasets
- **Tweets**: 1,328 samples (160 positive, 1,168 negative), scraped from specific URLs.
- **Yelp**: 1,000 samples (balanced), from `yelp_polarity`.
- **SST**: Stanford Sentiment Treebank, used for comparative analysis.

## Usage
- Run with default settings: `python main.py`
- Specify GloVe dimension: `python main.py --glove_dim 50`
- Change dataset: `python main.py --dataset yelp`
- Limit dataset size: `python main.py --limit 1000`
- Enable normalized confusion matrices: `python main.py --normalize`

## Expected Files
- GloVe embeddings in `resources/`: `glove.6B.50d.txt`, `glove.6B.100d.txt`, `glove.6B.200d.txt`, `glove.6B.300d.txt`.

## Dependencies
- numpy, matplotlib, seaborn, tensorflow, gensim, scikit-learn, datasets, spacy
- Install with: `pip install numpy matplotlib seaborn tensorflow gensim scikit-learn datasets spacy`
 there is an requirements file included in the project if venv installation is needed

## Results Summary
### SST Dataset
- **50D**:
  - Word-only: Accuracy 0.752, F1 0.739, Precision 0.785
  - Word+POS: Accuracy 0.749, F1 0.709, Precision 0.851
  - Weighted Word+POS: Accuracy 0.734, F1 0.776, Precision 0.673
- **100D**:
  - Word-only: Accuracy 0.733, F1 0.710, Precision 0.784
  - Word+POS: Accuracy 0.736, F1 0.716, Precision 0.780
  - Weighted Word+POS: Accuracy 0.714, F1 0.699, Precision 0.744
- **200D**:
  - Word-only: Accuracy 0.758, F1 0.740, Precision 0.805
  - Word+POS: Accuracy 0.756, F1 0.753, Precision 0.770
  - Weighted Word+POS: Accuracy 0.762, F1 0.763, Precision 0.763
- **300D**:
  - Word-only: Accuracy 0.772, F1 0.759, Precision 0.812
  - Word+POS: Accuracy 0.769, F1 0.754, Precision 0.814
  - Weighted Word+POS: Accuracy 0.760, F1 0.755, Precision 0.777

### Observations
- Performance generally improves with higher GloVe dimensions, with 300D achieving the highest accuracy (0.772) for the Word-only model.
- Word+POS models show slightly lower accuracy than Word-only but can improve precision (e.g., 0.851 at 50D, 0.814 at 300D).
- Weighted Word+POS models exhibit varied precision, indicating that POS weighting strategies may require further tuning for consistency.


