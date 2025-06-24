import pandas as pd
import os

# Define paths
input_file1 = r"E:\Afeka\nlp\final_project\embedding_pos_classification\x_tweets_scraper\selenium-twitter-scraper-master\tweets\2025-06-24_16-45-21_tweets_1-50.csv"
input_file2 = r"E:\Afeka\nlp\final_project\embedding_pos_classification\x_tweets_scraper\selenium-twitter-scraper-master\tweets\2025-06-24_17-10-46_tweets_1-1354.csv"
output_dir = r"E:\Afeka\nlp\final_project\embedding_pos_classification\data"
output_file = os.path.join(output_dir, "tweets_1-1404.csv")

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load and merge data
df1 = pd.read_csv(input_file1)
df2 = pd.read_csv(input_file2)
df_combined = pd.concat([df1, df2]).drop_duplicates(subset=["Tweet ID"])

# Save merged file
df_combined.to_csv(output_file, index=False)