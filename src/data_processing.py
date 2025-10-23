# data_processing.py
import re
import pandas as pd
import emoji
import logging
from data_extraction import load_data

logger = logging.getLogger('data_cleaning')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler('data_cleaning.log', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text by:
      - Lowercasing
      - Removing punctuation, symbols, and extra spaces
      - Handling emojis
      - Expanding contractions (optional simple handling)
    
    Args:
        text: The input review text
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()

    # Handle emojis (convert to text)
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Replace common contractions
    text = re.sub(r"’", "'", text)
    contractions = {
        "it's": "its",
        "i'm": "im",
        "can't": "cannot",
        "don't": "do not",
        "doesn't": "does not",
        "isn't": "is not",
        "aren't": "are not",
        "won't": "will not",
        "didn't": "did not",
        "you're": "you are",
        "they're": "they are",
        "we're": "we are",
    }
    for k, v in contractions.items():
        text = text.replace(k, v)

    # Remove punctuation and special characters (keep words and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the 'content' column of the DataFrame.
    Logs progress and returns a new DataFrame with cleaned text.
    """
    if 'content' not in df.columns:
        logger.error("'content' column not found in DataFrame.")
        return df
    
    logger.info("Starting text normalization on 'content' column...")
    df['clean_content'] = df['content'].apply(clean_text)
    logger.info("Text normalization complete.")
    
    return df


if __name__ == "__main__":
    data = load_data()
    if data is not None:
        cleaned = normalize_reviews(data)
        print(cleaned[['content', 'clean_content']].head())
