# data_processing.py
import re
import pandas as pd
import emoji
import logging
from data_extraction import load_data
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# ---------------- Logging Configuration ----------------
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

# ---------------- Text Cleaning ----------------
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

# ---------------- Normalization ---------------
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

# ---------------- Tokenization ----------------
def tokenization(df: pd.DataFrame) -> pd.DataFrame:
    if 'clean_content' not in df.columns:
        raise ValueError("Expected 'clean_content' column. Did you run normalize_reviews()?")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    logger.info("Starting tokenization...")

    encoded = tokenizer(
        df["clean_content"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )

    df["input_ids"] = encoded["input_ids"]
    df["attention_mask"] = encoded["attention_mask"]
    df["token_length"] = df["input_ids"].apply(len)

    logger.info("Tokenization complete.")
    return df

# ---------------- Data Splitting ----------------
def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and validation sets."""
    logger.info(f"Splitting dataset into {int((1-test_size)*100)}/{int(test_size*100)} train/validation...")
    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    logger.info(f"Train size: {len(train_df)}, Validation size: {len(eval_df)}")
    return train_df, eval_df


if __name__ == "__main__":
    data = load_data()
    if data is not None:
        cleaned = normalize_reviews(data)
        tokenized = tokenization(cleaned)

        train_dataset, eval_dataset = split_dataset(tokenized)

        train_dataset.to_csv("data/processed/train_tokenized.csv", index=False)
        eval_dataset.to_csv("data/processed/eval_tokenized.csv", index=False)

        logger.info("Saved tokenized train and eval datasets.")
        print("✅ Data processing complete. Files saved to data/processed/")
        print(f"Train: {len(train_dataset)} rows | Eval: {len(eval_dataset)} rows")