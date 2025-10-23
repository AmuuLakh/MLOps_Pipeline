import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from data_processing import load_data, clean_text, normalize_reviews

def test_clean_text_with_real_data():
    """Test text cleaning using actual loaded dataset content."""
    df = load_data()  # Load your CSV dataset
    assert df is not None, "Dataset could not be loaded."
    assert 'content' in df.columns, "'content' column missing in dataset."

    # Take a sample of reviews (avoid running on full dataset for speed)
    sample_texts = df['content'].dropna().head(5).tolist()
    cleaned_texts = [clean_text(text) for text in sample_texts]

    # Verify outputs are lowercase and cleaned
    for original, cleaned in zip(sample_texts, cleaned_texts):
        assert isinstance(cleaned, str)
        assert cleaned == cleaned.lower(), f"Text not lowercased: {cleaned}"
        assert not any(c in cleaned for c in ['!', '?', ',', '.', ';', ':']), f"Text not cleaned: {cleaned}"

    # Log or print first example for visual verification
    print("Example:", sample_texts[0], "→", cleaned_texts[0])


def test_normalize_reviews_creates_clean_column():
    data = pd.DataFrame({
        "reviewId": [1, 2],
        "content": ["It’s Great!", "I can’t believe it works!!!"]
    })
    cleaned = normalize_reviews(data)
    assert "clean_content" in cleaned.columns
    assert cleaned.loc[0, "clean_content"] == "its great"
    assert cleaned.loc[1, "clean_content"] == "i cannot believe it works"


def test_normalize_reviews_no_content_column(caplog):
    df = pd.DataFrame({"text": ["no content column"]})
    with caplog.at_level("ERROR"):
        cleaned = normalize_reviews(df)

    assert "clean_content" not in cleaned.columns
    assert any("'content' column not found" in msg for msg in caplog.text.splitlines())
