import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from data_processing import load_data, clean_text, normalize_reviews, tokenization

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


def test_tokenization_adds_columns():
    """Ensure tokenization adds input_ids, attention_mask, and token_length columns."""
    df = pd.DataFrame({"clean_content": ["hello world"]})
    result = tokenization(df)

    assert "input_ids" in result.columns, "input_ids column missing"
    assert "attention_mask" in result.columns, "attention_mask column missing"
    assert "token_length" in result.columns, "token_length column missing"

    # Single row checks
    assert isinstance(result.loc[0, "input_ids"], list)
    assert isinstance(result.loc[0, "attention_mask"], list)
    assert result.loc[0, "token_length"] == len(result.loc[0, "input_ids"])


def test_tokenization_padding_and_mask():
    """Test correct padding behavior and matching attention mask."""
    df = pd.DataFrame({"clean_content": ["hello world"]})
    result = tokenization(df)

    input_ids = result.loc[0, "input_ids"]
    attention_mask = result.loc[0, "attention_mask"]

    # Length must be max_length (512)
    assert len(input_ids) == 512
    assert len(attention_mask) == 512

    # First padding index
    pad_index = input_ids.index(0)  # 0 = pad token ID for BERT-uncased

    # Check masks before and after padding
    assert all(m == 1 for m in attention_mask[:pad_index])
    assert all(m == 0 for m in attention_mask[pad_index:])


def test_tokenization_missing_clean_content():
    """Tokenization should raise an error when clean_content column is missing."""
    df = pd.DataFrame({"content": ["missing clean column"]})

    with pytest.raises(ValueError) as exc_info:
        tokenization(df)

    assert "clean_content" in str(exc_info.value)
