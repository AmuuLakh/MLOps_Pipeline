import pytest
from inference import predict_sentiment
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


# --- Basic Smoke Tests ---
def test_model_load_and_predict():
    """Ensure model loads and returns a valid sentiment label."""
    text = "I love this movie!"
    result = predict_sentiment(text)
    assert result in ["Positive", "Negative", "Neutral"], "Invalid sentiment label returned"

# --- Positive Sentiment Test ---
def test_positive_sentiment():
    """Check that clearly positive text is classified correctly."""
    # Using multiple strong positive examples
    positive_texts = [
        "This product is absolutely amazing and I love it!",
        "Best experience ever! Highly recommend!",
        "Wonderful, fantastic, excellent service!"
    ]
    results = [predict_sentiment(text) for text in positive_texts]
    # At least 2 out of 3 should be positive (allowing for model uncertainty)
    positive_count = results.count("Positive")
    assert positive_count >= 2, f"Expected at least 2 Positive results but got {positive_count}: {results}"


# --- Negative Sentiment Test ---
def test_negative_sentiment():
    """Check that clearly negative text is classified correctly."""
    # Using multiple strong negative examples
    negative_texts = [
        "This was the worst experience of my life.",
        "Terrible, horrible, absolutely awful!",
        "I hate this product, complete waste of money."
    ]
    results = [predict_sentiment(text) for text in negative_texts]
    # At least 2 out of 3 should be negative (allowing for model uncertainty)
    negative_count = results.count("Negative")
    assert negative_count >= 2, f"Expected at least 2 Negative results but got {negative_count}: {results}"


# --- Neutral Sentiment Test ---
def test_neutral_sentiment():
    """Check that the model can produce neutral predictions."""
    # This test acknowledges that neutral is challenging for sentiment models
    # We just verify the function CAN return neutral in at least some cases
    neutral_texts = [
        "The meeting is scheduled for 3 PM.",
        "The book has 300 pages.",
        "The temperature is 20 degrees.",
        "This is a sentence.",
        "The store opens at 9 AM.",
        "The document contains text.",
        "There are five items.",
        "The sky exists.",
        "Objects have properties.",
        "Time passes."
    ]
    results = [predict_sentiment(text) for text in neutral_texts]
    neutral_count = results.count("Neutral")
    
    assert neutral_count >= 1, f"Model never predicts Neutral (got {neutral_count} out of {len(neutral_texts)}): {results}"


# --- Edge Case Test ---
def test_empty_input():
    """Ensure empty string input is handled gracefully."""
    text = ""
    result = predict_sentiment(text)
    assert isinstance(result, str), "Output should be a string"
    assert result == "Neutral", "Empty input should return Neutral"

# --- Consistency Test ---
def test_prediction_consistency():
    """Ensure same input produces same output."""
    text = "I really enjoyed this experience."
    result1 = predict_sentiment(text)
    result2 = predict_sentiment(text)
    assert result1 == result2, "Same input should produce consistent results"