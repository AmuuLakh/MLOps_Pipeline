from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os

# Get absolute path to the saved_model directory (next to this file)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Define the prediction function
def predict_sentiment(text: str):
    """
    Predict sentiment for a given input text.
    Returns: str -> "Positive", "Negative", or "Neutral"
    """
    if not text.strip():
        return "Neutral"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).flatten()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    # Extract individual class probabilities
    neg_prob = probs[0].item()
    neu_prob = probs[1].item()
    pos_prob = probs[2].item()

    # Get max probability and corresponding label
    max_prob = max(neg_prob, neu_prob, pos_prob)
    
    # If confidence is low (max prob < 0.5), default to Neutral
    # This helps with ambiguous or truly neutral text
    if max_prob < 0.5:
        return "Neutral"
    
    # Otherwise return the class with highest probability
    if max_prob == pos_prob:
        return "Positive"
    elif max_prob == neg_prob:
        return "Negative"
    else:
        return "Neutral"

# Example usage
if __name__ == "__main__":
    while True:
        text = input("Enter text (or 'quit' to exit): ")
        if text.lower() == "quit":
            break
        sentiment = predict_sentiment(text)
        print(f"Sentiment: {sentiment}\n")
