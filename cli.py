# cli.py
import argparse

from src.inference import predict_sentiment
from src.logger import log_sentiment


def main():
    parser = argparse.ArgumentParser(description="Sentiment analysis CLI")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")

    args = parser.parse_args()

    sentiment = predict_sentiment(args.text)
    print(sentiment)
    # Log the result to PostgreSQL
    log_sentiment(args.text, sentiment)


if __name__ == "__main__":
    main()
