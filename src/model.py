import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score, f1_score
from data_processing import split_dataset, normalize_reviews, tokenization
from data_extraction import load_data
import logging
import os

# ---------------- Logging Configuration ----------------
logger = logging.getLogger('model_training')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler('model_training.log', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

# ---------------- Metric Function ----------------
def compute_metrics(eval_pred):
    """Compute accuracy and F1-score for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ---------------- Load and Prepare Data ----------------
def prepare_datasets():
    df = load_data()
    if df is None:
        logger.error("No dataset found.")
        return None, None

    # Use existing 'content' column for text
    if 'content' not in df.columns or 'score' not in df.columns:
        raise ValueError("Dataset must have 'content' and 'score' columns.")
    
    # Convert score (1–5) into sentiment label (0=neg, 1=neutral, 2=pos)
    df['label'] = df['score'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))

    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    # Clean and tokenize
    df = normalize_reviews(df)
    df = tokenization(df)

    # Split dataset
    train_df, eval_df = split_dataset(df)

    from datasets import Dataset
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    return train_dataset, eval_dataset


# ---------------- Main Training Function ----------------
def train_model():
    """Initialize model, configure Trainer, train, and save."""
    logger.info("Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets()
    if train_dataset is None:
        return

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")

    # Save model and tokenizer
    save_dir = "./saved_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Model and tokenizer saved to {save_dir}")

