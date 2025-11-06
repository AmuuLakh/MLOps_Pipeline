# 🧩 Data Extraction & Processing Module

## Overview
This project is part of a collaborative sentiment analysis pipeline using a fine-tuned BERT model for text classification.
It demonstrates an MLOps-driven approach integrating data processing, model training, inference, testing, and CI/CD.

This module primarily covers data extraction, data cleaning, tokenization, and supports model training and inference
Core Responsibilities:
1. **Data extraction** Safely load and validate text datasets.  
2. **Data cleaning & normalization** Text preparation and standardization for NLP tasks. 
3. **Model Training** Fine-tune a pretrained BERT model for sentiment classification.
4. **Inference** Deploy the model to predict sentiment on unseen text.
5. **Automated Testing & Logging** Ensure data and model reliability.

---

## Project Structure
```
project/
├─ requirements.txt
├─ data_cleaning.log
├─ data_log.log
├─ model_training.log
├─ src/
│  ├─ data_extraction.py        # Handles dataset loading and logging
│  ├─ data_processing.py        # Cleans and normalizes textual reviews
│  ├─ model.py                  # Fine-tunes BERT
│  ├─ inference.py              # Predicts sentiment on new text
│  ├─ data/
│     ├─ processed/
│     │   ├─ eval_tokenized.csv     # evaluation data file
      │   └─ train_tokenized.csv    # training data file
│     └─ dataset.csv           # Source data file
└─ tests/
    └─ unit/
        ├─ test_data_extraction.py
        ├─ test_data_processing.py
        ├─ test_model.py
        └─ test_inference.py
```

---

## Features

### `data_extraction.py`
- Loads CSV datasets safely with:
  - Automatic delimiter detection (`csv.Sniffer`)
  - Fallback encoding handling (UTF-8 → Latin-1)
  - Logging for missing or empty files
- Returns a clean **pandas DataFrame**.

### `data_processing.py`
- Focuses on cleaning and normalizing textual data for sentiment analysis :
  - Lowercases all text  
  - Removes punctuation, symbols, and extra spaces  
  - Converts emojis to descriptive words (`😊` → `smile`)  
  - Expands contractions (`It’s` → `its`, `can’t` → `cannot`)
- Adds a new column `clean_content` containing cleaned text.

Example:
| content | clean_content |
|----------|----------------|
| It’s Great! ❤️ | its great heart |
| I can’t believe it works!!! | i cannot believe it works |

### `model.py`
- Handles BERT model setup, training, and evaluation.
- **Key Features:**
  - Loads a pretrained transformer model using `AutoModelForSequenceClassification` (e.g., `bert-base-uncased`)
  - Uses the Hugging Face Trainer API for streamlined fine-tuning  
  - Implements evaluation metrics (accuracy, precision, recall, F1)  
  - Supports GPU acceleration (CUDA) if available
  - Saves the fine-tuned model and tokenizer to /saved_models
- **Training Flow**:
1. Load tokenized datasets.
2. Initialize model and tokenizer.
3. Fine-tune on labeled sentiment data.
4. Save trained model and logs.

- **Outputs:**
```
  saved_models/
├─ config.json
├─ tokenizer_config.json
├─ special_token_map.json
├─ model.safetensors
├─ vocab.txt
└─ tokenizer.json
```


### `inference.py`
- Provides an interface for running predictions using the fine-tuned BERT model :
  - Loads the trained model and tokenizer automatically
  - Preprocesses input text with the same normalization as training 
  - Returns sentiment predictions (Positive, Negative, or Neutral) 
  - Supports both single-text and batch inference
- **Example Usage:**
``` python
from inference import predict_sentiment

text = "I absolutely loved this movie!"
print(predict_sentiment(text))
# Output: {'label': 'POSITIVE', 'score': 0.98
```
---

## Testing

All modules are unit-tested using **pytest**.

### Run all tests
```bash
pytest -v
```

### Test coverage
| Test File | Purpose |
|------------|----------|
| `test_data_extraction.py` | Verifies CSV loading, delimiter detection, and error handling |
| `test_data_processing.py` | Ensures text normalization works and logs missing columns |
| `test_model.py` | Checks model initialization, training pipeline, and output tensor shape |
| `test_inference.py` | Validates prediction logic and output labels |

Example test output:
```
==================== 12 passed in 1.27s ====================
```

---

## Example Usage
```python
from data_extraction import load_data
from data_processing import normalize_reviews
from inference import predict_sentiment

# Load dataset
df = load_data('data/dataset.csv')
cleaned_df = normalize_reviews(df)

# Run a sample prediction
sample = "This product is absolutely amazing!"
result = predict_sentiment(sample)
print(result)
```

---

## Logging
Three log files are generated for debugging and transparency:
- `data_load.log` → logs all file loading operations  
- `data_cleaning.log` → logs data cleaning and normalization process  
- `model_training.log` → logs model training process  

---

## Verification Example
Normalization example confirming expected output:

```text
Input:  “It’s Great!”
Output: “its great”
```

---

## Setup & Dependencies
Make sure to install the required dependencies before running the scripts:

```bash
pip install -r requirements.txt
```

Python 3.12+ is recommended.
