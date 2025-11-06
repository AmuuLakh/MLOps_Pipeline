# 🧩 Data Extraction & Processing Module

## Overview
This module is part of the MLOps data pipeline.  
It handles:
1. **Data extraction** from CSV files (automatic delimiter and encoding detection).  
2. **Data cleaning & normalization** for textual reviews, ensuring consistency and readiness for downstream analysis.  
3. **Automated testing** to verify data quality and text preprocessing correctness.

---

## Project Structure
```
project/
├─ data_extraction.py        # Handles dataset loading and logging
├─ data_processing.py        # Cleans and normalizes review text
├─ data/
│   └─ dataset.csv           # Source data file
└─ tests/
    └─ unit/
        ├─ test_data_extraction.py
        └─ test_data_processing.py
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
- Cleans and normalizes review text:
  - Lowercases all text  
  - Removes punctuation, symbols, and extra spaces  
  - Handles emojis (`😊` → `smile`)  
  - Expands contractions (`It’s` → `its`, `can’t` → `cannot`)
- Adds a new column `clean_content` containing cleaned text.

Example:
| content | clean_content |
|----------|----------------|
| It’s Great! ❤️ | its great heart |
| I can’t believe it works!!! | i cannot believe it works |

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

Example test output:
```
==================== 9 passed in 1.27s ====================
```

---

## Example Usage
```python
from data_extraction import load_data
from data_processing import normalize_reviews

# Load dataset
df = load_data()

# Clean reviews
if df is not None:
    cleaned_df = normalize_reviews(df)
    print(cleaned_df[['content', 'clean_content']].head())
```

---

## Logging
Two log files are generated for debugging and transparency:
- `data_load.log` → logs all file loading operations  
- `data_cleaning.log` → logs data cleaning and normalization process  

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
