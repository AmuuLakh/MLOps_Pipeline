# 🚀 MLOps Sentiment Analysis Pipeline

## Overview
This project is a comprehensive **MLOps pipeline** for sentiment analysis using a fine-tuned BERT model for text classification.
It demonstrates production-ready practices integrating data processing, model training, inference, automated testing, containerization, database logging, and CI/CD workflows.

### Core Capabilities:
1. **Data Extraction & Processing** - Safely load, clean, and normalize text datasets for NLP tasks
2. **Model Training** - Fine-tune pretrained BERT models for sentiment classification
3. **Inference Engine** - Deploy models to predict sentiment on unseen text via CLI
4. **Database Logging** - Persist predictions to PostgreSQL for analytics and monitoring
5. **Containerization** - Docker-based deployment for reproducibility and scalability
6. **Automated Testing** - Comprehensive unit tests with coverage reporting
7. **CI/CD Pipeline** - Automated testing, evaluation, and Docker image deployment

---

## 📁 Project Structure
```
MLOps_Pipeline/
├─ .github/
│  └─ workflows/
│     ├─ test.yml              # Automated testing pipeline
│     ├─ evaluate.yml          # Model evaluation workflow
│     └─ build.yml             # Docker build and push workflow
├─ src/
│  ├─ data_extraction.py       # Dataset loading with error handling
│  ├─ data_processing.py       # Text cleaning and normalization
│  ├─ model.py                 # BERT model training pipeline
│  ├─ inference.py             # Sentiment prediction engine
│  ├─ logger.py                # Database logging utilities
│  ├─ db.py                    # PostgreSQL connection handler
│  ├─ db_schema.sql            # Database schema initialization
│  ├─ data/
│  │  ├─ dataset.csv           # Source data file
│  │  └─ processed/
│  │     ├─ train_tokenized.csv    # Training data
│  │     └─ eval_tokenized.csv     # Evaluation data
│  └─ model/                   # Saved BERT model files
├─ tests/
│  └─ unit/
│     ├─ test_data_extraction.py
│     ├─ test_data_processing.py
│     ├─ test_model.py
│     └─ test_inference.py
├─ cli.py                      # Command-line interface
├─ Dockerfile                  # Container image definition
├─ docker-compose.yml          # Multi-container orchestration
├─ requirements.txt            # Python dependencies
├─ .dockerignore               # Docker build exclusions
├─ .gitignore                  # Git exclusions
├─ data_load.log               # Data loading logs
├─ data_cleaning.log           # Data processing logs
└─ model_training.log          # Model training logs
```

---

## 🔧 Component Descriptions

### 📥 `data_extraction.py`
**Purpose:** Robust dataset loading with comprehensive error handling

**Features:**
- Automatic delimiter detection using `csv.Sniffer`
- Fallback encoding handling (UTF-8 → Latin-1)
- Detailed logging for missing or empty files
- Returns clean pandas DataFrames ready for processing

### 🧹 `data_processing.py`
**Purpose:** Text cleaning and normalization for sentiment analysis

**Features:**
- Lowercases all text for consistency
- Removes punctuation, symbols, and extra whitespace
- Converts emojis to descriptive words (`😊` → `smile`)
- Expands contractions (`It's` → `its`, `can't` → `cannot`)
- Adds `clean_content` column with processed text

**Example:**
| content | clean_content |
|----------|----------------|
| It's Great! ❤️ | its great heart |
| I can't believe it works!!! | i cannot believe it works |

### 🤖 `model.py`
**Purpose:** BERT model training and evaluation pipeline

**Features:**
- Loads pretrained transformer models (`bert-base-uncased`)
- Uses Hugging Face Trainer API for streamlined fine-tuning
- Implements evaluation metrics (accuracy, precision, recall, F1)
- GPU acceleration support (CUDA)
- Saves fine-tuned models to `src/model/`

**Training Flow:**
1. Load tokenized datasets from `data/processed/`
2. Initialize model and tokenizer
3. Fine-tune on labeled sentiment data
4. Evaluate performance on validation set
5. Save trained model artifacts

**Model Outputs:**
```
src/model/
├─ config.json
├─ tokenizer_config.json
├─ special_token_map.json
├─ model.safetensors
├─ vocab.txt
└─ tokenizer.json
```

### 🔮 `inference.py`
**Purpose:** Sentiment prediction engine for production use

**Features:**
- Loads trained model and tokenizer automatically
- Preprocesses input text with same normalization as training
- Returns sentiment predictions (Positive, Negative, Neutral)
- Supports both single-text and batch inference
- Confidence scores for each prediction

**Example Usage:**
```python
from src.inference import predict_sentiment

text = "I absolutely loved this movie!"
print(predict_sentiment(text))
# Output: {'label': 'POSITIVE', 'score': 0.98}
```

### 🖥️ `cli.py`
**Purpose:** Command-line interface for sentiment analysis

**Features:**
- Simple CLI for analyzing text sentiment
- Automatically logs predictions to database
- Accepts text input via `--text` argument
- Integrates inference and logging in single command

**Usage:**
```bash
python cli.py --text "This is amazing!"
```

### 📊 `logger.py`
**Purpose:** Database logging utilities for prediction tracking

**Features:**
- Logs all sentiment predictions to PostgreSQL
- Stores input text and predicted sentiment
- Timestamps each prediction automatically
- Error handling for database connection issues
- Enables analytics and monitoring of model usage

### 🗄️ `db.py`
**Purpose:** PostgreSQL database connection handler

**Features:**
- Manages database connections using `psycopg2`
- Environment variable configuration for flexibility
- Default values for local development
- Supports containerized database deployments

**Configuration:**
- `DB_HOST` - Database host (default: `db`)
- `DB_NAME` - Database name (default: `sentiment_logs`)
- `DB_USER` - Database user (default: `mlops`)
- `DB_PASSWORD` - Database password (default: `mlops123`)

### 🐳 `Dockerfile`
**Purpose:** Container image definition for reproducible deployments

**Features:**
- Based on Python 3.12 official image
- Installs all dependencies from `requirements.txt`
- Sets up proper Python path for module imports
- Configures CLI as container entrypoint
- Optimized layer caching for faster builds

### 🐙 `docker-compose.yml`
**Purpose:** Multi-container orchestration for complete stack

**Features:**
- **API Service:** Runs sentiment analysis CLI
- **Database Service:** PostgreSQL 15 for logging
- **Persistent Volumes:** Preserves models, data, and database
- **Networking:** Bridge network for service communication
- **Auto-initialization:** Runs `db_schema.sql` on first start

**Services:**
- `api` - Sentiment analysis application
- `db` - PostgreSQL database server

**Volumes:**
- `sentiment_models` - Trained model files
- `sentiment_dataset` - Source datasets
- `sentiment_processed` - Processed data
- `sentiment_db` - PostgreSQL data directory

---

## 🔄 CI/CD Pipeline

### `.github/workflows/test.yml` - Automated Testing
**Triggers:** Push/PR to `main` or `develop` branches

**Steps:**
1. Checkout code and setup Python 3.12
2. Install dependencies and development tools
3. Lint code with `flake8`
4. Check formatting with `black`
5. Pull model weights using Git LFS
6. Run unit tests with `pytest`
7. Generate coverage reports
8. Upload coverage to Codecov

### `.github/workflows/evaluate.yml` - Model Evaluation
**Triggers:** After successful test pipeline completion

**Steps:**
1. Checkout code and setup Python
2. Install dependencies
3. Evaluate model performance
4. Check accuracy against threshold (80%)
5. Save metrics as artifacts
6. Fail build if accuracy below threshold

### `.github/workflows/build.yml` - Docker Build & Deploy
**Triggers:** After successful model evaluation

**Steps:**
1. Checkout code
2. Login to DockerHub
3. Build Docker image
4. Tag with `latest` and commit SHA
5. Push to DockerHub registry
6. Display deployment information

**Workflow Chain:**
```
Test Pipeline → Model Evaluation → Docker Build → DockerHub
```

---

## 🧪 Testing

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

**Test output:**

![WhatsApp Image 2025-11-06 at 10 25 44_5efd1871](https://github.com/user-attachments/assets/2594b600-a150-486e-be90-186b47c92f3e)
![WhatsApp Image 2025-11-06 at 10 26 07_fb74316f](https://github.com/user-attachments/assets/6c5ebd5d-a563-4870-9e16-a938e16d80ec)

---

## 💻 Usage Examples

### Python API
```python
from src.data_extraction import load_data
from src.data_processing import normalize_reviews
from src.inference import predict_sentiment

# Load and process dataset
df = load_data('src/data/dataset.csv')
cleaned_df = normalize_reviews(df)

# Run prediction
sample = "This product is absolutely amazing!"
result = predict_sentiment(sample)
print(result)
```

### Command Line Interface
```bash
# Analyze sentiment via CLI
python cli.py --text "I love this product!"

# Using Docker
docker-compose up

# Run with custom text
docker-compose run api python cli.py --text "Great experience!"
```

### Docker Deployment
```bash
# Build image
docker build -t sentiment-app .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## 📝 Logging

Three log files are generated for debugging and transparency:
- `data_load.log` → logs all file loading operations  
- `data_cleaning.log` → logs data cleaning and normalization process  
- `model_training.log` → logs model training process  

---

## ✅ Verification Example

Normalization example confirming expected output:

```text
Input:  "It's Great!"
Output: "its great"
```

---

## 🛠️ Setup & Dependencies

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (for containerized deployment)
- PostgreSQL 15 (or use Docker Compose)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd MLOps_Pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run with Docker Compose (Recommended)**
```bash
docker-compose up
```

4. **Or run locally**
```bash
# Ensure PostgreSQL is running
python cli.py --text "Your text here"
```

---

## 📚 Technologies Used

- **Language:** Python 3.12
- **ML Framework:** Hugging Face Transformers, PyTorch
- **Database:** PostgreSQL 15
- **Containerization:** Docker, Docker Compose
- **Testing:** pytest, pytest-cov
- **Code Quality:** flake8, black, isort
- **CI/CD:** GitHub Actions
- **Version Control:** Git, Git LFS

---

**Done by Amisha & Melek & Ariel**
