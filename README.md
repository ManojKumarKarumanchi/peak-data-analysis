# Expense Account Classification System

**Automated expense categorization for Peakflo using machine learning**

## Overview

This project implements an automated classification system that predicts the correct accounting account for expense transactions. Using a Linear Support Vector Machine with engineered text and categorical features, the model achieves **91.6% accuracy** across 103 account categories.

**Key Features:**
- 91.6% classification accuracy (exceeds 85% requirement)
- Handles 103 account categories with severe class imbalance
- FastAPI REST API for production deployment
- Streamlit web interface for interactive testing
- Comprehensive exploratory data analysis

---

## Project Structure

```
peak-data-analysis/                       # Main implementation directory
│   ├── accounts-bills.json          # Training dataset (4,894 records)
│   ├── eda.ipynb                    # Exploratory Data Analysis
│   ├── model_train_eval.ipynb       # Model training & evaluation
│   ├── models/                      # Trained model artifacts
│   │   ├── expense_classifier.pkl   # LinearSVC model
│   │   ├── tfidf_word.pkl          # Word n-gram vectorizer
│   │   ├── tfidf_char.pkl          # Character n-gram vectorizer
│   │   ├── ohe_vendor.pkl          # Vendor one-hot encoder
│   │   ├── amount_scaler.pkl       # Amount feature scaler
│   │   └── label_encoder.pkl       # Target label encoder
│   ├── api.py                       # FastAPI prediction service
│   └── streamlit_app.py             # Interactive demo interface
├── REPORT.md                        # Detailed technical report
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Quick Start

### Prerequisites

- Python 3.12 or higher
- pip package manager
- (Optional) Virtual environment tool

### Installation

1. **Clone the repository:**
   ```bash
   cd C:\Users\User\local\peak-data-analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Option 1: Train the Model from Scratch

Run the Jupyter notebook to train the model:

```bash
jupyter notebook model_train_eval.ipynb
```

**Steps in the notebook:**
1. Load and preprocess data
2. Engineer text, vendor, and amount features
3. Train LinearSVC with class balancing
4. Evaluate on test set (91.6% accuracy)
5. Save model artifacts to `models/` directory

**Expected Output:**
```
Accuracy: 0.9159836065573771
F1 weighted: 0.9174519917268195
F1 macro: 0.7734633581820562
All models saved to: models
```

### Option 2: Explore the Data

Run the EDA notebook to understand the dataset:

```bash
jupyter notebook eda.ipynb
```

**Analysis includes:**
- Missing value analysis
- Class distribution visualization
- Vendor-account relationship strength
- Amount distribution patterns
- Baseline model performance

### Option 3: Run the API Service

```
MODEL_DIR = "models"
```

Then start the FastAPI server:

```bash

uvicorn api:app --reload
```

**API Endpoints:**
- **POST** `/predict` - Classify an expense

**Example cURL request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "vendorId": "lL1pcuEf3q6ufBVg2R75",
    "itemName": "Slack subscription",
    "itemDescription": "Slack monthly subscription",
    "itemTotalAmount": 1000.0
  }'
```

**Response:**
```json
{
  "accountName": "611202 Online Subscription/Tool"
}
```

**API Documentation:**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Option 4: Run the Streamlit Demo

**Important:** Ensure the API is running first (see Option 3), then:

```bash
streamlit run streamlit_app.py
```

**Demo Interface:**
- Input: Vendor ID, Item Name, Description, Amount
- Output: Predicted account name
- Default example pre-populated for testing

**Access at:** http://localhost:8501

---

## Model Details

### Architecture

**Algorithm:** Linear Support Vector Classifier (LinearSVC)

**Feature Engineering:**
1. **Word-level TF-IDF** (10,000 features)
   - 1-3 word n-grams from item name + description
   - Captures semantic patterns
   
2. **Character-level TF-IDF** (5,000 features)
   - 3-5 character n-grams
   - Handles typos and abbreviations
   
3. **Vendor One-Hot Encoding** (337 features)
   - Strong predictor (73.9% baseline accuracy)
   
4. **Amount Features** (2 features)
   - Raw amount + log-transformed amount
   - StandardScaler normalized

**Total Features:** 13,576 dimensions (sparse matrix)

### Performance Metrics

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | - | **91.60%** |
| F1 Weighted | - | 91.75% |
| F1 Macro | - | 77.35% |
| Error Rate | - | 8.40% |

### Handling Class Imbalance

- **Imbalance Ratio:** 1179:1 (largest to smallest class)
- **Strategy:** `class_weight='balanced'` in LinearSVC
- **Sampling:** Stratified train/test split (80/20)
- **Singleton Classes:** 34 classes with only 1 sample (added to training set)

---

## Dataset Information

**Source:** `accounts-bills.json`

**Statistics:**
- **Records:** 4,894 expense transactions
- **Features:** vendorId, itemName, itemDescription, accountId, accountName, itemTotalAmount
- **Target:** accountName (103 unique categories)
- **Vendors:** 337 unique vendor IDs
- **Amount Range:** -$15,195 to $161,838,000 SGD

**Top 5 Categories:**
1. `611202 Online Subscription/Tool` - 1,179 records (24.1%)
2. `132098 IC Clearing account` - 706 records (14.4%)
3. `619203 Supplies/Expenses` - 225 records (4.6%)
4. `134001 Prepaid Operating Expense` - 193 records (3.9%)
5. `614123 Employee On Record` - 175 records (3.6%)

---

## API Reference

### Request Schema

```python
{
  "vendorId": str,          # Vendor identifier
  "itemName": str,          # Expense item name
  "itemDescription": str,   # Expense description
  "itemTotalAmount": float  # Transaction amount in SGD
}
```

### Response Schema

```python
{
  "accountName": str  # Predicted account category
}
```

### Feature Processing Pipeline

```python
def build_features(data):
    # 1. Text preprocessing
    text = combine_item_name_and_description(data)
    
    # 2. TF-IDF vectorization
    X_word = tfidf_word.transform([text])
    X_char = tfidf_char.transform([text])
    
    # 3. Vendor encoding
    X_vendor = ohe.transform([[data["vendorId"]]])
    
    # 4. Amount features
    amt = [data["itemTotalAmount"], log1p(abs(data["itemTotalAmount"]))]
    X_amt = scaler.transform([amt])
    
    # 5. Concatenate all features
    return hstack([X_word, X_char, X_vendor, X_amt])
```

---

## Requirements

Install all dependencies:
```bash
uv pip install -r requirements.txt
```

---

## Troubleshooting

### Issue: Streamlit shows connection error

**Solution:** Ensure FastAPI server is running:
```bash
# Terminal 1

uvicorn api:app --reload

# Terminal 2

streamlit run streamlit_app.py
```

### Issue: Convergence warning during training

**Solution:** Increase iterations in `model_train_eval.ipynb`:
```python
model = LinearSVC(
    C=5,
    class_weight='balanced',
    max_iter=2000,  # Increase from 1000
    random_state=42
)
```

---

## Development

### Running Tests

```bash
# Test API endpoint
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"vendorId":"test","itemName":"test","itemDescription":"test","itemTotalAmount":100}'
```

### Retraining the Model

1. Add new data to `accounts-bills.json`
2. Run `model_train_eval.ipynb`
3. Models automatically saved to `models/` directory
4. Restart API server to load new models

---

## Performance Benchmarks

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Vendor-only baseline | 73.9% | Simple lookup table |
| Vendor + exact name match | 98.6% | Requires exact duplicates |
| **LinearSVC (this model)** | **91.6%** | **Generalizes to unseen data** |
| Internal benchmark | 92% | Target performance |

---

## Future Improvements

**Short-Term:**
- Hyperparameter tuning (grid search over C, max_iter)
- Ensemble methods (voting classifier)
- Feature selection (remove low-importance features)

**Medium-Term:**
- Pre-trained embeddings (sentence-transformers)
- Hierarchical classification (account groups → sub-accounts)
- Confidence thresholds for manual review

**Long-Term:**
- Temporal validation (train on past, test on future)
- Active learning pipeline
- SHAP explainability
- Model monitoring dashboard
