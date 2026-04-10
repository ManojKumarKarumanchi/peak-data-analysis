# Expense Account Classification - Technical Report

## Executive Summary

This report presents a machine learning solution for automated expense account classification at Peakflo. Using a **Linear Support Vector Classifier (LinearSVC)** with engineered text and categorical features, the model achieves **91.6% accuracy** on held-out test data, surpassing the minimum 85% requirement. The solution processes 4,894 expense records across 103 account categories, demonstrating robust performance despite significant class imbalance (1179:1 ratio). Key features include TF-IDF vectorization of item names/descriptions, vendor one-hot encoding, and normalized amount features.

---

## 1. Data Analysis & Exploration

### 1.1 Dataset Overview

The provided dataset (`accounts-bills.json`) contains **4,894 expense transactions** with the following characteristics:

- **Records:** 4,894 expenses
- **Features:** vendorId, itemName, itemDescription, accountId, accountName, itemTotalAmount
- **Target Variable:** accountName (103 unique categories)
- **Unique Vendors:** 337
- **Amount Range:** -$15,195 to $161,838,000 SGD

### 1.2 Data Quality Assessment

**Missing Values:**
- `itemDescription`: 31 records (0.63%) have missing descriptions
- All other fields: Complete (0% missing)
- **Action:** Missing descriptions filled with empty strings

**Data Integrity:**
- No duplicate transaction IDs
- All required fields present
- Consistent data types across records

### 1.3 Class Distribution Analysis

The target variable exhibits **severe class imbalance**:

| Metric | Value |
|--------|-------|
| Total Classes | 103 |
| Largest Class | 611202 Online Subscription/Tool (1,179 records, 24.1%) |
| Smallest Class | 1 record (34 singleton classes) |
| Imbalance Ratio | 1179:1 |
| Classes with <5 samples | 34 (33% of all classes) |

**Top 5 Account Categories:**
1. `611202 Online Subscription/Tool` - 1,179 records (24.1%)
2. `132098 IC Clearing account` - 706 records (14.4%)
3. `619203 Supplies/Expenses` - 225 records (4.6%)
4. `134001 Prepaid Operating Expense` - 193 records (3.9%)
5. `614123 Employee On Record` - 175 records (3.6%)

### 1.4 Feature Analysis

**Vendor Signal Strength:**
- Baseline vendor-only model: **73.9% accuracy**
- Vendor + exact item name matching: **98.6% accuracy**
- **Insight:** Strong vendor-account relationships exist, but text features are critical for disambiguation

**Amount Distribution:**
- Mean: $442,736 SGD
- Median: $2,036 SGD
- High variance indicates extreme outliers
- Log transformation applied for normalization

**Text Features:**
- Item names follow structured patterns (e.g., "0126 GAM (SG)", "Slack subscription")
- Many include date prefixes (0126, 1225) indicating billing periods
- Descriptions often duplicate or elaborate on item names

---

## 2. Methodology

### 2.1 Algorithm Selection

**Chosen Algorithm: Linear Support Vector Classifier (LinearSVC)**

**Rationale:**
1. **Scalability:** Efficient for high-dimensional sparse features (13,576 features)
2. **Multiclass Performance:** Handles 103 classes with one-vs-rest strategy
3. **Text Classification:** Proven track record with TF-IDF features
4. **Imbalance Handling:** Built-in `class_weight='balanced'` parameter
5. **Interpretability:** Linear coefficients enable feature importance analysis

**Alternative Approaches Considered:**
- Logistic Regression (similar performance, chosen SVC for margin maximization)
- Random Forest (slower, less effective with high-dimensional sparse features)
- Neural Networks (overkill for tabular data, requires more data for 103 classes)

### 2.2 Feature Engineering

The final feature set combines **4 feature groups** totaling **13,576 dimensions**:

#### 2.2.1 Text Features (Word-Level TF-IDF)
- **Method:** TfidfVectorizer with 1-3 word n-grams
- **Parameters:**
  - `max_features=10,000`
  - `ngram_range=(1, 3)`
  - `min_df=2` (appears in at least 2 documents)
  - `sublinear_tf=True` (log-scaled term frequency)
- **Preprocessing:** Lowercase, strip whitespace, combine itemName + itemDescription
- **Rationale:** Captures semantic meaning and multi-word patterns

#### 2.2.2 Character N-Gram Features
- **Method:** TfidfVectorizer with character-level analysis
- **Parameters:**
  - `analyzer='char_wb'` (word boundaries preserved)
  - `ngram_range=(3, 5)`
  - `max_features=5,000`
  - `min_df=2`
- **Rationale:** Handles typos, abbreviations, and partial word matches

#### 2.2.3 Vendor Encoding
- **Method:** One-Hot Encoding (337 vendors)
- **Parameters:** `handle_unknown='ignore'` for production robustness
- **Rationale:** Vendor identity is a strong predictor (73.9% baseline)

#### 2.2.4 Amount Features
- **Features:**
  1. Raw `itemTotalAmount`
  2. Log-transformed `log1p(abs(itemTotalAmount))`
- **Normalization:** StandardScaler applied
- **Rationale:** Accounts may have characteristic expense ranges

**Feature Combination:**
All feature groups horizontally stacked using `scipy.sparse.hstack` to preserve memory efficiency.

### 2.3 Handling Class Imbalance

**Strategy 1: Class Weighting**
- Set `class_weight='balanced'` in LinearSVC
- Automatically weights classes inversely proportional to frequency
- Formula: `w_i = n_samples / (n_classes × n_samples_i)`

**Strategy 2: Stratified Sampling**
- Train/test split preserves class distribution (80/20 split)
- Only applied to classes with 2+ samples
- Singleton classes (34 total) added to training set

**Impact:** Improved minority class recall while maintaining overall accuracy

### 2.4 Model Training

**Hyperparameters:**
```python
LinearSVC(
    C=5,                      # Regularization strength (inverse)
    class_weight='balanced',  # Handle imbalance
    max_iter=1000,           # Iterations for convergence
    random_state=42          # Reproducibility
)
```

**Training Data:**
- 3,918 samples (after stratified split + singleton addition)
- 13,576 features
- 103 classes

**Convergence Note:** Model exhibited a convergence warning; performance remained stable, but `max_iter` could be increased in production.

### 2.5 Validation Strategy

**Approach:** Stratified Train/Test Split (80/20)
- **Training Set:** 3,918 records (80%)
- **Test Set:** 976 records (20%)
- **Stratification:** Ensures proportional class representation in both sets
- **Random Seed:** 42 (reproducibility)

**Note:** Cross-validation not applied due to singleton classes, but stratified split provides reliable holdout validation.

---

## 3. Results

### 3.1 Overall Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | **91.60%** | 894/976 correct predictions |
| **F1-Score (Weighted)** | 91.75% | Accounts for class imbalance |
| **F1-Score (Macro)** | 77.35% | Average performance across all classes |
| **Error Rate** | 8.40% | 82/976 misclassifications |

**Result:** **Exceeds 85% minimum requirement** and approaches the 92% internal benchmark.

### 3.2 Performance by Category

**Top Performing Categories (100% Accuracy):**
- Many low-frequency classes with distinctive patterns
- Example: Specific vendor-item combinations

**Challenging Categories:**
Analysis of errors reveals common misclassifications between:
1. **IC Clearing account** ↔ **Online Subscription/Tool** (7+ errors)
2. **Prepaid Operating Expense** ↔ **Online Subscription/Tool** (4+ errors)
3. **Customer Prepayment** variants (multiple account codes)

**Error Patterns:**
- Higher-value transactions more likely to be misclassified
  - Avg error transaction: $755,885 SGD
  - Avg correct transaction: $203,531 SGD
- Ambiguous text descriptions (e.g., "0225 Enteprise - Data")
- Unusual prepayments/refunds with non-standard patterns

### 3.3 Sample Misclassifications

| Item Name | True Label | Predicted Label | Amount |
|-----------|-----------|-----------------|--------|
| CMG Tote Bags | 612016 Collateral | 612023 Print Ads | $642 |
| 0424 Enteprise - Data | Online Subscription/Tool | IC Clearing account | $1,472 |
| Annual Bill for Docker-2025 | Prepaid Operating Expense | Online Subscription/Tool | $8,514 |

**Analysis:** Errors often occur with:
- Ambiguous item names lacking clear category signals
- Cross-category items (e.g., subscriptions paid via clearing accounts)
- Rare transaction types

---

## 4. Discussion

### 4.1 Strengths of the Approach

1. **Strong Feature Engineering:**
   - Multi-level text analysis (word + character n-grams)
   - Combines semantic and syntactic patterns
   - Vendor identity leveraged effectively

2. **Robust to Imbalance:**
   - Class weighting prevents majority-class bias
   - Stratified split ensures fair evaluation

3. **Scalable & Efficient:**
   - Sparse matrix operations handle 13K+ features
   - LinearSVC trains in seconds on CPU

4. **Production-Ready:**
   - Saved model artifacts enable deployment
   - FastAPI + Streamlit interfaces provided
   - Handles unseen vendors gracefully

### 4.2 Limitations

1. **Convergence Issues:**
   - LibLinear failed to fully converge within 1,000 iterations
   - **Mitigation:** Increase `max_iter` or use `lbfgs` solver

2. **Singleton Class Challenge:**
   - 34 classes have only 1 training example
   - Model may overfit to these specific instances
   - **Impact:** Macro F1 (77.35%) lower than weighted F1 (91.75%)

3. **Amount Feature Underutilization:**
   - Only 2 amount features vs. 13,574 text/vendor features
   - Could explore amount-category interactions

4. **No Temporal Validation:**
   - Random split doesn't test time-series generalization
   - Production model should validate on recent months

5. **Error Bias Toward High-Value Transactions:**
   - Complex, large-amount expenses harder to classify
   - May need specialized handling for outliers

### 4.3 Ideas for Improvement

**Short-Term (1-2 Days):**
1. **Hyperparameter Tuning:** Grid search over `C`, `max_iter`, loss functions
2. **Feature Selection:** Remove low-importance features to reduce noise
3. **Ensemble Methods:** Combine LinearSVC with Logistic Regression (voting)
4. **Error Analysis:** Manual review of misclassifications to refine features

**Medium-Term (1-2 Weeks):**
1. **Advanced NLP:** 
   - Use pre-trained embeddings (e.g., sentence-transformers)
   - Fine-tune BERT-style models for domain-specific terms
2. **Hierarchical Classification:**
   - First predict account category (6xx, 1xx, etc.)
   - Then predict specific sub-account
3. **Active Learning:**
   - Flag low-confidence predictions for human review
   - Retrain with corrected labels

**Long-Term (Production Deployment):**
1. **Temporal Validation:** Train on months 1-9, test on month 10-12
2. **A/B Testing:** Compare model suggestions to human categorizations
3. **Feedback Loop:** Continuously retrain with corrected predictions
4. **Explainability:** Add SHAP values to show why each classification was made
5. **Monitoring:** Track accuracy drift over time as new vendors/categories emerge

### 4.4 Business Considerations for Deployment

**Confidence Thresholds:**
- Flag predictions with low confidence (<80%) for manual review
- Reduces error rate while automating high-confidence cases

**Gradual Rollout:**
- Phase 1: Assist mode (suggest categories to finance team)
- Phase 2: Auto-categorize high-confidence predictions
- Phase 3: Full automation with exception handling

**Maintenance:**
- Retrain quarterly with new data
- Monitor for new vendors/account categories
- Track model drift and retrain triggers

**Integration:**
- API endpoint for real-time predictions (FastAPI implemented)
- Batch processing for monthly expense imports
- Logging of predictions for audit trail

**Cost-Benefit:**
- Current: Manual review of ~5,000 expenses/month
- Automated: ~91.6% correct → 410 errors to review
- **Time Savings:** ~85% reduction in manual categorization effort

---

## 5. Deliverables Summary

### Code Structure
```
├── accounts-bills.json          # Training dataset
├── eda.ipynb                    # Exploratory data analysis
├── model_train_eval.ipynb       # Model training & evaluation
├── models/                      # Saved model artifacts
│   ├── expense_classifier.pkl
│   ├── tfidf_word.pkl
│   ├── tfidf_char.pkl
│   ├── ohe_vendor.pkl
│   ├── amount_scaler.pkl
│   └── label_encoder.pkl
├── api.py                       # FastAPI prediction service
└── streamlit_app.py             # Interactive demo UI
```

### Reproducibility
- All random seeds set to 42
- Dependencies listed in `requirements.txt`
- Step-by-step execution in Jupyter notebooks

---

## 6. Conclusions

The developed expense classification system successfully meets and exceeds the 85% accuracy requirement, achieving **91.6% accuracy** on held-out test data. The solution demonstrates:

1. **Strong Performance:** Surpasses minimum threshold, approaches 92% benchmark
2. **Robust Methodology:** Handles severe class imbalance and sparse text features
3. **Production Readiness:** Deployed as FastAPI service with Streamlit interface
4. **Clear Documentation:** Code, analysis, and business insights provided

**Next Steps:**
- Address convergence warnings with solver tuning
- Implement confidence-based routing for manual review
- Deploy in "assist mode" for finance team validation
- Establish retraining cadence and monitoring dashboard

---

## Appendices

### A. Key Technologies Used
- **Python 3.x**
- **scikit-learn:** Model training, feature engineering, evaluation
- **pandas/numpy:** Data manipulation
- **FastAPI:** REST API deployment
- **Streamlit:** Interactive UI
- **joblib:** Model serialization

### B. Model Files
All trained models saved to `models/`:
- `expense_classifier.pkl` (11.2 MB)
- `tfidf_word.pkl` (351 KB)
- `tfidf_char.pkl` (171 KB)
- `ohe_vendor.pkl` (8.4 KB)
- `amount_scaler.pkl` (631 bytes)
- `label_encoder.pkl` (3.2 KB)

### C. Performance Metrics Explained
- **Accuracy:** (TP + TN) / Total - Simple correctness rate
- **F1 Weighted:** Harmonic mean of precision/recall, weighted by class frequency
- **F1 Macro:** Unweighted average F1 across all classes (emphasizes minority classes)
