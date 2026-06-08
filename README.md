# Call Result Prediction Model

Predictive model for forecasting the likelihood of successful outbound collection calls in the Soft Collection domain.


## Table of Contents

- [Business Problem](#business-problem)
- [Solution Overview](#solution-overview)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Insights](#results--insights)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Business Problem

In robotic outbound calling for debt collection, **not all calls are equally likely to succeed**. Some clients are more receptive and likely to commit to payment (result), while others consistently decline or don't answer.

**Challenge:** How can we predict which calls are most likely to be successful **before** making them?

**Impact:** By prioritizing high-probability calls, we can:
- ✅ Increase overall collection rate
- ✅ Optimize dialer resource allocation
- ✅ Reduce wasted effort on low-probability calls
- ✅ Improve ROI of collection campaigns

---

## Solution Overview

This project builds a **binary classification model** to predict the probability that an outbound call will result in a positive outcome (payment promise or commitment).

### Key Features:
- **End-to-end ML pipeline**: From raw data to production-ready model
- **Multiple algorithms tested**: Logistic Regression, Random Forest, LightGBM
- **Class imbalance handling**: Weighted classes to handle 9% positive class rate
- **Temporal validation**: Train/test split by time to avoid data leakage
- **Feature engineering**: Historical client behavior, temporal patterns, campaign characteristics

### Target Variable:
- **`is_result = 1`**: Call resulted in payment promise/commitment
- **`is_result = 0`**: Call did not result in commitment

---

## Dataset

**Source:** Synthetic data generated to mirror real-world collection call patterns (see [synthetic-call-data-generator](../synthetic-call-data-generator))

### Dataset Statistics:
- **Total Calls**: 79,876 unique calls
- **Date Range**: October 2024 - March 2025 (5+ months)
- **Clients**: 10,000 unique clients
- **Positive Class Rate**: 8.92% (result calls)
- **Features**: 14 engineered features

### Feature Categories:

**1. Client Demographics:**
- Region (Almaty, Astana, Shymkent, etc.)
- Age group (18-25, 26-35, 36-50, 50+)
- Debt amount bucket (<10k, 10k-50k, etc.)
- Product type (Credit Card, Cash Loan, Installment)
- Risk score (0-1)

**2. Call Characteristics:**
- Hour of day (0-23)
- Day of week / Is weekend
- Campaign segment (1-2 RB, 3-6 НБ, etc.)
- Campaign tree/script variant

**3. Historical Features:**
- `client_call_number`: Call attempt number for this client
- `client_prev_result_rate`: Historical success rate for this client
- `client_prev_contact_rate`: Historical contact rate
- `days_since_last_call`: Recency of last interaction

---

## Model Performance

### Best Model: **Logistic Regression with Balanced Classes**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **0.601** | Better than random (0.50), reasonable for highly imbalanced data |
| **PR-AUC** | 0.126 | Captures 26% more positive cases than random baseline |
| **Precision** | 0.14 | Among predicted results, 14% are true positives |
| **Recall** | 0.22 | Finds 22% of all actual result calls |
| **F1-Score** | 0.17 | Harmonic mean of precision and recall |

### Model Comparison:

| Model | ROC-AUC | Recall | True Positives Found |
|-------|---------|--------|---------------------|
| **Logistic Regression** | **0.601** | **22%** | 266 / 1,196 |
| Random Forest | 0.587 | 11% | 131 / 1,196 |
| LightGBM | 0.576 | 27% | 323 / 1,196 |

**Why Logistic Regression wins:** Best balance between ROC-AUC and interpretability. While LightGBM has higher recall, Logistic Regression provides better overall discrimination and is more interpretable for business stakeholders.

---

## Project Structure
```
call-result-prediction/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   └── model_config.yaml        # Model hyperparameters and features
├── data/
│   ├── raw/                     # Raw data from generator
│   └── processed/               # Processed train/test sets
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb        # Model training & comparison
│   └── 04_evaluation.ipynb      # Detailed evaluation
├── src/
│   ├── data_preparation.py      # Data loading & preprocessing
│   ├── feature_engineering.py   # Feature creation & encoding
│   ├── train_model.py           # Model training functions
│   └── evaluate_model.py        # Evaluation & visualization
├── models/
│   ├── logistic_regression_model.pkl  # Trained model
│   ├── encoders.pkl             # Categorical encoders
│   └── model_config.pkl         # Config snapshot
├── results/
│   ├── roc_curve.png            # ROC curve visualization
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   ├── logistic_regression_metrics.json
│   └── model_comparison.csv
├── train.py                     # Main training pipeline
└── copy_data.py                 # Utility to copy data from generator
```

---

## Installation

### Prerequisites:
- Python 3.9+
- pip

### Setup:
```bash
# Clone the repository
git clone https://github.com/your-username/call-result-prediction.git
cd call-result-prediction

# Install dependencies
pip install -r requirements.txt

# Copy data from generator (if you have it)
python copy_data.py
```

---

## Usage

### Option 1: Run Full Training Pipeline
```bash
python train.py
```

**Output:**
- Trained models saved to `models/`
- Evaluation metrics saved to `results/`
- Visualizations (ROC curve, feature importance, etc.)

### Option 2: Interactive Exploration (Jupyter Notebooks)
```bash
jupyter notebook
```

Then open:
1. `notebooks/01_eda.ipynb` - Explore the data
2. `notebooks/02_feature_engineering.ipynb` - Feature preparation
3. `notebooks/03_modeling.ipynb` - Train models interactively
4. `notebooks/04_evaluation.ipynb` - Deep-dive into model performance

### Option 3: Use Trained Model for Predictions
```python
import pickle
import pandas as pd

# Load model
with open('models/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoders
with open('models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Prepare new data (apply same feature engineering)
# ... your feature engineering code ...

# Predict
probabilities = model.predict_proba(X_new)[:, 1]
predictions = model.predict(X_new)
```

---

## Results & Insights

### Key Findings from EDA:

1. **Temporal Patterns:**
   - **Best time to call:** 17:00-21:00 (evening hours)
   - **Best day:** Mid-week (Tuesday-Thursday)
   - **Worst time:** Early morning (8:00-10:00)

2. **Client Behavior:**
   - **First call success rate:** 12% vs Later calls: 6%
   - Clients with `client_prev_result_rate > 0.2` have 3x higher success probability
   - `days_since_last_call < 3` → lower success (client fatigue)

3. **Campaign Performance:**
   - Segment "1-2 RB" (recent balance): Highest success rate (11%)
   - Segment "3-6 НБ" (non-performing): Lowest success rate (7%)

### Feature Importance (Top 5):

1. **`client_prev_result_rate`** (Historical success) - Most predictive
2. **`client_call_number`** (Call attempt) - Decays with attempts
3. **`hour`** (Time of day) - Evening peak
4. **`segment`** (Campaign type) - Different strategies work for different segments
5. **`risk_score`** (Client risk) - Higher risk → lower success

### Business Impact:

**Scenario:** Using model to prioritize top 20% of calls by predicted probability

- **Results:** Captures **35-40%** of all successful outcomes
- **Efficiency:** **1.8x lift** over random calling
- **ROI:** Reduces wasted efforts on low-probability calls by 80%

---

## Technologies Used

- **Python 3.9+**
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, LightGBM
- **Visualization:** matplotlib, seaborn
- **Model Interpretation:** SHAP
- **Config Management:** PyYAML
- **Notebooks:** Jupyter

---

## Future Improvements

### Model Enhancements:
- [ ] **Increase positive class rate in data** (15-20%) for better training
- [ ] **SMOTE (Synthetic Minority Over-sampling)** for class balancing
- [ ] **Hyperparameter tuning** (GridSearchCV, Bayesian optimization)
- [ ] **Ensemble models** (stacking Logistic Regression + LightGBM)
- [ ] **Deep learning** (TabNet, neural networks) for complex patterns

### Feature Engineering:
- [ ] **Recent contact patterns** (rolling 3-call success rate)
- [ ] **Time since first call** (client lifecycle stage)
- [ ] **Campaign-level aggregates** (segment average performance)
- [ ] **Interaction features** (hour × segment, call_number × prev_result_rate)

### Deployment:
- [ ] **API endpoint** (FastAPI) for real-time predictions
- [ ] **Streamlit dashboard** for business users
- [ ] **Model monitoring** (drift detection, performance tracking)
- [ ] **A/B testing framework** for production validation

---

## Author

**Serik Karybayev**  
Data Analyst | Aspiring Data Scientist

- GitHub: [@SerikKarybaev](https://github.com/SerikKarybaev/)
- LinkedIn: [Serik Karybaev](www.linkedin.com/in/serik-karybaev-29a544116)


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Synthetic data generated using [synthetic-call-data-generator](../synthetic-call-data-generator)
- Inspired by real-world challenges in debt collection analytics
- Built as part of a Data Science portfolio project

---