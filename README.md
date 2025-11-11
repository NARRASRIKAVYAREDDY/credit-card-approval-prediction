# Credit Card Approval Prediction (ML Project)

This repository contains an end-to-end **Machine Learning pipeline** for predicting **credit card application approval**.  

The model is built using **Python** and **scikit-learn**, and can be trained on any credit card approval dataset
with similar structure (e.g., UCI / Kaggle credit card approval datasets).

---

## Features

- End-to-end workflow:
  - Data loading & cleaning
  - Exploratory checks & preprocessing
  - Encoding categorical variables
  - Scaling numeric features
  - Train/validation split
  - Cross-validation
  - Hyperparameter-tuned Gradient Boosting model
  - Metrics & artifacts saved for inspection
- Produces:
  - Trained model in `models/`
  - Evaluation report in `reports/metrics.json`
  - Feature importances in `reports/feature_importance.csv`

---

## Repository Structure

```text
credit-card-approval-ml/
│
├─ data/
│   ├─ raw/                # place your raw dataset here (e.g. credit_card_approval.csv)
│   └─ processed/          # auto-generated cleaned dataset
│
├─ models/
│   └─ ...                 # saved models (.joblib)
│
├─ reports/
│   └─ ...                 # metrics, feature importance, etc.
│
├─ src/
│   ├─ train_model.py      # main training script
│   ├─ predict_single.py   # sample prediction script
│   └─ utils.py            # shared preprocessing utilities
│
├─ requirements.txt
└─ README.md
```

---

## How to Use

1. **Create / download a dataset**

Use a credit card approval dataset (for example from Kaggle or UCI).  
Save it as:

```text
data/raw/credit_card_approval.csv
```

The script expects a binary target column named `approved` with values like:
- `1` / `0` or `Yes` / `No` (it will normalize these).

Other columns can include numeric and categorical fields such as:
age, income, gender, marital_status, employment_status, education, existing_cards, etc.

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model**

```bash
python src/train_model.py
```

This will:
- Load `data/raw/credit_card_approval.csv`
- Clean & preprocess the data
- Train a tuned Gradient Boosting classifier with cross-validation
- Save metrics to `reports/metrics.json`
- Save the model to `models/credit_approval_model.joblib`
- Save processed data to `data/processed/credit_card_approval_clean.csv`

4. **Run a sample prediction**

After training:

```bash
python src/predict_single.py
```

This script demonstrates how to:
- Load the saved model
- Apply the same preprocessing
- Predict approval for a sample applicant profile.

---

## Why this project is strong for your portfolio

- Shows understanding of the **full ML lifecycle**, not just a single notebook.
- Uses **pipelines, ColumnTransformer, cross-validation**, and **feature importance** — tools used in real teams.
- Can be plugged into dashboards / APIs later.
- Easy for recruiters to run and review.

You can customize:
- Feature list
- Hyperparameters
- Visualization / EDA in a separate notebook.
