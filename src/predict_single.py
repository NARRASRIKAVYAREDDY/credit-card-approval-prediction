import json
import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = "models/credit_approval_model.joblib"

# Example applicant (you can edit this for demos)
sample_applicant = {
    "age": 29,
    "income": 55000,
    "gender": "Female",
    "marital_status": "Single",
    "employment_status": "Employed",
    "education": "Bachelors",
    "existing_cards": 1,
    "loan_amount": 2000,
    "loan_purpose": "Personal",
}

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first with `python src/train_model.py`."
        )

    model = joblib.load(MODEL_PATH)

    df_app = pd.DataFrame([sample_applicant])

    proba = model.predict_proba(df_app)[0, 1]
    pred = model.predict(df_app)[0]

    label = "Approved" if pred == 1 else "Rejected"

    print("Sample applicant:")
    print(df_app.to_string(index=False))
    print(f"\nPrediction: {label}")
    print(f"Approval probability: {proba:.4f}")

if __name__ == "__main__":
    main()
