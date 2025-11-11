import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

TARGET_COL = "approved"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
    y = df[TARGET_COL]

    # Normalize a few common patterns
    if y.dtype == "O":
        y = y.str.strip().str.lower().map(
            {"yes": 1, "y": 1, "approved": 1, "1": 1,
             "no": 0, "n": 0, "rejected": 0, "0": 0}
        )
    df[TARGET_COL] = pd.to_numeric(y, errors="coerce")
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with all NaNs
    df = df.dropna(how="all").copy()

    # Simple cleaning: strip column names
    df.columns = [c.strip() for c in df.columns]

    # Example: drop ID-like columns if present
    for col in ["id", "ID", "customer_id", "application_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Identify numeric vs categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
