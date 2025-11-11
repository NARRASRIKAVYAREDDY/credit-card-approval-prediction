import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import (
    load_data,
    normalize_target,
    clean_features,
    split_features_target,
    build_preprocessor,
    train_val_split,
)

DATA_PATH = "data/raw/credit_card_approval.csv"
PROCESSED_PATH = "data/processed/credit_card_approval_clean.csv"
MODEL_PATH = "models/credit_approval_model.joblib"
METRICS_PATH = "reports/metrics.json"
FEAT_IMP_PATH = "reports/feature_importance.csv"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Please place your dataset there."
        )

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load & clean
    df = load_data(DATA_PATH)
    df = normalize_target(df)
    df = clean_features(df)

    # Save a cleaned snapshot
    df.to_csv(PROCESSED_PATH, index=False)

    # Split
    X, y = split_features_target(df)

    preprocessor = build_preprocessor(X)

    # Base model (strong baseline)
    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )

    from sklearn.pipeline import Pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # Cross-validation for robustness
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"CV Accuracy: mean={cv_accuracy.mean():.4f}, std={cv_accuracy.std():.4f}")
    print(f"CV ROC-AUC:  mean={cv_auc.mean():.4f}, std={cv_auc.std():.4f}")

    # Final train/val split for report-style metrics
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    print(f"\nHold-out Accuracy: {acc:.4f}")
    print(f"Hold-out ROC-AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Saved trained model to {MODEL_PATH}")

    # Extract feature importances from underlying model
    # (after preprocessing; approximate mapping)
    classifier = model.named_steps["classifier"]
    pre = model.named_steps["preprocessor"]

    # Build feature name list
    num_features = pre.transformers_[0][2]
    cat_features = pre.transformers_[1][2]
    ohe = pre.transformers_[1][1].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_features) if len(cat_features) > 0 else []
    all_feature_names = list(num_features) + list(cat_feature_names)

    importances = getattr(classifier, "feature_importances_", None)

    if importances is not None and len(importances) == len(all_feature_names):
        fi_df = pd.DataFrame({
            "feature": all_feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(FEAT_IMP_PATH, index=False)
        print(f"✅ Saved feature importances to {FEAT_IMP_PATH}")
    else:
        print("⚠️ Could not reliably map feature importances (skipping).")

    # Save metrics
    metrics = {
        "cv_accuracy_mean": float(cv_accuracy.mean()),
        "cv_accuracy_std": float(cv_accuracy.std()),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "holdout_accuracy": float(acc),
        "holdout_roc_auc": float(auc),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Saved metrics to {METRICS_PATH}")

if __name__ == "__main__":
    main()
