"""
Simple model pipeline for the hospital readmission case study.
This script generates synthetic data, trains a logistic regression and an XGBoost classifier (if available),
computes confusion matrix, precision and recall, and saves a sample model output.

Run: python code\model_pipeline.py

Requirements: see requirements.txt
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Seed for reproducibility
RANDOM_STATE = 42

# Create synthetic dataset
def generate_synthetic_readmission_data(n=5000, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    # Demographics
    age = rng.normal(65, 16, n).clip(18, 95).astype(int)
    sex = rng.binomial(1, 0.52, n)  # 1 male, 0 female
    # Utilization
    prior_adm_6m = rng.poisson(0.6, n)
    ed_visits_6m = rng.poisson(0.8, n)
    # Clinical
    charlson = rng.poisson(2.0, n)
    labs_flag = rng.binomial(1, 0.15, n)  # missingness indicator
    med_count = rng.poisson(5, n)
    length_of_stay = rng.exponential(3.0, n).clip(1, 30)

    # Construct a risk score (latent)
    risk_score = (
        0.02 * (age - 65)
        + 0.6 * (prior_adm_6m > 0).astype(float)
        + 0.4 * (ed_visits_6m > 1).astype(float)
        + 0.3 * (charlson / (charlson.max() + 1))
        + 0.05 * med_count
        + 0.1 * (length_of_stay > 5).astype(float)
        + rng.normal(0, 0.6, n)
    )

    # Convert to binary label using logistic
    prob = 1 / (1 + np.exp(-risk_score))
    readmit = (rng.uniform(0, 1, n) < prob * 0.4).astype(int)  # scale to reasonable prevalence

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "prior_adm_6m": prior_adm_6m,
        "ed_visits_6m": ed_visits_6m,
        "charlson": charlson,
        "labs_flag": labs_flag,
        "med_count": med_count,
        "length_of_stay": length_of_stay,
        "readmit_30d": readmit,
    })

    return df


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating synthetic data...")
    df = generate_synthetic_readmission_data(3000)
    print("Data shape:", df.shape)

    X = df.drop(columns=["readmit_30d"]) 
    y = df["readmit_30d"]

    # Train/val/test split (70/15/15)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.17647, random_state=RANDOM_STATE, stratify=y_trainval)
    # 0.17647 * 0.85 ~= 0.15 overall

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train_s, y_train)

    # Validation
    val_pred = clf.predict(X_val_s)
    val_prob = clf.predict_proba(X_val_s)[:, 1]
    print("Validation classification report:\n", classification_report(y_val, val_pred))

    # Test
    test_pred = clf.predict(X_test_s)
    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)

    print("Test confusion matrix:\n", cm)
    print(f"Test precision: {precision:.3f}")
    print(f"Test recall: {recall:.3f}")

    # Save model and scaler
    joblib.dump(clf, os.path.join(out_dir, "logistic_model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    df.to_csv(os.path.join(out_dir, "synthetic_data_sample.csv"), index=False)

    print(f"Saved artifacts to {out_dir}")

if __name__ == '__main__':
    main()
