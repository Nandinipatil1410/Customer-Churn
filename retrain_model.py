"""
Retrain the KNN churn prediction model with proper class imbalance handling.

Root cause: The dataset is 88.3% "Churn=Yes" (883/1000), so models trained
on raw data predict churn for nearly all inputs.

Fix: Use SMOTE to balance the training set, distance-weighted KNN,
and GridSearchCV with F1 scoring.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# ── 1. Load & encode ────────────────────────────────────────────────
df = pd.read_csv("customer_churn_data.csv")

X = df[["Age", "Gender", "Tenure", "MonthlyCharges"]].copy()
X["Gender"] = X["Gender"].apply(lambda x: 1 if x == "Female" else 0)

y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

print("=== Original class distribution ===")
print(y.value_counts().rename({1: "Churn", 0: "No Churn"}))
print()

# ── 2. Train/test split (reproducible) ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")
print()

# ── 3. SMOTE oversampling on training set ───────────────────────────
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("=== After SMOTE ===")
print(pd.Series(y_train_balanced).value_counts().rename({1: "Churn", 0: "No Churn"}))
print()

# ── 4. Scale features ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# ── 5. GridSearch KNN with distance weighting ──────────────────────
knn = KNeighborsClassifier(weights="distance")
param_grid = {"n_neighbors": [3, 5, 7, 9, 11, 15, 21]}

grid = GridSearchCV(knn, param_grid, cv=5, scoring="f1")
grid.fit(X_train_scaled, y_train_balanced)

print(f"Best params: {grid.best_params_}")
print(f"Best CV F1:  {grid.best_score_:.3f}")
print()

# ── 6. Evaluate on test set ────────────────────────────────────────
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("=== Test Set Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print()
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── 7. Sanity-check predictions ────────────────────────────────────
test_cases = [
    ([35, 0, 60, 40],  "Long-term low-cost male"),
    ([25, 1, 50, 35],  "Long-term low-cost female"),
    ([50, 0,  5, 100], "Short-term high-cost male"),
    ([45, 1,  2, 120], "Short-term high-cost female"),
    ([60, 0, 70, 30],  "Senior long-term male"),
]

print("=== Sample Predictions ===")
for features, label in test_cases:
    scaled = scaler.transform([features])
    pred = best_model.predict(scaled)[0]
    result = "Churn" if pred == 1 else "No Churn"
    print(f"  {label:35s} => {result}")
print()

# ── 8. Save model & scaler ─────────────────────────────────────────
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved model.pkl and scaler.pkl")
