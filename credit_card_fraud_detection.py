"""
CREDIT CARD FRAUD DETECTION
============================
Level: Basic to Intermediate
Goal : Classify credit card transactions as Fraudulent (1) or Genuine (0).

NOTES ON DATA
-------------
The real-world benchmark dataset for this project is the Kaggle
"Credit Card Fraud Detection" dataset (creditcard.csv), which contains
284,807 transactions made by European cardholders, with 28 anonymized
PCA features (V1-V28), plus 'Time', 'Amount', and the target 'Class'.

To keep this script self-contained and runnable without downloading
anything, we GENERATE a synthetic dataset that mimics the same
structure (highly imbalanced, ~0.5% fraud). If you have the real
creditcard.csv file, just replace the "Load / Generate Data" section
with:
    df = pd.read_csv("creditcard.csv")
and the rest of the pipeline will work unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.datasets import make_classification

# -----------------------------------------------------------------
# 1. LOAD / GENERATE DATA
# -----------------------------------------------------------------
# make_classification lets us simulate a realistic fraud-detection
# scenario: many features, and a heavy class imbalance (fraud is rare).
X, y = make_classification(
    n_samples=20000,       # total transactions
    n_features=29,         # similar to V1-V28 + Amount
    n_informative=12,      # features that actually carry signal
    n_redundant=5,
    weights=[0.995, 0.005],  # ~0.5% fraud, mimicking real-world imbalance
    flip_y=0.001,           # small amount of label noise
    random_state=42,
)

feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
df = pd.DataFrame(X, columns=feature_names)
df["Class"] = y

print("Dataset shape:", df.shape)
print("Fraud cases:", df["Class"].sum(), "out of", len(df))
print("Fraud percentage: {:.3f}%".format(100 * df["Class"].mean()))

# -----------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# -----------------------------------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# stratify=y keeps the same fraud/genuine ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------------------------------------------
# 3. FEATURE SCALING
# -----------------------------------------------------------------
# Logistic Regression is sensitive to feature scale, so we standardize
# (mean=0, std=1). Tree-based models don't strictly need this, but it
# doesn't hurt and keeps the pipeline consistent.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# 4. MODEL 1 - LOGISTIC REGRESSION (simple, interpretable baseline)
# -----------------------------------------------------------------
# class_weight='balanced' automatically upweights the rare fraud class,
# which is important since fraud is <1% of the data.
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_reg_preds = log_reg.predict(X_test_scaled)
log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n===== Logistic Regression Results =====")
print(classification_report(y_test, log_reg_preds, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, log_reg_probs), 4))

# -----------------------------------------------------------------
# 5. MODEL 2 - RANDOM FOREST (usually stronger for this kind of task)
# -----------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",   # again, compensate for class imbalance
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)   # trees don't need scaled data
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("\n===== Random Forest Results =====")
print(classification_report(y_test, rf_preds, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, rf_probs), 4))

# -----------------------------------------------------------------
# 6. CONFUSION MATRIX (which errors matter most?)
# -----------------------------------------------------------------
# In fraud detection, a False Negative (missed fraud) is usually far
# more costly than a False Positive (flagging a genuine transaction).
cm = confusion_matrix(y_test, rf_preds)
print("\nConfusion Matrix (Random Forest):")
print("                Predicted Genuine  Predicted Fraud")
print(f"Actual Genuine        {cm[0][0]:<15} {cm[0][1]}")
print(f"Actual Fraud          {cm[1][0]:<15} {cm[1][1]}")

# -----------------------------------------------------------------
# 7. FEATURE IMPORTANCE (which signals drive the model?)
# -----------------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=feature_names)
print("\nTop 5 most important features:")
print(importances.sort_values(ascending=False).head(5))

# -----------------------------------------------------------------
# 8. NEXT STEPS (ideas to extend this project)
# -----------------------------------------------------------------
# - Try SMOTE (imblearn) to oversample the minority (fraud) class.
# - Tune the classification threshold using the precision_recall_curve
#   instead of the default 0.5, since fraud detection often prioritizes
#   recall over precision.
# - Try XGBoost / LightGBM for potentially better performance.
# - Add a cost-based evaluation (e.g., cost of missed fraud vs cost of
#   a false alarm) instead of only accuracy/F1.
