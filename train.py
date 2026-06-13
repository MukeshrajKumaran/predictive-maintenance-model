"""
TRAIN.PY — Model training and evaluation
==========================================

WHY RANDOM FOREST?
------------------
Random Forest is an ensemble of decision trees. Each tree is trained
on a random subset of data and features (bagging). Final prediction
is the majority vote across all trees.

For tabular sensor features it's near-optimal because:
  - Handles non-linear relationships between features and faults
  - Built-in feature importance ranking
  - Robust to outliers (unlike SVM)
  - No feature scaling needed (unlike neural nets)
  - Fast to train and interpret

In production, you'd deploy this model on the edge (STM32 + emlearn
or TensorFlow Lite) to run inference directly on the microcontroller.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import LabelEncoder

FAULT_NAMES = ["Normal", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]


def load_data():
    df = pd.read_csv("data/bearing_features.csv")
    feature_cols = [c for c in df.columns if c not in ["fault_type", "fault_name"]]
    X = df[feature_cols].values
    y = df["fault_type"].values
    return X, y, feature_cols


def train_and_evaluate():
    X, y, feature_names = load_data()

    # 80/20 train-test split, stratified to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}\n")

    # --- Train model ---
    model = RandomForestClassifier(
        n_estimators=100,      # 100 trees
        max_depth=None,        # grow full trees
        min_samples_split=2,
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=FAULT_NAMES))

    # --- Cross-validation (more reliable than single split) ---
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # --- Plots ---
    os.makedirs("outputs", exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=FAULT_NAMES, yticklabels=FAULT_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("\nSaved: outputs/confusion_matrix.png")

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(feature_names)), importances[indices], color="#4A90D9")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("Saved: outputs/feature_importance.png")

    # Save model + metadata
    joblib.dump(model, "outputs/model.pkl")
    joblib.dump(feature_names, "outputs/feature_names.pkl")
    print("Saved: outputs/model.pkl")

    return model, acc, cv_scores.mean()


if __name__ == "__main__":
    train_and_evaluate()
