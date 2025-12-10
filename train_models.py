"""Train and evaluate multiple classifiers on the cleaned spam dataset.

Assumes the following artifacts exist (produced by data.py):
- tfidf_vectorizer.pkl
- X_train.npz, X_test.npz
- y_train.csv, y_test.csv
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load artifacts
vectorizer_path = "tfidf_vectorizer.pkl"
X_train_path = "X_train.npz"
X_test_path = "X_test.npz"
y_train_path = "y_train.csv"
y_test_path = "y_test.csv"

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

X_train = sp.load_npz(X_train_path)
X_test = sp.load_npz(X_test_path)
y_train = pd.read_csv(y_train_path).iloc[:, 0]
y_test = pd.read_csv(y_test_path).iloc[:, 0]

# Ensure y is 1D array
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

models = [
    ("LogisticRegression", LogisticRegression(max_iter=200, n_jobs=-1)),
    ("LinearSVC", LinearSVC()),
    ("MultinomialNB", MultinomialNB()),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)),
]

results = []

for name, model in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    })
    print(f"\n{name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

# Summary table
print("\n=== Model Comparison ===")
summary_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
print(summary_df.to_string(index=False, float_format="{:.4f}".format))

# Show classification report for the best model
best_model_name = summary_df.iloc[0]["model"]
best_model = next(m for n, m in models if n == best_model_name)
best_preds = best_model.predict(X_test)
print(f"\n=== Classification Report: {best_model_name} ===")
print(classification_report(y_test, best_preds, target_names=["ham", "spam"]))
