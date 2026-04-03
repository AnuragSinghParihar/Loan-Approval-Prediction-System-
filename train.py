import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess

print("Loading dataset from loan_test_data_10k.csv...")
df = pd.read_csv("loan_test_data_10k.csv")

# Ensure there are no nulls in notes
df = df.dropna(subset=["Notes"])

# Auto-labeling logic since the CSV does not contain a "Label/Status" column.
# We map known negative signals to 0 (Rejected) and positive signals to 1 (Approved).
negative_keywords = [
    "default", "poor repayment", "bad credit", "no income", "unemployed", 
    "missed emi", "overdue", "defaulter", "high debt", "instability"
]

def auto_label(text):
    text_lower = text.lower()
    for word in negative_keywords:
        if word in text_lower:
            return 0  # Rejected
    return 1  # Approved

df["label"] = df["Notes"].apply(auto_label)

print(f"Total dataset size: {len(df)}")
print(f"Distribution: \\n{df['label'].value_counts()}")

# Clean text
print("Preprocessing text...")
df["clean_text"] = df["Notes"].apply(preprocess)

# TF-IDF
print("Vectorizing...")
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1500)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train ML Model
print("Training Logistic Regression Model...")
model = LogisticRegression(class_weight="balanced", max_iter=1000, C=5.0)

# Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"  CV Mean Accuracy: {cv_scores.mean():.4f}")

# Final Fit
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"  Test Accuracy   : {acc:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

# Save everything
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/loan_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
print("\\n✅ Model and Vectorizer trained on 10K dataset and saved!")
