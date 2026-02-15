import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import pickle
from evaluate_model import evaluate_model, plot_confusion_matrix

# -----------------------------
# Load local dataset
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "train_with_urgency.csv"))

# Split into train/test (80/20)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)


# -----------------------------
# Select features
# -----------------------------
X_train = train_df["text"]
y_train = train_df["category_id"]

X_test = test_df["text"]
y_test = test_df["category_id"]


# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------
# Train Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf, y_train)


# -----------------------------
# Training Accuracy
# -----------------------------
train_pred = log_model.predict(X_train_tfidf)
print("\nTraining Accuracy:", accuracy_score(y_train, train_pred))


# -----------------------------
# Evaluation
# -----------------------------
y_pred = evaluate_model(log_model, X_test_tfidf, y_test)
plot_confusion_matrix(y_test, y_pred)


# -----------------------------
# Cross Validation
# -----------------------------
cv_scores = cross_val_score(log_model, X_train_tfidf, y_train, cv=5)
print("\nCross Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


# -----------------------------
# Save model
# -----------------------------
pickle.dump(log_model, open("models/email_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("\nModel saved successfully.")
