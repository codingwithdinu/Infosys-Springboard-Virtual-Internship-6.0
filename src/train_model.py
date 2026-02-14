from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import pickle
from evaluate_model import evaluate_model, plot_confusion_matrix


# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("jason23322/high-accuracy-email-classifier")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])


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
