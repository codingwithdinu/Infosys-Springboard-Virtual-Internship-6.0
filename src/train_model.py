from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

# Load dataset
dataset = load_dataset("jason23322/high-accuracy-email-classifier")

train_data = dataset["train"]
test_data = dataset["test"]

# Convert to pandas
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Select features
X_train = train_df["text"]
y_train = train_df["category_id"]

X_test = test_df["text"]
y_test = test_df["category_id"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=50000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train_tfidf, y_train)

# Training Accuracy
train_pred = log_model.predict(X_train_tfidf)
print("\nTraining Accuracy:", accuracy_score(y_train, train_pred))

# Test Accuracy
log_pred = log_model.predict(X_test_tfidf)
print("Test Accuracy:", accuracy_score(y_test, log_pred))

# Classification Report
print("\nLogistic Regression Report:\n")
print(classification_report(y_test, log_pred))

# Cross Validation
cv_scores = cross_val_score(log_model, X_train_tfidf, y_train, cv=5)
print("\nCross Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Save model
pickle.dump(log_model, open("models/email_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("\nModel saved successfully.")
