import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/processed/train_with_urgency.csv")

X = df["text"]
y = df["urgency"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Logistic Regression
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
pickle.dump(model, open("models/urgency_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/urgency_vectorizer.pkl", "wb"))

print("\nUrgency model saved successfully.")
