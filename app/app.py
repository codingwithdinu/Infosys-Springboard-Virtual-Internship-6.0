import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# -------------------------------
# Base Directory
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------
# Load Models
# -------------------------------
category_model = pickle.load(
    open(os.path.join(BASE_DIR, "models", "email_classifier.pkl"), "rb")
)

category_vectorizer = pickle.load(
    open(os.path.join(BASE_DIR, "models", "vectorizer.pkl"), "rb")
)

urgency_model = pickle.load(
    open(os.path.join(BASE_DIR, "models", "urgency_model.pkl"), "rb")
)

urgency_vectorizer = pickle.load(
    open(os.path.join(BASE_DIR, "models", "urgency_vectorizer.pkl"), "rb")
)

# -------------------------------
# Category Mapping (Important)
# -------------------------------
CATEGORY_MAPPING = {
    0: "Academic",
    1: "Complaint",
    2: "Request",
    3: "Feedback",
    4: "Spam",
    5: "General"
}

# -------------------------------
# Store Prediction History
# -------------------------------
prediction_history = []


# -------------------------------
# Home Route
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# Predict Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email"]

    # Category prediction
    cat_vector = category_vectorizer.transform([email_text])
    category_id = category_model.predict(cat_vector)[0]
    category = CATEGORY_MAPPING.get(category_id, "Unknown")
    cat_conf = max(category_model.predict_proba(cat_vector)[0]) * 100

    # Urgency prediction
    urg_vector = urgency_vectorizer.transform([email_text])
    urgency = urgency_model.predict(urg_vector)[0]
    urg_conf = max(urgency_model.predict_proba(urg_vector)[0]) * 100

    result = {
        "email": email_text[:80] + "..." if len(email_text) > 80 else email_text,
        "category": category,
        "cat_conf": round(cat_conf, 2),
        "urgency": urgency,
        "urg_conf": round(urg_conf, 2)
    }

    prediction_history.append(result)

    return render_template(
        "dashboard.html",
        result=result,
        history=prediction_history
    )


if __name__ == "__main__":
    app.run(debug=True)
