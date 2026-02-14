import pickle

# Load category model
category_model = pickle.load(open("models/email_classifier.pkl", "rb"))
category_vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Load urgency model
urgency_model = pickle.load(open("models/urgency_model.pkl", "rb"))
urgency_vectorizer = pickle.load(open("models/urgency_vectorizer.pkl", "rb"))


def predict_email(text):
    # Category prediction
    text_vector = category_vectorizer.transform([text])
    category = category_model.predict(text_vector)[0]
    category_conf = max(category_model.predict_proba(text_vector)[0]) * 100

    # Urgency prediction
    urgency_vector = urgency_vectorizer.transform([text])
    urgency = urgency_model.predict(urgency_vector)[0]
    urgency_conf = max(urgency_model.predict_proba(urgency_vector)[0]) * 100

    return category, category_conf, urgency, urgency_conf


if __name__ == "__main__":
    user_input = input("Enter email text: ")

    cat, cat_conf, urg, urg_conf = predict_email(user_input)

    print(f"Category: {cat} ({cat_conf:.2f}%)")
    print(f"Urgency: {urg} ({urg_conf:.2f}%)")
