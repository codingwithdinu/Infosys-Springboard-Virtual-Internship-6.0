from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def create_vectorizer(max_features=5000):
    return TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )

def fit_vectorizer(vectorizer, X_train):
    return vectorizer.fit_transform(X_train)

def transform_vectorizer(vectorizer, X_test):
    return vectorizer.transform(X_test)

def save_vectorizer(vectorizer, path="models/vectorizer.pkl"):
    pickle.dump(vectorizer, open(path, "wb"))

def load_vectorizer(path="models/vectorizer.pkl"):
    return pickle.load(open(path, "rb"))
