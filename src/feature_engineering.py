from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def create_tfidf_features(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    features = vectorizer.fit_transform(corpus)
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    return features
