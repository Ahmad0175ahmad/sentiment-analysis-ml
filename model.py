import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()

def predict_sentiment(text):
    cleaned = clean_text(text)   # 🔥 THIS WAS MISSING
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]

    reply = (
        "Thank you for your feedback" 
        if prediction == "positive" 
        else "Sorry for the inconvenience"
    )

    return prediction, reply