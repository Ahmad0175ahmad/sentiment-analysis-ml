import pandas as pd
df=pd.read_csv("IMDB Dataset.csv")
df.head()

import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words =set(stopwords.words('english'))

def clean_text(text):
  text = re.sub(r'<.*?>','',text)
  text = re.sub(r'[^a-zA-Z]',' ',text)
  text =text.lower()
  words =text.split()
  words =[w for w in words if w not in stop_words]
  return " ".join(words)
df["cleaned"] =df["review"].apply(clean_text)
df.head()  

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


x =df["cleaned"]
y =df["sentiment"]

x_train, x_test, y_train, y_test =train_test_split(x,y, random_state=42, test_size=0.2)

vectorizer =TfidfVectorizer(max_features=20000, ngram_range=(1,2) )
x_train_vec =vectorizer.fit_transform(x_train)
x_test_vec =vectorizer.transform(x_test)
model =LogisticRegression(max_iter=500)
model.fit(x_train_vec,y_train)
y_pred =model.predict(x_test_vec)

print('Accuracy :', accuracy_score(y_test, y_pred))

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
