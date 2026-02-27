# app.py
import streamlit as st
from model import predict_sentiment

st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")
st.title("Sentiment Analysis on Moives/videos Comments")
st.write("Enter any comment below, and the app will tell if it's Good or Bad and reply accordingly.")

# User input
user_input = st.text_area("Enter your comment here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        sentiment, reply = predict_sentiment(user_input)
        st.success(f"Sentiment: {sentiment.upper()}")
        st.info(f"Reply: {reply}")
    else:
        st.warning("Please enter some text!")