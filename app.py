import streamlit as st
import joblib

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("Amazon Review Sentiment Analysis")

review = st.text_area("Enter your review")

if st.button("Predict"):
    vec = tfidf.transform([review])
    result = model.predict(vec)

    if result == 1:
        st.success("Positive 😊")
    elif result == 0:
        st.warning("Neutral 😐")
    else:
        st.error("Negative 😡")