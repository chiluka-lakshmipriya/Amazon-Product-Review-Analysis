import streamlit as st
import joblib

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("Amazon Product Review Sentiment Analyzer")
st.write("Enter an Amazon product review below. The model will predict whether the review expresses a Positive, Neutral, or Negative sentiment.")

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
