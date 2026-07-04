import streamlit as st
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Page title
st.title("🛒 Amazon Product Review Sentiment Analyzer")

# Project description
st.markdown("""
Welcome! 👋

Enter an **Amazon product review** below. This application uses a **Machine Learning model (Logistic Regression)** to predict whether the review expresses a **Positive**, **Neutral**, or **Negative** sentiment.
""")

# Input box
review = st.text_area("✍️ Enter your review here")

# Prediction
if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review before clicking Predict.")
    else:
        vec = tfidf.transform([review])
        result = model.predict(vec)

        if result == 1:
            st.success("😊 Sentiment: Positive")
        elif result == 0:
            st.info("😐 Sentiment: Neutral")
        else:
            st.error("😡 Sentiment: Negative")

# Footer
st.markdown("---")
st.markdown("**Developed by:** Chiluka Laxmi Priya")
