import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("IMDB Review Sentiment Analyzer")

review = st.text_area("Enter your movie review:")

if st.button("Analyze"):
    # Preprocess input
    review_clean = review.lower()
    review_vec = vectorizer.transform([review_clean])
    
    prediction = model.predict(review_vec)[0]
    label = "Positive 😊" if prediction == 1 else "Negative 😠"
    st.write(f"**Sentiment:** {label}")