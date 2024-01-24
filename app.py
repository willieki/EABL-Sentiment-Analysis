# app.py
import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load(xgb_model'best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
def main():
    st.title('Sentiment Analysis App')
    text = st.text_area('Enter text for sentiment analysis:', '')

    if st.button('Analyze'):
        sentiment = predict_sentiment(text)
        st.write('Sentiment:', sentiment)

def predict_sentiment(text):
    # Preprocess the input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])

    # Use the loaded model to predict sentiment
    sentiment = model.predict(text_vectorized)[0]

    # Adjust this based on your specific model and preprocessing steps
    return 'Positive' if sentiment == 1 else 'Negative'

if __name__ == '__main__':
    main()
