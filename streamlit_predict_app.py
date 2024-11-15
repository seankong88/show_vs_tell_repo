import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Download necessary NLTK resources (only needs to happen once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools for text processing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)  # Return preprocessed sentence as a string

# Load pre-trained models and vectorizers
def load_model_and_vectorizer():
    try:
        model = joblib.load('LogisticRegression_All_shots_data_model.pkl')
        vectorizer = joblib.load('LogisticRegression_All_shots_data_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

# Predict whether a sentence is "show" or "tell"
def predict_sentences(sentences, model, vectorizer):
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    X_test = vectorizer.transform(preprocessed_sentences)  # Vectorize the sentences
    predictions = model.predict(X_test)  # Make predictions
    return predictions

# Streamlit App
st.title("Show or Tell Prediction App")

# Text input area for user to enter data story
student_input = st.text_area("Paste your data story here:")

# Analyze button
if st.button("Analyze"):
    if student_input:
        # Split the input text into individual sentences
        sentences = nltk.sent_tokenize(student_input)

        # Load the model and vectorizer
        model, vectorizer = load_model_and_vectorizer()

        # Get predictions for each sentence
        try:
            predictions = predict_sentences(sentences, model, vectorizer)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            st.stop()

        # Display each sentence with its prediction
        for i, sentence in enumerate(sentences):
            st.write(f"Sentence: {sentence}")
            st.write(f"Prediction: {'Show' if predictions[i] == 0 else 'Tell'}")
            st.write("----")
    else:
        st.warning("Please enter a data story to analyze.")
