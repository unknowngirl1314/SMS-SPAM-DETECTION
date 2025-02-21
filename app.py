import nltk
import os
import string
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources if not already present
if not os.path.exists(os.path.join(nltk.data.path[0], 'corpora', 'stopwords')):
    nltk.download('stopwords')
if not os.path.exists(os.path.join(nltk.data.path[0], 'tokenizers', 'punkt')):
    nltk.download('punkt')

# Initialize the PorterStemmer
ps = PorterStemmer()


# Function to preprocess and transform the text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tokens = [i for i in tokens if i.isalnum()]  # Remove non-alphanumeric characters
    tokens = [i for i in tokens if
              i not in stopwords.words('english') and i not in string.punctuation]  # Remove stopwords and punctuation
    tokens = [ps.stem(i) for i in tokens]  # Apply stemming
    return " ".join(tokens)


# Load the vectorizer and model
#try:
 #   tk = pickle.load(open("vectorizer.pkl", 'rb'))  # Load vectorizer
  #  model = pickle.load(open("model.pkl", 'rb'))  # Load model
#except FileNotFoundError as e:
 #   st.error(f"Error: {e}")

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Streamlit app UI
st.title("SMS Spam Detection Model")
#st.write("*Made with ❤️‍🔥 by Shrudex👨🏻‍💻*")

# Input field for SMS
input_sms = st.text_input("Enter the SMS")

# Predict button
if st.button('Predict'):
    if input_sms:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the input text using the loaded vectorizer
        vector_input = vectorizer.transform([transformed_sms])

        # 3. Make prediction using the loaded model
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter an SMS message to classify.")
