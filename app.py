import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pickle

# Load your model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Spam Email Classifier")

# Example: Ask for 5 features – replace or update based on your model input
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")
f5 = st.number_input("Feature 5")

if st.button("Predict"):
    features = np.array([[f1, f2, f3, f4, f5]])  # Replace with actual number of features
    prediction = model.predict(features)[0]
    st.write("🔎 Prediction:", "SPAM" if prediction == 1 else "NOT SPAM")
