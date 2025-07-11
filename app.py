import streamlit as st
import joblib
import cv2
import numpy as np
from extract_features import extract_features

# Load the trained model
model = joblib.load("model.pkl")

st.title("💵 Fake Currency Detection App")
st.write("Upload a currency note image and detect whether it's Real or Fake.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Show image
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    # Extract features and predict
    features = extract_features("temp.jpg")
    if features is not None:
        prediction = model.predict([features])[0]
        label = "🟢 Real" if prediction == 0 else "🔴 Fake"
        st.subheader(f"Prediction: {label}")
    else:
        st.error("Could not process the image.")
