import streamlit as st
import numpy as np
from PIL import Image
import pickle
import requests
from io import BytesIO

try:
    model = pickle.load(
        open("/mount/src/emotionanalysisproject/models/model.pkl", "rb")
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Streamlit app layout
st.title("Emotion Classification App")
st.write("Upload an image to classify the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to 48x48
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    # Display prediction
    st.write(f"Predicted Emotion: **{predicted_emotion}**")
