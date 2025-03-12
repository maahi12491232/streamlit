import streamlit as st
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
from PIL import Image

# Function to load and preprocess the uploaded image
def load_and_preprocess_image(uploaded_image):
    # Open the image
    img = Image.open(uploaded_image)

    # Resize the image to match the input size of the model (e.g., 224x224 for a CNN model)
    img = img.resize((224, 224))

    # Convert the image to a numpy array and normalize the pixel values
    img_array = np.array(img) / 255.0  # Normalize the pixel values between 0 and 1

    # Expand dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app interface
st.title("Sugar Cane Disease Detection")
st.sidebar.header("Upload Image")

# Upload the image
uploaded_image = st.sidebar.file_uploader("Choose a sugar cane leaf image", type=["jpg", "png", "jpeg"])

# Load the pre-trained model (if available)
model_path = st.sidebar.text_input("Enter the path to the trained model", "efficientnetB0_model.h5")

# Disease information dictionary
disease_info = {
    "Rust": {
        "Cause": "Fungus called Puccinia melanocephala. Thrives in warm, humid conditions.",
        "Symptoms": "Yellowish-brown pustules on leaves, stunted growth.",
        "Suggestions": "Ensure proper crop spacing, avoid over-irrigation, consider resistant varieties or fungicides."
    },
    "RedRot": {
        "Cause": "Fungal disease caused by Colletotrichum falcatum. Often due to waterlogging or infected material.",
        "Symptoms": "Reddish discoloration of internodes, white spots, sour smell.",
        "Suggestions": "Use disease-free planting material, improve drainage, apply fungicides."
    },
    "Mosaic": {
        "Cause": "Viral infection, often transmitted by aphids or contaminated tools.",
        "Symptoms": "Irregular patches of light and dark green on leaves, reduced photosynthesis.",
        "Suggestions": "Control aphids, sterilize tools, use resistant varieties."
    },
    "Healthy": {
        "Cause": "No disease detected.",
        "Symptoms": "Leaves are healthy with no visible issues.",
        "Suggestions": "Maintain proper care for the plant."
    },
    "Yellow": {
        "Cause": "Potential nutrient deficiency or other minor stress factors.",
        "Symptoms": "Yellowish discoloration of leaves.",
        "Suggestions": "Check soil nutrients and improve fertilization practices."
    }
}

if uploaded_image is not None and model_path:
    # Load the pre-trained model
    model = load_model(model_path)

    # Preprocess the image
    img_array = load_and_preprocess_image(uploaded_image)

    # Perform prediction
    predictions = model.predict(img_array)

    # Get the predicted class (if you have class names, use them here)
    class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
    predicted_class = class_names[np.argmax(predictions, axis=1)[0]]

    # Display results
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    st.write(f"Prediction: {predicted_class} (Class {np.argmax(predictions, axis=1)[0]})")
    st.write(f"Raw Prediction Scores: {predictions}")

    # Display disease-specific information
    st.subheader("Disease Information")
    info = disease_info.get(predicted_class, {})
    for key, value in info.items():
        st.write(f"**{key}:** {value}")
