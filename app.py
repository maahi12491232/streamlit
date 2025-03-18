import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Custom CSS for styling (optional)
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        margin: 0 auto;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
#
st.markdown(
    """
    <style>
    .image-container {
        border: 2px solid #ccc; /* Light gray border */
        border-radius: 10px;    /* Rounded corners */
        padding: 10px;         /* Padding inside the container */
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        margin-bottom: 20px;   /* Add some spacing below the image */
    }

    .image-container .stImage > img { /* Target images within Streamlit's image container */
        display: block;
        max-width: 100%;
        height: auto;
        border-radius: 8px;  /* Slightly rounded image corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
#
# -----------------------------------------------------------
# Initialize session state
# -----------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = "login"  # Possible values: "login", "prediction"
if 'login_error' not in st.session_state:
    st.session_state.login_error = False

# -----------------------------------------------------------
# Callback functions
# -----------------------------------------------------------
def login_callback():
    # Get the current username and password from session state keys used in text_input
    username = st.session_state.username_input
    password = st.session_state.password_input
    # Replace this logic with your actual validation mechanism
    if username == "admin" and password == "password":
        st.session_state.page = "prediction"
        st.session_state.login_error = False
    else:
        st.session_state.login_error = True

def logout_callback():
    st.session_state.page = "login"
    # Optionally clear username/password if desired:
    st.session_state.username_input = ""
    st.session_state.password_input = ""

# -----------------------------------------------------------
# UI Sections
# -----------------------------------------------------------

def login_section():
    st.markdown("<h1 style='text-align: center;'>🌿Sugar Cane Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Login to access the application</h3>", unsafe_allow_html=True)

    st.text_input("Username", key="username_input")
    st.text_input("Password", type="password", key="password_input")

    # The login button uses the callback so that one click is enough.
    st.button("Login", key="login_button", on_click=login_callback)

    if st.session_state.login_error:
        st.error("Incorrect username or password.")

def prediction_section():
    # Sidebar: Logout button using callback
    st.sidebar.button("Logout", key="logout_button", on_click=logout_callback)
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    model_path = st.sidebar.text_input("Enter the path to the trained model", "efficientnetB0_model.keras")

    @st.cache_resource
    def load_model(model_path):
        return tf.keras.models.load_model(model_path)

    @st.cache_data
    def load_and_preprocess_image(uploaded_image):
      image = Image.open(uploaded_image)
      image = image.resize((224, 224))
      img_array = np.array(image) / 255.0  # Normalize
      return image, img_array # Return both image and image array

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

    st.markdown("<h1 style='text-align: center;'>🌿Sugar Cane Disease Detection🌿</h1>", unsafe_allow_html=True)
    uploaded_images = st.file_uploader("Choose sugar cane leaf images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if 'model' not in st.session_state:
        st.session_state.model = load_model(model_path)
    model = st.session_state.model

    if uploaded_images and model_path:
        class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
        for uploaded_image in uploaded_images:
            image, img_array = load_and_preprocess_image(uploaded_image)
            img_array = img_array.reshape(1, 224, 224, 3)
            with st.spinner('Classifying...'):
                predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions, axis=1)[0]]
            confidence = np.max(predictions, axis=1)[0]

            # Display the image with label and styling
            with st.container():  # Wrap st.pyplot in a container
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                fig, ax = plt.subplots(1, figsize=(6, 4))
                ax.imshow(image)
                label_text = f"{predicted_class}: {confidence:.2f}"
                ax.text(10, 10, label_text, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            if confidence >= threshold:
                st.write(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
                with st.expander("Disease Information"):
                    info = disease_info.get(predicted_class, {})
                    for key, value in info.items():
                        st.write(f"**{key}:** {value}")
            else:
                st.write("Prediction: **Uncertain** (Confidence below threshold)")


# -----------------------------------------------------------
# Main App Flow
# -----------------------------------------------------------
if st.session_state.page == "login":
    login_section()
elif st.session_state.page == "prediction":
    prediction_section()
