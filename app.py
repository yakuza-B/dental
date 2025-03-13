import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import requests

# Define the direct download URL for the model file
MODEL_URL = "https://drive.google.com/uc?id=1BuV2xdN8UEcf1ILWHLBxu3845821HGao"  # Replace with your actual file ID
MODEL_PATH = "dental_classification_model.h5"

# Download the model file if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        try:
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.content)
                print("Model downloaded successfully.")
            else:
                print(f"Failed to download the model. HTTP Status Code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while downloading the model: {e}")

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    download_model()  # Ensure the model is downloaded
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please check the download process.")
        st.stop()
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

# Preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Map class indices to class names
def get_class_name(pred_idx, valid_classes):
    return valid_classes[pred_idx]

# Streamlit app
def main():
    st.title("Dental Image Classification")
    st.write("Upload an image to classify whether it's an Implant, Filling, Impacted Tooth, or Cavity.")
    
    valid_classes = ['Implant', 'Fillings', 'Impacted Tooth', 'Cavity']
    model = load_trained_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = get_class_name(predicted_idx, valid_classes)
        confidence = np.max(predictions) * 100
        
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
