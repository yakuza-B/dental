import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "model/model.h5"  # Path to your trained model
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the model is in the 'model/' directory.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.convert("RGB")  # Ensure image is in RGB format
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict the class of the image
def predict(image, model, threshold=0.5):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]  # Get prediction probabilities

    # Log raw predictions for debugging
    st.write("### Raw Model Predictions (Confidence Scores)")
    confidence_data = [{"Condition": CLASS_LABELS[i], "Confidence": f"{prob:.2f}"} for i, prob in enumerate(predictions)]
    st.table(confidence_data)

    # Identify all conditions above the confidence threshold (multi-label support)
    detected_conditions = [
        (CLASS_LABELS[i], predictions[i]) for i in range(len(predictions)) if predictions[i] > threshold
    ]

    # If no class meets the threshold, return the highest probability class
    if not detected_conditions:
        max_index = np.argmax(predictions)
        detected_conditions = [(CLASS_LABELS[max_index], predictions[max_index])]

    return detected_conditions

# Map class indices to labels (ensure these match your dataset)
CLASS_LABELS = {
    0: "Cavity",
    1: "Fillings",
    2: "Impacted Tooth",
    3: "Implant"
}

# Streamlit app layout
def main():
    st.title("Teeth Condition Classifier 🦷")
    st.markdown("""
    Upload an image of teeth, and the model will predict whether it has **Cavity, Fillings, Impacted Tooth, or Implant**.
    """)

    # Load the model
    model = load_model()
    if model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image_uploaded = Image.open(uploaded_file)
            st.image(image_uploaded, caption="Uploaded Image", use_column_width=True)

            # Perform prediction
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    detected_conditions = predict(image_uploaded, model)

                    # Display the predicted conditions and confidence levels
                    if detected_conditions:
                        st.success("### Prediction Results:")
                        for condition, confidence in detected_conditions:
                            st.write(f"✅ **Condition**: {condition}")
                            st.write(f"📊 **Confidence**: {confidence:.2f}")
                    else:
                        st.warning("No conditions detected above the confidence threshold.")
        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image. Please upload a valid image file.")

if __name__ == "__main__":
    main()
