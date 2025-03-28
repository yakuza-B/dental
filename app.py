import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

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
def predict(image, model):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]  # Get prediction probabilities

    # Log raw predictions for debugging
    st.write("### Raw Model Predictions (Confidence Scores)")
    for i, prob in enumerate(predictions):
        st.write(f"**{CLASS_LABELS[i]}**: {prob:.2f}")

    # Identify the predicted class with the highest confidence
    max_index = np.argmax(predictions)
    predicted_class = CLASS_LABELS[max_index]
    confidence = predictions[max_index]

    return predicted_class, confidence

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
    st.write("Upload an image of teeth, and the model will predict whether it has **Cavity, Fillings, Impacted Tooth, or Implant.**")

    # Load the model
    model = load_model()
    if model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image_uploaded = Image.open(uploaded_file)
        st.image(image_uploaded, caption="Uploaded Image", use_column_width=True)

        # Perform prediction
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                predicted_class, confidence = predict(image_uploaded, model)

                # Display the predicted condition and confidence level
                st.success("### Prediction Result:")
                st.write(f"✅ **Condition**: {predicted_class}")
                st.write(f"📊 **Confidence**: {confidence:.2f}")

if __name__ == "__main__":
    main()
