import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

# ==============================
# 1. Load Model and Class Labels
# ==============================

@st.cache_resource
def load_model():
    """Loads the trained model."""
    model_path = "model/model.h5"
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

def load_class_labels(df):
    """Dynamically generates class labels from the dataset."""
    unique_classes = sorted(df['class'].unique())
    return {i: cls for i, cls in enumerate(unique_classes)}

# Define paths
train_path = r"C:\Users\barry\Documents\ip3\train"
train_csv = os.path.join(train_path, "_annotations.csv")

# Load annotations to generate class labels
df_train = pd.read_csv(train_csv) if os.path.exists(train_csv) else None
CLASS_LABELS = load_class_labels(df_train) if df_train is not None else {}

# ==============================
# 2. Preprocessing and Prediction
# ==============================

def preprocess_image(img):
    """Preprocesses the uploaded image for prediction."""
    img = img.convert("RGB")  # Ensure image is in RGB format
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(image, model, threshold=0.5):
    """Predicts the class of the image."""
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]  # Get prediction probabilities

    # Debugging: Print predictions and CLASS_LABELS
    print("Predictions:", predictions)
    print("CLASS_LABELS:", CLASS_LABELS)

    # Log raw predictions for debugging
    confidence_data = [
        {"Condition": CLASS_LABELS.get(i, f"Unknown Class {i}"), "Confidence": f"{prob:.2f}"}
        for i, prob in enumerate(predictions)
    ]
    st.table(confidence_data)

    # Identify all conditions above the confidence threshold
    detected_conditions = [
        (CLASS_LABELS.get(i, f"Unknown Class {i}"), predictions[i])
        for i in range(len(predictions)) if predictions[i] > threshold
    ]

    # If no class meets the threshold, return the highest probability class
    if not detected_conditions:
        max_index = np.argmax(predictions)
        detected_conditions = [(CLASS_LABELS.get(max_index, f"Unknown Class {max_index}"), predictions[max_index])]

    return detected_conditions

# ==============================
# 3. Streamlit App Layout
# ==============================

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
