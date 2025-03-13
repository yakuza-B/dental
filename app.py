import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('dental_classification_model.h5')  # Replace with your model path
    return model

# Preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    # Resize the image
    img = image.resize(target_size)
    
    # Convert to NumPy array and normalize
    img = np.array(img).astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Map class indices to class names
def get_class_name(pred_idx, valid_classes):
    return valid_classes[pred_idx]

# Streamlit app
def main():
    st.title("Dental Image Classification")
    st.write("Upload an image to classify whether it's an Implant, Filling, Impacted Tooth, or Cavity.")
    
    # Define valid classes (update based on your dataset)
    valid_classes = ['Implant', 'Fillings', 'Impacted Tooth', 'Cavity']
    
    # Load the model
    model = load_trained_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make predictions
        predictions = model.predict(processed_image)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = get_class_name(predicted_idx, valid_classes)
        confidence = np.max(predictions) * 100
        
        # Display the result
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
