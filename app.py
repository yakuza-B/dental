import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# 1. Dataset Preparation
# ==============================

def load_annotations(csv_file):
    """Loads annotations from a CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    return pd.read_csv(csv_file)

def count_images_in_split(df):
    """Counts unique images in a split."""
    return len(df['filename'].unique())

def visualize_random_images(df, image_folder, num_images=9):
    """Visualizes random original images from the dataset."""
    unique_images = df['filename'].unique()
    random_images = random.sample(list(unique_images), min(num_images, len(unique_images)))

    fig, axes = plt.subplots(len(random_images), 4, figsize=(12, len(random_images) * 3))
    for i, img_name in enumerate(random_images):
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load {img_name}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Display images
        axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title("Grayscale")
        axes[i, 2].imshow(adaptive_thresh, cmap='gray')
        axes[i, 2].set_title("Adaptive Threshold")
        axes[i, 3].imshow(blurred, cmap='gray')
        axes[i, 3].set_title("Gaussian Blur")

    plt.tight_layout()
    plt.show()

# Define paths
train_path = r"C:\Users\barry\Documents\ip3\train"
train_csv = os.path.join(train_path, "_annotations.csv")
test_path = r"C:\Users\barry\Documents\ip3\test"
test_csv = os.path.join(test_path, "_annotations.csv")
valid_path = r"C:\Users\barry\Documents\ip3\valid"
valid_csv = os.path.join(valid_path, "_annotations.csv")

# Load annotations
df_train = load_annotations(train_csv)
df_test = load_annotations(test_csv)
df_valid = load_annotations(valid_csv)

# Display unique classes in the training set
unique_classes = df_train['class'].unique()
print("Unique classes in training data:", unique_classes)

# Count images in each split
print("Number of images in training set:", count_images_in_split(df_train))
print("Number of images in testing set:", count_images_in_split(df_test))
print("Number of images in validation set:", count_images_in_split(df_valid))

# Visualize random images
visualize_random_images(df_train, train_path)

# ==============================
# 2. Data Preprocessing
# ==============================

def prepare_dataset(df, image_folder, batch_size=32, img_size=(224, 224), subset=None):
    """Prepares the dataset for training, validation, or testing."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2 if subset else 0.0
    )
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_folder,
        x_col="filename",
        y_col="class",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset=subset
    )
    return generator

# Prepare datasets
train_generator = prepare_dataset(df_train, train_path, subset="training")
val_generator = prepare_dataset(df_train, train_path, subset="validation")
test_generator = prepare_dataset(df_test, test_path)

# Compute class weights
def compute_class_weights(df):
    """Computes class weights for handling class imbalance."""
    class_labels = pd.Categorical(df['class']).codes
    unique_classes = np.unique(class_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=class_labels)
    return dict(zip(unique_classes, class_weights))

class_weights = compute_class_weights(df_train)

# ==============================
# 3. Model Creation and Training
# ==============================

def create_model(num_classes):
    """Creates a MobileNetV2-based classification model."""
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Create and train the model
num_classes = len(unique_classes)
model = create_model(num_classes)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    verbose=1,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# Save the trained model
os.makedirs("model", exist_ok=True)
model.save("model/model.h5")
print("Model saved successfully.")

# ==============================
# 4. Evaluation
# ==============================

def evaluate_model(model, test_generator):
    """Evaluates the model on the test set."""
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

evaluate_model(model, test_generator)

# ==============================
# 5. Streamlit Deployment
# ==============================

import streamlit as st
from PIL import Image, UnidentifiedImageError

@st.cache_resource
def load_model():
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

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image, model, threshold=0.5):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]

    confidence_data = [{"Condition": CLASS_LABELS[i], "Confidence": f"{prob:.2f}"} for i, prob in enumerate(predictions)]
    st.table(confidence_data)

    detected_conditions = [(CLASS_LABELS[i], predictions[i]) for i in range(len(predictions)) if predictions[i] > threshold]
    if not detected_conditions:
        max_index = np.argmax(predictions)
        detected_conditions = [(CLASS_LABELS[max_index], predictions[max_index])]
    return detected_conditions

CLASS_LABELS = {
    0: "Cavity",
    1: "Fillings",
    2: "Impacted Tooth",
    3: "Implant"
}

def main():
    st.title("Teeth Condition Classifier 🦷")
    st.markdown("""
    Upload an image of teeth, and the model will predict whether it has **Cavity, Fillings, Impacted Tooth, or Implant**.
    """)

    model = load_model()
    if model is None:
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image_uploaded = Image.open(uploaded_file)
            st.image(image_uploaded, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    detected_conditions = predict(image_uploaded, model)

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
