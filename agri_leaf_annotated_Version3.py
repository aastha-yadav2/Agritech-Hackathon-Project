"""
AgriLeaf: AI-Powered Plant Disease Detection Using CNN
Annotated Python script combining model training, inference, and Streamlit UI.

Author: Your Name
"""

# --- Imports ---
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

# For Streamlit web app
try:
    import streamlit as st
    from PIL import Image
except ImportError:
    st = None  # Streamlit not installed, ignore if only training/inference needed

# --- Hyperparameters & Paths ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10  # Reduce for sample/demo purposes
DATASET_DIR = "data/PlantVillage"      # Update as per your folder structure
MODEL_PATH = "models/agri_leaf_cnn.h5" # Where the trained model is saved
CLASS_INDICES_PATH = "models/class_indices.npy"
REMEDY_PATH = "data/disease_remedy.json"

# --- Data Preparation & Augmentation (Training) ---
def prepare_data():
    """
    Prepares data generators for training and validation using image augmentation.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    return train_gen, val_gen

# --- Model Definition ---
def build_cnn_model(num_classes):
    """
    Builds and returns a simple CNN model for image classification.
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Training Function ---
def train_model():
    """
    Trains the CNN model and saves it along with the class indices mapping.
    """
    train_gen, val_gen = prepare_data()
    model = build_cnn_model(train_gen.num_classes)
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    np.save(CLASS_INDICES_PATH, train_gen.class_indices)
    print("Model trained and saved.")

# --- Inference Function ---
def load_model_and_mapping():
    """
    Loads the trained model and class index mapping for inference.
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

def preprocess_image(img_path):
    """
    Loads and preprocesses an image for model prediction.
    """
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_remedy(disease_name):
    """
    Returns a remedy or advice string for a given disease.
    """
    if not os.path.exists(REMEDY_PATH):
        return "No remedy info available."
    with open(REMEDY_PATH, "r") as f:
        data = json.load(f)
    return data.get(disease_name, "No remedy info available.")

def predict_disease(img_path):
    """
    Performs inference on a single image and prints the result.
    """
    model, idx_to_class = load_model_and_mapping()
    img = preprocess_image(img_path)
    preds = model.predict(img)
    pred_idx = np.argmax(preds, axis=1)[0]
    disease = idx_to_class[pred_idx]
    confidence = preds[0][pred_idx]
    remedy = get_remedy(disease)
    print(f"Disease: {disease}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Remedy: {remedy}")

# --- Streamlit Web App ---
def run_streamlit():
    """
    Launches a Streamlit web app for uploading an image and getting predictions.
    """
    if st is None:
        print("Streamlit or PIL not installed. Please install them to run the web app.")
        return

    st.title("🌱 AgriLeaf: Plant Disease Detection")
    st.write("Upload a leaf image to detect disease and get remedy advice.")

    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Model loading (cache for performance)
        @st.cache_resource
        def get_model():
            return load_model_and_mapping()
        model, idx_to_class = get_model()
        # Preprocess and predict
        img_resized = img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
        img_array = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_input)
        pred_idx = np.argmax(preds, axis=1)[0]
        disease = idx_to_class[pred_idx]
        confidence = preds[0][pred_idx]
        remedy = get_remedy(disease)

        st.markdown(f"### Disease: **{disease}**")
        st.markdown(f"**Confidence:** {confidence:.2%}")
        st.markdown(f"**Remedy:** {remedy}")

# --- Main block for CLI usage ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Usage: python agri_leaf_annotated.py [train|predict|webapp] [image_path]
        action = sys.argv[1]
        if action == "train":
            train_model()
        elif action == "predict" and len(sys.argv) == 3:
            predict_disease(sys.argv[2])
        elif action == "webapp":
            run_streamlit()
        else:
            print("Usage:")
            print("  python agri_leaf_annotated.py train")
            print("  python agri_leaf_annotated.py predict path_to_image.jpg")
            print("  python agri_leaf_annotated.py webapp")
    else:
        print("No arguments provided. For help, run: python agri_leaf_annotated.py")