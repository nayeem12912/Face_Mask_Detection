import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import os

# Paths (update if files are in a subfolder, e.g., 'models/face_mask_model_final.keras')
model_path = 'face_mask_model_final.keras'
class_indices_path = 'face_mask_class_indices.json'

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    with open(class_indices_path) as f:
        idx = json.load(f)
    inv_map = {v: k for k, v in idx.items()}
    return model, inv_map

model, inv_map = load_model()

# Prediction function
def predict(img_path):
    IMG_SIZE = (224, 224)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    p = model.predict(x)[0][0]
    label_idx = 0 if p < 0.5 else 1
    label = inv_map[label_idx]
    conf = (1 - p) if label_idx == 0 else p
    return label, conf

# Streamlit UI
st.title("Face Mask Detection Demo")
st.write("Upload an image to check if the person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = os.path.join(".", uploaded_file.name)  # Save in current directory
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.image(temp_path, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    with st.spinner("Predicting..."):
        label, conf = predict(temp_path)
        st.success(f"Prediction: **{label}** (Confidence: {conf:.2f})")
    
    # Clean up temp file
    os.remove(temp_path)